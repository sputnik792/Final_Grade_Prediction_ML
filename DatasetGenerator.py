import javalang
import pandas as pd
import numpy as np
import sys
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler


class DataGenerator:
    def __init__(self, base_path='./', max_problems=5000, max_tokens=1000,
                 max_number_of_sequences=32, max_sequence_length=50):

        self.base_path = base_path
        self.max_problems = max_problems
        self.max_number_sequences = max_number_of_sequences
        self.max_sequence_length = max_sequence_length

        self.code_file_train = base_path + 'Train/Data/CodeStates/CodeStates.csv'
        self.early_problems_train_file = base_path + 'Train/early.csv'
        self.main_table_train_file = base_path + 'Train/Data/MainTable.csv'
        self.link_table_train = base_path + 'Train/Data/LinkTables/Subject.csv'

        self.code_file_test = base_path + 'Test/Data/CodeStates/CodeStates.csv'
        self.early_problems_test_file = base_path + 'Test/early.csv'
        self.main_table_test_file = base_path + 'Test/Data/MainTable.csv'
        self.link_table_test = base_path + 'Test/Data/LinkTables/Subject.csv'

        self.vectorizer = tf.keras.layers.TextVectorization(max_tokens=max_tokens,
                                                            output_sequence_length=self.max_sequence_length, )
        self.sequences = []
        self.child_map = []

        self.use_features = True

        self.scaler = None
        if self.use_features:
            self.scaler = MinMaxScaler()

        self.record = None
        self.train_data = None
        self.train_labels = None
        self.train_children = None

        self.test_data = None
        self.test_labels = None
        self.test_children = None

    def generate_mapping(self, ast):
        sequence = []

        for i, values in enumerate(ast):
            path, node = values

            sequence.append(str(node))

        self.child_map.append(self.get_child_map(ast))
        self.vectorizer.adapt(sequence)
        self.sequences.append(sequence)

    @staticmethod
    def get_child_map(root):
        children = []

        parent_index = 0
        depth0 = 0
        depth1 = 0
        for path, node in root:
            """if depth0 >= self.max_number_sequences:
                break"""
            children.append([])

            child_index = parent_index + 1
            for _, child in node:
                """if depth1 >= self.max_number_sequences:
                    break"""
                children[parent_index].append(child_index - 1)
                child_index += 1
                depth1 += 1
            parent_index += 1
            depth0 += 1
        return children

    def generate_train_data(self):
        train_code = pd.read_csv(self.code_file_train)
        main_table_train = pd.read_csv(self.main_table_train_file)
        link_table = pd.read_csv(self.link_table_train)

        grouped_main_table = main_table_train.groupby(['SubjectID'])

        train_data = None
        child_data = None

        labels = []
        students = []

        max_problems = self.max_problems
        count = 1
        for name, group in grouped_main_table:
            print(f'Starting {name}\'s data')
            by_problem = group.groupby('ProblemID')

            for problem, problem_group in by_problem:
                for i, row in problem_group.iterrows():
                    if not row['Compile.Result'] == 'Success':
                        continue

                    code_state = row['CodeStateID']
                    code = str(train_code.loc[train_code['CodeStateID'] == code_state]['Code'].values[0])
                    code = 'package test;/**\n@author nil**/\nclass ' \
                           + "".join(filter(str.isalpha, str(row['SubjectID']))) + '{\n\r' + code + '}'

                    try:  # We should actually parse the tree line by line
                        code_tree = javalang.parse.parse(code)
                    except Exception:
                        continue

                    self.generate_mapping(code_tree)

            count += 1
            print(f'finished processing {count} student(s)')
            grade = link_table[link_table['SubjectID'] == name]['X-Grade'].values[0]
            labels.append(grade)
            students.append(name)

            stack = None
            for sequence in self.sequences:
                vec = self.vectorizer(sequence)

                if stack is None:
                    stack = vec
                    continue
                stack = np.vstack([stack, vec])

                if stack.shape[0] >= max_problems:
                    break

            if stack.shape[0] >= max_problems:
                diff = stack.shape[0] - max_problems
                stack = stack[:stack.shape[0]-diff]
            else:
                diff = max_problems - stack.shape[0]
                stack = np.vstack([stack, np.empty(shape=(diff, stack.shape[1]))])
            stack = np.expand_dims(stack, 0)

            if train_data is None:
                train_data = stack
            if train_data is not None and stack is not None:
                train_data = np.vstack([train_data, stack])

            stack = None
            self.child_map = [pad_sequences(v, maxlen=self.max_sequence_length, value=0, truncating='post',
                                            padding='post') for v in self.child_map]
            for child in self.child_map:
                if stack is None:
                    stack = child
                    continue
                stack = np.vstack([stack, child])

                if stack.shape[0] >= max_problems:
                    break

            self.sequences = []
            self.child_map = []

            if stack.shape[0] >= max_problems:
                diff = stack.shape[0] - max_problems
                stack = stack[:stack.shape[0] - diff]
            else:
                diff = max_problems - stack.shape[0]
                stack = np.vstack([stack, np.empty(shape=(diff, stack.shape[1]))])
            stack = np.expand_dims(stack, 0)

            if child_data is None:
                child_data = stack
            if child_data is not None and stack is not None:
                child_data = np.vstack([child_data, stack])

            if count > 5:
                break

        self.train_data = train_data[1:]
        self.train_labels = np.array(labels)
        self.train_children = np.array(child_data[1:max_problems])

        record = pd.DataFrame(columns=['Student_ID', 'Label'])
        record['Student_ID'] = students
        record['Label'] = labels
        self.record = record

        loc = 'SpringData'

        if self.base_path == './Spring_data/F19_Release_Train_06-28-21/':
            loc = 'FallData'

        with open('Data_v2/'+loc+'/nodes_train.npy', 'wb') as f:
            np.save(f, train_data)
        with open('Data_v2/'+loc+'/children_train.npy', 'wb') as f:
            np.save(f, self.train_children)
        with open('Data_v2/'+loc+'/labels_regression_train.npy', 'wb') as f:
            np.save(f, labels)

        record.to_csv('Data_v2/' + loc + '/record_train.csv')

    def generate_test_data(self):
        train_code = pd.read_csv(self.code_file_test)
        main_table_train = pd.read_csv(self.main_table_test_file)
        link_table = pd.read_csv(self.link_table_test)

        grouped_main_table = main_table_train.groupby(['SubjectID'])

        train_data = None
        child_data = None

        labels = []
        students = []

        max_problems = self.max_problems
        count = 1
        for name, group in grouped_main_table:
            print(f'Starting {name}\'s data')
            by_problem = group.groupby('ProblemID')

            for problem, problem_group in by_problem:
                for i, row in problem_group.iterrows():
                    if not row['Compile.Result'] == 'Success':
                        continue

                    code_state = row['CodeStateID']
                    code = str(train_code.loc[train_code['CodeStateID'] == code_state]['Code'].values[0])
                    code = 'package test;/**\n@author nil**/\nclass ' \
                           + "".join(filter(str.isalpha, str(row['SubjectID']))) + '{\n\r' + code + '}'

                    try:  # We should actually parse the tree line by line
                        code_tree = javalang.parse.parse(code)
                    except Exception:
                        continue

                    self.generate_mapping(code_tree)

            count += 1
            print(f'finished processing {count} student(s)')
            # grade = link_table[link_table['SubjectID'] == name]['X-Grade'].values[0]
            labels.append(0)
            students.append(name)

            stack = None
            for sequence in self.sequences:
                vec = self.vectorizer(sequence)

                if stack is None:
                    stack = vec
                    continue
                stack = np.vstack([stack, vec])

                if stack.shape[0] >= max_problems:
                    break

            if stack.shape[0] >= max_problems:
                diff = stack.shape[0] - max_problems
                stack = stack[:stack.shape[0] - diff]
            else:
                diff = max_problems - stack.shape[0]
                stack = np.vstack([stack, np.empty(shape=(diff, stack.shape[1]))])
            stack = np.expand_dims(stack, 0)

            if train_data is None:
                train_data = stack
            if train_data is not None and stack is not None:
                train_data = np.vstack([train_data, stack])

            stack = None
            self.child_map = [pad_sequences(v, maxlen=self.max_sequence_length, value=0, truncating='post',
                                            padding='post') for v in self.child_map]
            for child in self.child_map:
                if stack is None:
                    stack = child
                    continue
                stack = np.vstack([stack, child])

                if stack.shape[0] >= max_problems:
                    break

            self.sequences = []
            self.child_map = []

            if stack.shape[0] >= max_problems:
                diff = stack.shape[0] - max_problems
                stack = stack[:stack.shape[0] - diff]
            else:
                diff = max_problems - stack.shape[0]
                stack = np.vstack([stack, np.empty(shape=(diff, stack.shape[1]))])
            stack = np.expand_dims(stack, 0)

            if child_data is None:
                child_data = stack
            if child_data is not None and stack is not None:
                child_data = np.vstack([child_data, stack])

        self.test_data = train_data[1:]
        self.test_labels = np.array(labels)
        self.test_children = np.array(child_data[1:max_problems])

        record = pd.DataFrame(columns=['Student_ID', 'Label'])
        record['Student_ID'] = students
        record['Label'] = labels
        self.record = record

        loc = 'SpringData'

        if self.base_path == './Spring_data/F19_Release_Train_06-28-21/':
            loc = 'FallData'

        with open('Data_v2/' + loc + '/nodes_test.npy', 'wb') as f:
            np.save(f, train_data)
        with open('Data_v2/' + loc + '/children_test.npy', 'wb') as f:
            np.save(f, self.train_children)
        with open('Data_v2/' + loc + '/labels_regression_test.npy', 'wb') as f:
            np.save(f, labels)

        record.to_csv('Data_v2/' + loc + '/record_test.csv')

    def load_train_data(self, semester='spring'):
        base_path = './Data_v2/SpringData/'

        if semester == 'fall':
            base_path = './Data_v2/FallData/'

        with open(base_path + 'nodes_train.npy', 'rb') as f:
            vectorized = np.load(f)
        with open(base_path + 'children_train.npy', 'rb') as f:
            children = []  # np.array(np.load(f, allow_pickle=True))
        with open(base_path + 'labels_regression_train.npy', 'rb') as f:
            labels = np.array(np.load(f))
        self.record = pd.read_csv(base_path+'record_train.csv')
        return vectorized, children, labels

    def load_test_data(self, semester='spring'):
        base_path = './Data_v2/SpringData/'

        if semester == 'fall':
            base_path = './Data_v2/FallData/'

        with open(base_path + 'nodes_test.npy', 'rb') as f:
            vectorized = np.load(f)
        with open(base_path + 'children_test.npy', 'rb') as f:
            children = []  # np.load(f, allow_pickle=True)
        with open(base_path + 'labels_regression_test.npy', 'rb') as f:
            labels = np.load(f)
        self.record = pd.read_csv(base_path + 'record_test.csv')
        return vectorized, children, labels

    @staticmethod
    def _generate_batch(vectorized, children, labels, batch_size):
        start = 0
        batches = []
        for i in range(int(len(vectorized) / (batch_size + 1))):
            v = vectorized[start:start + batch_size]
            c = children[start:start + batch_size]
            y = labels[start:start + batch_size]

            start += batch_size
            batches.append((v, c, y))

        for b in batches:
            yield b

    def generate_batch(self, batch_size=1, training=True):
        if training:
            v, c, y = self.load_train_data()
        else:
            v, c, y = self.load_test_data()

        return self._generate_batch(v, c, y, batch_size)

    def generate_features(self, semester='spring', data='train'):
        file_name = 'spring_train_features.csv'

        if semester == 'fall' and data == 'train':
            file_name = 'fall_features_train.csv'

        if not data == 'train':
            file_name = 'fall_test_features.csv'

        features = pd.read_csv(file_name)
        grouped_features = features.groupby('SubjectID')

        columns = ['n1AvgFinComp', 'n2AvgFinComp', 'N1AvgFinComp', 'N2AvgFinComp', 'PVAvgFinComp',
                   'PLAvgFinComp', 'n_hatAvgFinComp', 'VAvgFinComp', 'DAvgFinComp', 'EAvgFinComp',
                   'TAvgFinComp', 'BAvgFinComp']

        values = dict()
        for col in columns:
            values[col] = []

        aggregated_features = pd.DataFrame(columns=columns)
        for name, group in grouped_features:
            for col in columns:
                v = group[col].to_numpy()[0]
                values[col].append(v)

        for key in values.keys():
            aggregated_features[key] = values[key]

        out_file = 'agg_features_spring'
        if semester == 'fall':
            out_file = 'agg_features_fall'
        if data == 'train':
            out_file += '_train.npy'
        else:
            out_file += '_test.npy'

        with open(out_file, 'wb') as f:
            d = aggregated_features.to_numpy()

            if semester == 'spring' and data == 'train':
                d = self.scaler.fit_transform(d)

            np.save(f, d)

    @staticmethod
    def load_feature(base_path='./', semester='spring', data='train'):

        out_file = 'agg_features_spring'
        if semester == 'fall':
            out_file = 'agg_features_fall'
        if data == 'train':
            out_file += '_train.npy'
        else:
            out_file += '_test.npy'

        with open(base_path+out_file, 'rb') as f:
            features = np.load(f)
        return features


if __name__ == '__main__':
    gen = DataGenerator('./Spring_data/F19_Release_Train_06-28-21/', max_problems=5000)
    gen.generate_features(semester='fall', data='train')
    """x = gen.load_feature()
    y = gen.load_feature(semester='fall')
    z = gen.load_feature(semester='fall', data='test')
    print(0)"""

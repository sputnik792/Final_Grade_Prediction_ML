import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras_tuner as kt
from keras.layers import Layer
from keras.utils.vis_utils import plot_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from keras.callbacks import EarlyStopping
from DatasetGenerator import DataGenerator

"""import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'"""

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


class Pooling(Layer):
    def __init__(self):
        super(Pooling, self).__init__(name='Pooling')

    def call(self, inputs):
        pooled = tf.reduce_max(inputs, axis=1)
        return pooled


class SimpleTreeCNN:
    def __init__(self, output_dim, num_conv, conv_dim, feature_dim=32, max_sequence_length=50,
                 max_number_of_sequences=32, inference_model='FFNN', use_residual='vanilla'):

        self.output_dim = output_dim
        self.num_conv = num_conv
        self.conv_dim = conv_dim
        self.feature_dim = feature_dim
        self.max_seq_length = max_sequence_length
        self.max_num_seq = max_number_of_sequences
        self.residual = use_residual

        self.conv_nodes = None
        self.trainable = True
        self.use_features = True
        self.modify = None

        self.inference_model = inference_model

        self.lstm_stack = 1
        self.feature_dim = 32
        self.num_tokens = 1000
        self.embedding_dim = 32

        self.network = self.build_model()

    def build_model(self, label_type='regression'):
        activation = None
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        metric = [tf.keras.metrics.BinaryAccuracy()]

        if label_type == 'regression':
            activation = 'linear'
            loss = tf.keras.losses.MeanSquaredError()  # 'mse'
            metric = [tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanSquaredError()]

        nodes = tf.keras.Input(shape=(self.max_num_seq, self.max_seq_length), name='Nodes')

        model_input = tf.keras.Input(shape=(self.max_seq_length,), name='Embedding_Input')
        embedding_layer = tf.keras.layers.Embedding(
            self.num_tokens + 1,
            self.embedding_dim,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(),
            trainable=True,
            mask_zero=True
        )(model_input)
        embedding_layer = tf.keras.layers.BatchNormalization()(embedding_layer)
        # embedding_layer = tf.keras.layers.Dense(self.embedding_dim, activation='relu')(embedding_layer)
        self.modify = embedding_layer.shape[1] * embedding_layer.shape[2]

        embedding_layer = tf.keras.layers.Reshape((-1, self.modify))(embedding_layer)
        embedding_layer = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=1))(embedding_layer)
        embedding_model = tf.keras.Model(inputs=model_input, outputs=embedding_layer)
        embedding_model.summary()

        hidden_layers = tf.keras.layers.TimeDistributed(embedding_model)(nodes)

        conv_layers = []
        for _ in range(self.num_conv):
            conv_layers.append(tf.keras.layers.Conv1D(self.output_dim, 3, activation='relu')(hidden_layers))
        hidden_layers = tf.keras.layers.Concatenate(axis=2)(conv_layers)

        inference_layers = self.attach_inference_model(hidden_layers, activation)

        model = tf.keras.Model(inputs=nodes, outputs=inference_layers, name='TBCNN')
        model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=1e-3, clipnorm=1.0), loss=loss,
                      metrics=metric)
        model.summary()

        plot_model(model, to_file='TBCNN.png', show_shapes=True, show_layer_names=True, show_layer_activations=True)

        return model

    """
    # Method to hyper-parameter tune TreeCNN model, requires hyperparameter object from kt-tuner
    # hp: Hyper-parameter object
    """

    def hyper_build(self, hp):
        loss = tf.keras.losses.MeanSquaredError()  # 'mse'
        metric = [tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanSquaredError()]

        # Input vectorized AST
        nodes = tf.keras.Input(shape=(self.max_num_seq, self.max_seq_length), name='Nodes')

        if self.use_features:
            features = tf.keras.Input(shape=(12,), name='features')

        # Beginning of embedding layer
        model_input = tf.keras.Input(shape=(self.max_seq_length,), name='Embedding_Input')
        embedding_layer = tf.keras.layers.Embedding(
            self.num_tokens + 1,
            self.embedding_dim,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(),
            trainable=True,
            mask_zero=True
        )(model_input)
        embedding_layer = tf.keras.layers.BatchNormalization()(embedding_layer)
        self.modify = embedding_layer.shape[1] * embedding_layer.shape[2]

        # Must reshape for TBCNN purposes (ndim = 3)
        embedding_layer = tf.keras.layers.Reshape((-1, self.modify))(embedding_layer)
        embedding_layer = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=1))(embedding_layer)
        embedding_model = tf.keras.Model(inputs=model_input, outputs=embedding_layer)

        # End of embedding layer, wrap with time distributed to embedded all subsequences
        hidden_layers = tf.keras.layers.TimeDistributed(embedding_model)(nodes)

        # Apply convolution to the sequence using ngrams. This differs from the TBCNN approach
        # with top, left and bottom weights
        if self.residual == 'normal':
            print('Choosing Normal Mode')
            conv_layers = []
            # hp_num_conv = hp.Int('num_conv', min_value=1, max_value=4, step=1)
            for _ in range(self.num_conv):
                hp_conv_units = hp.Int('kernel_size', min_value=1, max_value=3, step=1)
                hp_filter = hp.Int('filters', min_value=32, max_value=64, step=8)
                conv_layers.append(tf.keras.layers.Conv1D(hp_filter, hp_conv_units, activation='relu')(hidden_layers))

            # Concatenate all convolution filters
            hidden_layers = tf.keras.layers.Concatenate(axis=2)(conv_layers)
        elif self.residual == 'resnet':
            hidden_layers = self.build_resnet(hidden_layers, hp)
        elif self.residual == 'stacked':
            hp_num_conv = hp.Int('num_conv', min_value=1, max_value=4, step=1)
            for i in range(hp_num_conv):
                k_name = 'kernel_size' + str(i)
                f_name = 'filters' + str(i)
                hp_conv_units = hp.Int(k_name, min_value=1, max_value=3, step=1)
                hp_filter = hp.Int(f_name, min_value=32, max_value=512, step=32)

                hidden_layers = tf.keras.layers.Conv1D(hp_filter, hp_conv_units, strides=2, padding='same',
                                                       kernel_initializer='he_normal')(hidden_layers)
                hidden_layers = tf.keras.layers.BatchNormalization()(hidden_layers)
                hidden_layers = tf.keras.layers.Activation('relu')(hidden_layers)
        else:
            conv_layers = []
            hp_lstm = hp.Int('lstm_units', min_value=8, max_value=32, step=8)
            hp_dropout = hp.Choice('dropout', values=[0.3, 0.5, 0.7])
            for _ in range(self.num_conv):
                hp_conv_units = hp.Int('kernel_size', min_value=1, max_value=3, step=1)
                hp_filter = hp.Int('filters', min_value=32, max_value=512, step=32)
                conv_layers.append(tf.keras.layers.Conv1D(hp_filter, hp_conv_units, activation='relu')(hidden_layers))

            # Concatenate all convolution filters
            hidden_layers = tf.keras.layers.Concatenate(axis=2)(conv_layers)

            xf, _, _ = tf.keras.layers.LSTM(2, return_sequences=True,
                                            return_state=True, recurrent_dropout=hp_dropout)(hidden_layers)

            xb, _, _ = tf.keras.layers.LSTM(2, return_sequences=True, go_backwards=True,
                                            return_state=True, recurrent_dropout=hp_dropout)(hidden_layers)

            hidden_layers = tf.keras.layers.concatenate([xf, xb], axis=-1, name='bilstm_out')

        # Build Hidden Layers for inference module
        hp_reg = hp.Choice('kernel_regularizer', values=['l1', 'l2'])
        hidden_layers = Pooling()(hidden_layers)

        hp_unit1 = hp.Int('units1', min_value=8, max_value=64, step=8)
        hp_unit2 = hp.Int('units2', min_value=8, max_value=64, step=8)
        hidden_layers = tf.keras.layers.Dense(hp_unit1, activation='relu', kernel_regularizer=hp_reg)(hidden_layers)
        # hidden_layers = tf.keras.layers.Flatten()(hidden_layers)
        hidden_layers = tf.keras.layers.Dense(hp_unit2, activation='relu', kernel_regularizer=hp_reg)(hidden_layers)

        if self.use_features:
            feature_hidden = tf.keras.layers.Dense(hp_unit1, activation='relu', kernel_regularizer=hp_reg)(features)
            feature_hidden = tf.keras.layers.Dense(hp_unit2, activation='relu',
                                                   kernel_regularizer=hp_reg)(feature_hidden)

            hidden_layers = tf.keras.layers.Concatenate(axis=1)([hidden_layers, feature_hidden])

        # We need scores to be [0,100]
        def limiter(val):
            return tf.keras.activations.relu(val, max_value=100)

        inference_layers = tf.keras.layers.Dense(1, activation=limiter)(hidden_layers)

        # Build model and add optimizer
        if self.use_features:
            model = tf.keras.Model(inputs=[nodes, features], outputs=inference_layers, name='TBCNN')
        else:
            model = tf.keras.Model(inputs=nodes, outputs=inference_layers, name='TBCNN')

        hp_lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_lr,
                                                         clipnorm=1.0), loss=loss, metrics=metric)
        model.summary()

        plot_model(model, to_file='TBCNN.png', show_shapes=True, show_layer_names=True, show_layer_activations=True)

        if self.network is None:
            self.network = model

        return model

    def attach_inference_model(self, x, activation):
        if self.inference_model == 'FFNN':
            reg = 'l2'
            x = Pooling()(x)
            x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=reg)(x)
            # x = tf.keras.layers.Dropout(.1)(x)
            # x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=reg)(x)

            # x = tf.keras.layers.BatchNormalization()(x)

            def limiter(val):
                return tf.keras.activations.relu(val, max_value=100)

            x = tf.keras.layers.Dense(1, activation=limiter)(x)
        else:
            lstm_units = 128
            recurrent_dropout = .5

            xf, _, _ = tf.keras.layers.LSTM(lstm_units, return_sequences=True,
                                            return_state=True, recurrent_dropout=recurrent_dropout)(x)
            if self.lstm_stack > 1:
                for i in range(self.lstm_stack):
                    xf, _, _ = tf.keras.layers.LSTM(lstm_units, return_sequences=True,
                                                    return_state=True, recurrent_dropout=recurrent_dropout)(xf)

            xb, _, _ = tf.keras.layers.LSTM(lstm_units, return_sequences=True, go_backwards=True,
                                            return_state=True, recurrent_dropout=recurrent_dropout)(x)
            if self.lstm_stack > 1:
                for i in range(self.lstm_stack):
                    xb, _, _ = tf.keras.layers.LSTM(lstm_units, return_sequences=True, go_backwards=True,
                                                    return_state=True, recurrent_dropout=recurrent_dropout)(xb)

            x = tf.keras.layers.concatenate([xf, xb], axis=-1, name='bilstm_out')
            x = tf.keras.layers.GlobalMaxPool1D()(x)

            x = tf.keras.layers.Dropout(0.5)(x)
            x = tf.keras.layers.Dense(32, activation='tanh',
                                      kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dense(16, activation='tanh',
                                      kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dense(1, activation=activation, name='hidden_out')(x)

        return x

    def convolution_block(self, input_tensor, hp):
        hp_conv_units = hp.Int('kernel_size1', min_value=1, max_value=3, step=1)
        hp_filter = hp.Int('filter', min_value=32, max_value=64, step=2)

        x = tf.keras.layers.Conv1D(hp_filter, hp_conv_units, strides=1, padding='same', kernel_initializer='he_normal')(
            input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv1D(hp_filter, hp_conv_units, padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        hp_conv_units = hp.Int('kernel_size2', min_value=3, max_value=6, step=1)
        s = tf.keras.layers.Conv1D(hp_filter, hp_conv_units, padding='same', strides=1, kernel_initializer='he_normal')(
            input_tensor)
        s = tf.keras.layers.BatchNormalization()(s)

        out = tf.keras.layers.Add()([x, s])
        out = tf.keras.layers.Activation('relu')(out)

        return out

    def residual_block(self, input_tensor, hp):
        hp_conv_units = hp.Int('kernel_size3', min_value=1, max_value=3, step=1)
        hp_filter = hp.Int('filter', min_value=32, max_value=64, step=8)

        x = tf.keras.layers.Conv1D(hp_filter, hp_conv_units, strides=1, padding='same', kernel_initializer='he_normal')(
            input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        hp_conv_units = hp.Int('kernel_size4', min_value=3, max_value=6, step=1)
        x = tf.keras.layers.Conv1D(hp_filter, hp_conv_units, strides=1, padding='same', kernel_initializer='he_normal')(
            x)
        x = tf.keras.layers.BatchNormalization()(x)

        out = tf.keras.layers.Add()([x, input_tensor])
        out = tf.keras.layers.Activation('relu')(out)
        return out

    def build_resnet(self, hidden_layers, hp):
        hp_conv_units = hp.Int('kernel_size', min_value=1, max_value=3, step=1)
        hp_filter = hp.Int('filter', min_value=32, max_value=128, step=32)

        hidden_layers = tf.keras.layers.Conv1D(hp_filter, hp_conv_units, strides=1, kernel_initializer='he_normal')(
            hidden_layers)
        hidden_layers = tf.keras.layers.BatchNormalization()(hidden_layers)
        hidden_layers = tf.keras.layers.Activation('relu')(hidden_layers)

        # Stage 2
        hidden_layers = self.convolution_block(hidden_layers, hp)
        hidden_layers = self.residual_block(hidden_layers, hp)
        hidden_layers = self.residual_block(hidden_layers, hp)

        # stage 3
        hidden_layers = self.convolution_block(hidden_layers, hp)
        hidden_layers = self.residual_block(hidden_layers, hp)
        hidden_layers = self.residual_block(hidden_layers, hp)
        hidden_layers = self.residual_block(hidden_layers, hp)
        return hidden_layers


def main():
    generator = DataGenerator()
    test = True
    use_residual = 'stacked'
    use_features = True

    tree = SimpleTreeCNN(100, 1, 1024, max_number_of_sequences=5000, use_residual=use_residual)
    epoch = 200
    patience = 5
    es = EarlyStopping(monitor='val_loss', patience=patience)

    vectorized_spring, _, labels_spring = generator.load_train_data('spring')
    # labels_spring = labels_spring * 100
    # vectorized_spring = np.reshape(vectorized_spring, newshape=(247, 7000, 50))

    vectorized_fall, _, labels_fall = generator.load_train_data('fall')
    # vectorized_fall = np.reshape(vectorized_fall, newshape=(368, 7000, 50))

    # Combine Fall and Spring into a single dataset
    vectorized = np.concatenate([vectorized_fall[1:], vectorized_spring[1:]])
    labels = np.concatenate([labels_fall, labels_spring])

    if use_features:
        features_fall = generator.load_feature(semester='fall', data='train')
        features_spring = generator.load_feature()

        features = np.concatenate([features_fall, features_spring])

        features_test = generator.load_feature(semester='fall', data='test')

    def model_builder(hp):
        return tree.hyper_build(hp)

    batch_size = 8

    tuner = kt.BayesianOptimization(
        model_builder,
        objective='val_loss',
        max_trials=10,
        directory='./tuner',
        project_name='code_embedding(MSE 249_feat_normal)'
    )

    if use_features:
        # Split the data for cross validation and hyper-parameter tuning
        vec_train, vec_test, feat_train, feat_test, y_train, y_test = train_test_split(vectorized,
                                                                                       features, labels,
                                                                                       test_size=.2, random_state=19)

        # Search for best parameters
        tuner.search([vec_train, feat_train], y_train, epochs=100,
                     batch_size=batch_size, validation_split=0.2, callbacks=[es])
        tuner.search_space_summary()

        hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        tree.network = tuner.hypermodel.build(hps)
        tree.network.summary()

        vec_train = tf.convert_to_tensor(vec_train, dtype=tf.float32)
        feat_train = tf.convert_to_tensor(feat_train, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

        # print(f'The average MSE is {np.mean(errors)}')

        # Now train on all the training data less 20% validation split for final inference model
        total_history = tree.network.fit([vec_train, feat_train], y_train,
                                         epochs=epoch, batch_size=batch_size, validation_split=0.1,
                                         verbose=1, callbacks=[es], use_multiprocessing=True)

        # Plot overall training progression
        plt.plot(total_history.history['loss'])
        plt.plot(total_history.history['val_loss'])
        plt.title('model loss overall')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('./train_plots/training_overall.png')
        plt.clf()
    else:
        # Split the data for cross validation and hyper-parameter tuning
        x_train, x_test, y_train, y_test = train_test_split(vectorized, labels, test_size=.2, random_state=19)

        # Search for best parameters
        tuner.search(x_train, y_train, epochs=100, batch_size=batch_size, validation_split=0.2, callbacks=[es])
        tuner.search_space_summary()

        hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        tree.network = tuner.hypermodel.build(hps)

        x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

        # print(f'The average MSE is {np.mean(errors)}')

        # Now train on all the training data less 20% validation split for final inference model
        total_history = tree.network.fit(x_train, y_train, epochs=epoch, batch_size=batch_size, validation_split=0.1,
                                         verbose=1, callbacks=[es], use_multiprocessing=True)

        # Plot overall training progression
        plt.plot(total_history.history['loss'])
        plt.plot(total_history.history['val_loss'])
        plt.title('model loss overall')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('./train_plots/training_overall.png')
        plt.clf()

        tree.network.summary()

    sk = KFold(n_splits=3, random_state=19, shuffle=True)

    histories = []
    errors = []
    count = 1

    # Perform K-Fold cross validation to ensure good parameter choice
    """for train_index, test_index in sk.split(x_train, y_train):
        m = tuner.hypermodel.build(hps)
        x_tr, x_te = x_train[train_index], x_train[test_index]
        y_tr, y_te = y_train[train_index], y_train[test_index]
        x_tr = tf.convert_to_tensor(x_tr, dtype=tf.float32)
        y_tr = tf.convert_to_tensor(y_tr, dtype=tf.float32)
        x_te = tf.convert_to_tensor(x_te, dtype=tf.float32)
        y_te = tf.convert_to_tensor(y_te, dtype=tf.float32)
        history = m.fit(x_tr, y_tr, epochs=epoch, batch_size=batch_size, validation_split=0.1,
                        verbose=1, callbacks=[es], use_multiprocessing=True)
        histories.append(history)
        y_pred = np.ravel(m.predict(x_te, batch_size=batch_size))
        report = mean_squared_error(y_true=y_te, y_pred=y_pred)
        errors.append(report)
        print(f'Finished fold {count}')
        count += 1
        del m
    # summarize history for loss
    for i, hist in enumerate(histories):
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title('model loss fold ' + str(i))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('./train_plots/training_fold_' + str(i) + '.png')
        plt.clf()"""

    tree.network.save('./Model/TBCNN_regression')
    print(f'The best hyper-parameters are {hps}')

    if use_features:
        # Predict on held out validation set (Will be used for class report?)
        y_pred = np.ravel(tree.network.predict([vec_test, feat_test], batch_size=batch_size))
    else:
        # Predict on held out validation set (Will be used for class report?)
        y_pred = np.ravel(tree.network.predict(x_test, batch_size=batch_size))

    print(f'Predictions for regression are {np.unique(y_pred, return_counts=True)}')
    print(f'The labels for regression are {np.unique(y_test, return_counts=True)}')
    report = mean_squared_error(y_true=y_test, y_pred=y_pred, squared=False)
    print(f'The models RMSE on the test set is {report}')
    report = mean_squared_error(y_true=y_test, y_pred=y_pred)
    print(f'The models MSE on the test set is {report}')

    df = pd.DataFrame(columns=['Predictions', 'Labels'])
    df['Predictions'] = y_pred
    df['Labels'] = y_test
    df.to_csv('train_val_output.csv')

    # If directed we will evaluate model on actual Fall test and save for upload to CSEDM
    if test:
        final_vectorized, _, _ = generator.load_test_data('fall')
        final_vectorized = tf.convert_to_tensor(final_vectorized[1:], dtype=tf.float32)

        if use_features:
            y_pred = np.ravel(tree.network.predict([final_vectorized, features_test], batch_size=batch_size))
        else:
            y_pred = np.ravel(tree.network.predict(final_vectorized, batch_size=batch_size))

        record = generator.record
        record['X-Grade'] = y_pred
        record.to_csv('Test_Predictions_Fall.csv')


if __name__ == "__main__":
    main()

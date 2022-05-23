import math
import numpy as np
import pandas as pd
import sklearn
import javalang
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge, PassiveAggressiveRegressor, HuberRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, StackingRegressor, VotingRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, cross_validate

# operands
primitive_types = ["int", "long", "char", "double", "boolean", "float", "void", "short", "byte"]
var_types = ["String", "StringBuilder", "StringBuffer", "Integer", "Double", "Float", "Boolean", "Byte"]
packages = ["Math", "System"]
functions_math = ["abs", "max", "min", "round", "sqrt", "cbrt", "pow", "signum", "ceil", "copySign", "nextAfter", "nextUp",
	"nextDown", "floor", "floorDiv", "random", "log", "exp", "log10", "sin", "cos", "tan"]
functions_sys = ["out", "print", "println"]
functions_parse = ["parseInt", "parseDouble", "parseFloat", "parseLong"]
functions_str = ["charAt", "codePointAt", "codePointBefore", "codePointCount", "compareTo", "compareToIgnoreCase", "concat",
	"contains", "contentEquals", "copyValueOf", "endsWith", "equals", "equalsIgnoreCase", "format", "getBytes", "getChars",
	"hashCode", "indexOf", "intern", "isEmpty", "lastIndexOf", "length", "matches", "offsetByCodePoints", "regionMatches", 
	"replace", "replaceFirst", "replaceAll", "split", "startsWith", "subSequence", "substring", "toCharArray", "toLowerCase",
	"toString", "toUpperCase", "trim", "valueOf"]
functions_data_structs = ["append", "delete", "insert", "reverse", "pop", "push", "poll", "peek", "remove", "add", "size"]

# operators
reserved_words = ["abstract", "continue", "for", "new", "switch", "assert", "default", "goto", "package", "synchronized", 
	"do", "if", "private", "this", "break", "implements", "protected", "throws", "throw", "true", "false", "const"
	"else", "import", "public", "throws", "case", "enum", "instanceof", "return", "transient", "catch", "extends"	
	"try", "final", "interface", "static", "class", "finally", "strictfp", "volatile", "native", "super", "while", "null", "main"]
operator_symbols = ["!", "!=", "%", "%=", "&", "&&", "||", "&=", "(", "*", "*=", "+", "++", "+=", ",",
	"-", "--", "-=", "->", ".", "...", "/", "/=", ":", "::", "<", "<<", "<<=", "<=", "=", "==", ">", ">=",
	">>", ">>=", "?", "[", "^", "^=", "{", "|", "|=", "~", "}", ")", "]"]
left_side_brackets = ["{", "(", "["]
right_side_brackets = ["}", ")", "]"]

special_case = {"case":1, "for":2, "if":2, "switch":2, "while":2, "catch":2}

operator_corpus = reserved_words + operator_symbols


def listToString(s): 
	arr_len = len(s)
	strv = str(round(s[0], 4)) 
	for i in range(1,arr_len): 
		strv = strv + "," + str(round(s[i], 4))
	return strv

### because the definitions of operators and operands slightly vary between languages,
### we define them using a similar corpus format as the CMTJava by Verifysoft Technology
# we count brackets as 2 operators except in the special cases
def get_operators_operands(code):
	operands_dict = {}
	operators_dict = {}
	tokens_list = list(javalang.tokenizer.tokenize(code))
	tokens = []
	for x in tokens_list:
		tokens.append(x.value)
	for i in range(0, len(tokens)):
		t = tokens[i]
		### differentiate true operators from constant values that use operators,
		# i.e + vs '+' string 
		if str(tokens_list[i])[0:8] == "Operator" or str(tokens_list[i])[0:9] in "Separator":
			if t in operator_corpus:
				if t in operators_dict:
					operators_dict[t] += 1 
				else:
					operators_dict[t] = 1
		else:
			if t in operator_corpus:
				if t in operators_dict:
					operators_dict[t] += 1 
				else:
					operators_dict[t] = 1
			else:
				if t in operands_dict:
					operands_dict[t] += 1 
				else:
					operands_dict[t] = 1
	
	n1 = len(operators_dict.keys())
	n2 = len(operands_dict.keys())
	
	N1 = sum(operators_dict.values())
	N2 = sum(operands_dict.values())

	for key, val in operators_dict.items():
		if key in special_case:
			N1 = N1 - val*special_case[key]
	
	o_vec = [n1, n2, N1, N2]

	return o_vec

def halstead_complexity_metrics(problem_vector):
	n1 = problem_vector[0]
	n2 = problem_vector[1]
	N1 = problem_vector[2]
	N2 = problem_vector[3]

	# program voaculary
	PV = n1 + n2

	# program length
	PL = N1 + N2

	# estimated program length
	N_hat = n1 * math.log(n1) + n2 * math.log(n2)
	# volume
	V = PL * math.log(PV)
	# difficulty
	D = n1/2 * N2/n2
	# effort
	E = D*V

	# time 
	T = E/18

	#bugs
	B = math.pow(E, (2/3))/3000
	# B_alt = V/3000

	return [PV, PL, N_hat, V, D, E, T, B]

def main():
	### Load Data
	extract_columns = ["SubjectID", "AssignmentID", "ProblemID", "CodeStateID", "Compile.Result"]
	
	### Fall2019 -------------------------------
	### Training Set
	codeStates_train_fall_df = pd.read_csv("data\\F19_Release_Train_06-28-21\\Train\\Data\\CodeStates\\CodeStates.csv")
	subject_train_fall_df = pd.read_csv("data\\F19_Release_Train_06-28-21\\Train\\Data\\LinkTables\\Subject.csv")
	mainTable_train_fall_df = pd.read_csv("data\\F19_Release_Train_06-28-21\\Train\\Data\\MainTable.csv")
 	# train_early_fall_df = pd.read_csv()
	# train_late_fall_df = pd.read_csv()
	
	### Test Set
	codeStates_test_fall_df = pd.read_csv("data\\F19_Release_Test_06-28-21\\Test\\Data\\CodeStates\\CodeStates.csv")
	subject_test_fall_df = pd.read_csv("data\\F19_Release_Test_06-28-21\\Test\\Data\\LinkTables\\Subject.csv")
	mainTable_test_fall_df = pd.read_csv("data\\F19_Release_Test_06-28-21\\Test\\Data\\MainTable.csv")
	# test_early_fall_df =
	# test_late_fall_df =

	### ------------ Spring2019 -------------------
	### Training Set
	codeStates_train_spring_df = pd.read_csv("data\\S19_Release_6_28_21\\Train\\Data\\CodeStates\\CodeStates.csv")
	subject_train_spring_df = pd.read_csv("data\\S19_Release_6_28_21\\Train\\Data\\LinkTables\\Subject.csv")
	mainTable_train_spring_df = pd.read_csv("data\\S19_Release_6_28_21\\Train\\Data\\MainTable.csv")
	#train_early_spring_df = pd.read_csv("data\\S19_Release_6_28_21\\Train\\early.csv")
	#train_late_spring_df = pd.read_csv("data\\S19_Release_6_28_21\\Train\\late.csv")

	### Test Set
	codeStates_test_spring_df = pd.read_csv("data\\S19_Release_6_28_21\\Test\\Data\\CodeStates\\CodeStates.csv")
	subject_test_spring_df = pd.read_csv("data\\S19_Release_6_28_21\\Test\\Data\\LinkTables\\Subject.csv")
	mainTable_test_spring_df = pd.read_csv("data\\S19_Release_6_28_21\\Test\\Data\\MainTable.csv")
	# = pd.read_csv("data\\S19_Release_6_28_21\\Test\\early.csv")
	#test_late_spring_df = pd.read_csv("data\\S19_Release_6_28_21\\Test\\late.csv")
	# --------------------

	### Spring Train 
	## for scaling the grades to 0-100 for cross training between semesters
	# subject_train_spring_df['X-Grade'] = subject_train_spring_df['X-Grade']*100

	mainTable_subset_train_spring_df = mainTable_train_spring_df[extract_columns]
	mainTable_subset_train_spring_df = mainTable_subset_train_spring_df.dropna(subset=['Compile.Result'])
	mainTable_subset_train_spring_df = mainTable_subset_train_spring_df.astype({'AssignmentID': 'int32'})
	mainTable_subset_train_spring_df = mainTable_subset_train_spring_df[ mainTable_subset_train_spring_df['Compile.Result'] == 'Success' ]
	
	spring_train_merged_df = pd.merge(mainTable_subset_train_spring_df, codeStates_train_spring_df, on="CodeStateID")
	
	maintable_subset_train_spring_scores_df = mainTable_train_spring_df[["CodeStateID", "Score"]]
	maintable_subset_train_spring_scores_df = maintable_subset_train_spring_scores_df.dropna(subset=['Score'])

	spring_train_merged_df = pd.merge(spring_train_merged_df, maintable_subset_train_spring_scores_df, on="CodeStateID")
	
	### Spring Test
	mainTable_subset_test_spring_df = mainTable_test_spring_df[extract_columns]
	mainTable_subset_test_spring_df = mainTable_subset_test_spring_df.dropna(subset=['Compile.Result'])
	mainTable_subset_test_spring_df = mainTable_subset_test_spring_df.astype({'AssignmentID': 'int32'})
	mainTable_subset_test_spring_df = mainTable_subset_test_spring_df[ mainTable_subset_test_spring_df['Compile.Result'] == 'Success' ]
	
	spring_test_merged_df = pd.merge(mainTable_subset_test_spring_df, codeStates_test_spring_df, on="CodeStateID")
	
	maintable_subset_test_spring_scores_df = mainTable_test_spring_df[["CodeStateID", "Score"]]
	maintable_subset_test_spring_scores_df = maintable_subset_test_spring_scores_df.dropna(subset=['Score'])

	spring_test_merged_df = pd.merge(spring_test_merged_df, maintable_subset_test_spring_scores_df, on="CodeStateID")

	### Fall Train
	mainTable_subset_train_fall_df = mainTable_train_fall_df[extract_columns]
	mainTable_subset_train_fall_df = mainTable_subset_train_fall_df.dropna(subset=['Compile.Result'])
	mainTable_subset_train_fall_df = mainTable_subset_train_fall_df.astype({'AssignmentID': 'int32'})
	mainTable_subset_train_fall_df = mainTable_subset_train_fall_df[ mainTable_subset_train_fall_df['Compile.Result'] == 'Success' ]
	
	fall_train_merged_df = pd.merge(mainTable_subset_train_fall_df, codeStates_train_fall_df, on="CodeStateID")

	maintable_subset_train_fall_scores_df = mainTable_train_fall_df[["CodeStateID", "Score"]]
	maintable_subset_train_fall_scores_df = maintable_subset_train_fall_scores_df.dropna(subset=['Score'])

	fall_train_merged_df = pd.merge(fall_train_merged_df, maintable_subset_train_fall_scores_df, on="CodeStateID")

	### Fall Test
	mainTable_subset_test_fall_df = mainTable_test_fall_df[extract_columns]
	mainTable_subset_test_fall_df = mainTable_subset_test_fall_df.dropna(subset=['Compile.Result'])
	mainTable_subset_test_fall_df = mainTable_subset_test_fall_df.astype({'AssignmentID': 'int32'})
	mainTable_subset_test_fall_df = mainTable_subset_test_fall_df[ mainTable_subset_test_fall_df['Compile.Result'] == 'Success' ]
	
	fall_test_merged_df = pd.merge(mainTable_subset_test_fall_df, codeStates_test_fall_df, on="CodeStateID")

	maintable_subset_test_fall_scores_df = mainTable_test_fall_df[["CodeStateID", "Score"]]
	maintable_subset_test_fall_scores_df = maintable_subset_test_fall_scores_df.dropna(subset=['Score'])

	fall_test_merged_df = pd.merge(fall_test_merged_df, maintable_subset_test_fall_scores_df, on="CodeStateID")

	### Merge Codestates with features 
	halstead_labels = ['n1', 'n2', 'N1', 'N2', 'PV', 'PL', 'n_hat', 'V', 'D', 'E', 'T', 'B']
	# codeDummyBit = codeStates_train_fall_df["Code"][14095]
	# problem_index = [2452, 24560, 43036, 43037, 43038, 43039, 43040, 43041]
	# data1 = [[], [], [], [], [], [], [], [], [], [], [], []]
	# fall_len = len(fall_train_merged_df)
	# for x in range(0, fall_len):
	# 	if x in problem_index:
	# 		if x > 43035:
	# 			code = codeDummyBit
	# 		else:
	# 			y = x + 1
	# 			code = fall_train_merged_df["Code"][y]
	# 	else:
	# 		code = fall_train_merged_df["Code"][x]
	# 	vec_1 = get_operators_operands(code)
	# 	vec_2 = halstead_complexity_metrics(vec_1)
	# 	halstead_vec = vec_1 + vec_2
	# 	for i in range(0,12):
	# 		data1[i].append(halstead_vec[i])
	# for indx in range(0,12):
	# 	ft = halstead_labels[indx]
	# 	fall_train_merged_df[ft] = data1[indx]
	# fall_train_merged_df.to_csv('CodeStates_Fall_Train_With_Halstead.csv', encoding='utf-8', index=False)

	# data2 = [[], [], [], [], [], [], [], [], [], [], [], []]
	# for code in fall_test_merged_df["Code"]:
	# 	vec_1 = get_operators_operands(code)
	# 	vec_2 = halstead_complexity_metrics(vec_1)
	# 	halstead_vec = vec_1 + vec_2
	# 	for i in range(0,12):
	# 		data2[i].append(halstead_vec[i])
	# for indx in range(0,12):
	# 	ft = halstead_labels[indx]
	# 	fall_test_merged_df[ft] = data2[indx]
	# fall_test_merged_df.to_csv('CodeStates_Fall_Test_With_Halstead.csv', encoding='utf-8', index=False)

	# data3 = [[], [], [], [], [], [], [], [], [], [], [], []]
	# for code in spring_train_merged_df["Code"]:
	# 	vec_1 = get_operators_operands(code)
	# 	vec_2 = halstead_complexity_metrics(vec_1)
	# 	halstead_vec = vec_1 + vec_2
	# 	for i in range(0,12):
	# 		data3[i].append(halstead_vec[i])
	# for indx in range(0,12):
	# 	ft = halstead_labels[indx]
	# 	spring_train_merged_df[ft] = data3[indx]
	# spring_train_merged_df.to_csv('CodeStates_Spring_Train_With_Halstead.csv', encoding='utf-8', index=False)

	# data4 = [[], [], [], [], [], [], [], [], [], [], [], []]
	# for code in spring_test_merged_df["Code"]:
	# 	vec_1 = get_operators_operands(code)
	# 	vec_2 = halstead_complexity_metrics(vec_1)
	# 	halstead_vec = vec_1 + vec_2
	# 	for i in range(0,12):
	# 		data4[i].append(halstead_vec[i])
	# for indx in range(0,12):
	# 	ft = halstead_labels[indx]
	# 	spring_test_merged_df[ft] = data4[indx]
	# spring_test_merged_df.to_csv('CodeStates_Spring_Test_With_Halstead.csv', encoding='utf-8', index=False)

	spring_train_feats_df = pd.read_csv('data\\Features\\CodeStates_Spring_Train_With_Halstead.csv')
	spring_test_feats_df = pd.read_csv('data\\Features\\CodeStates_Spring_Test_With_Halstead.csv')
	fall_train_feats_df = pd.read_csv('data\\Features\\CodeStates_Fall_Train_With_Halstead.csv')
	fall_test_feats_df = pd.read_csv('data\\Features\\CodeStates_Fall_Test_With_Halstead.csv')

	# average of the features
	# how to aggregate?
	#	by problem per assignment per subjectID before total aggregation, 
	halstead_labels = ['n1', 'n2', 'N1', 'N2', 'PV', 'PL', 'n_hat', 'V', 'D', 'E', 'T', 'B']

	halstead_labels_2 = []
	for z in halstead_labels:
		halstead_labels_2.append(z + "Avg")
	
	halstead_labels_3 = []
	for z in halstead_labels_2:
		halstead_labels_3.append(z + "Fin")
	
	halstead_labels_4 = []   # feature format is 'n1AvgFinComp'
	for z in halstead_labels_3:
		halstead_labels_4.append(z + "Comp") 


	subsetCols = ['SubjectID','AssignmentID', 'ProblemID']
	subsetCols2 = ['SubjectID','AssignmentID']
	subsetCols3 = ['SubjectID']
	halstead_labels_fin2 = subsetCols2 + halstead_labels_3
	halstead_labels_fin = ["SubjectID"] + halstead_labels_4
	##----------------------------------------------

	# Fall Training Features, SubjectID, and Grade
	agg_by_prob_fall_train_df = fall_train_feats_df.drop_duplicates(subset=subsetCols, keep="last")
	for label in halstead_labels:
		newlabel = label+"Avg"
		get_avg_df = fall_train_feats_df.groupby(subsetCols)[label].mean().reset_index().rename(columns={label:newlabel})
		agg_by_prob_fall_train_df = pd.merge(agg_by_prob_fall_train_df, get_avg_df, on=subsetCols)
	
	agg_by_assign_fall_train_df = agg_by_prob_fall_train_df.drop_duplicates(subset=subsetCols2, keep="last")
	for label in halstead_labels_2:
		newlabel = label+"Fin"
		get_avg_df = agg_by_prob_fall_train_df.groupby(subsetCols2)[label].mean().reset_index().rename(columns={label:newlabel})
		agg_by_assign_fall_train_df = pd.merge(agg_by_assign_fall_train_df, get_avg_df, on=subsetCols2)
	
	temp_agg_sub_assign_df = agg_by_assign_fall_train_df[halstead_labels_fin2]
	temp_agg_sub_assign_df.to_csv('Fall_by_Subject-Assignment_Grades.csv', encoding='utf-8', index=False)
	
	curr_subject = "014604ba54339d4b1266cf78e125053a5ac11dd861ef3cc0b4ed777ed0e2af0a"
	assign_list = [439, 487, 492, 494, 502]
	subject_dict = {curr_subject:0}
	subject_dict_2 = {curr_subject:assign_list}
	
	for col in range(0, len(agg_by_assign_fall_train_df)):
		new_subject = agg_by_assign_fall_train_df.iloc[col]["SubjectID"]
		new_assign = agg_by_assign_fall_train_df.iloc[col]["AssignmentID"]
		if new_subject == curr_subject:
			subject_dict[new_subject] += 1
			subject_dict_2[new_subject].remove(new_assign)
		else:
			subject_dict[new_subject] = 1
			subject_dict_2[new_subject] = [439, 487, 492, 494, 502]
			subject_dict_2[new_subject].remove(new_assign)
			curr_subject = new_subject
	null_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	for subj in subject_dict:
		if subject_dict[subj] < 5:
			for assign in subject_dict_2[subj]:
				temp_agg_sub_assign_df.loc[len(temp_agg_sub_assign_df.index)] = [subj, assign] + null_list
	###
	### padded with missing assignments
	agg_by_assign_fall_train_df = temp_agg_sub_assign_df

	temp_agg_sub_assign_df = temp_agg_sub_assign_df.sort_values(subsetCols2)

	ext_feat_labels = []
	for i in range(60):
		ext_feat_labels.append("ft" + str(i))
	ext_feat_labels = ['SubjectID'] + ext_feat_labels
	new_agg_df = pd.DataFrame(columns=ext_feat_labels)
	extended_feats = []
	for j in range(len(temp_agg_sub_assign_df)):
		if j % 5 != 4:
			for indx in halstead_labels_3:
				extended_feats.append(temp_agg_sub_assign_df.iloc[j][indx])
		else:
			for indx in halstead_labels_3:
				extended_feats.append(temp_agg_sub_assign_df.iloc[j][indx])
			subj = temp_agg_sub_assign_df.iloc[j]['SubjectID']
			new_agg_df.loc[len(new_agg_df.index)] = [subj] + extended_feats
			extended_feats = []
	new_agg_df.to_csv('Fall_by_Subject_60Feats.csv', encoding='utf-8', index=False)

	agg_by_std_fall_train_df = agg_by_assign_fall_train_df.drop_duplicates(subset=subsetCols3, keep="last")
	for label in halstead_labels_3:
		newlabel = label+"Comp"
		get_avg_df = agg_by_assign_fall_train_df.groupby(subsetCols3)[label].mean().reset_index().rename(columns={label:newlabel})
		agg_by_std_fall_train_df = pd.merge(agg_by_std_fall_train_df, get_avg_df, on=subsetCols3)
	
	fall_train_feats_aggragated_df = agg_by_std_fall_train_df[halstead_labels_fin]
	fall_train_final_df = pd.merge(fall_train_feats_aggragated_df, subject_train_fall_df, on="SubjectID")

	fall_train_final_df.to_csv('Fall_Train_Subject_With_Halstead.csv', encoding='utf-8', index=False)

	### Fall Test Features and Subject
	agg_by_prob_fall_test_df = fall_test_feats_df.drop_duplicates(subset=subsetCols, keep="last")
	for label in halstead_labels:
		newlabel = label+"Avg"
		get_avg_df = fall_test_feats_df.groupby(subsetCols)[label].mean().reset_index().rename(columns={label:newlabel})
		agg_by_prob_fall_test_df = pd.merge(agg_by_prob_fall_test_df, get_avg_df, on=subsetCols)
	
	agg_by_assign_fall_test_df = agg_by_prob_fall_test_df.drop_duplicates(subset=subsetCols2, keep="last")
	for label in halstead_labels_2:
		newlabel = label+"Fin"
		get_avg_df = agg_by_prob_fall_test_df.groupby(subsetCols2)[label].mean().reset_index().rename(columns={label:newlabel})
		agg_by_assign_fall_test_df = pd.merge(agg_by_assign_fall_test_df, get_avg_df, on=subsetCols2)
	
	agg_by_std_fall_test_df = agg_by_assign_fall_test_df.drop_duplicates(subset=subsetCols3, keep="last")
	for label in halstead_labels_3:
		newlabel = label+"Comp"
		get_avg_df = agg_by_assign_fall_test_df.groupby(subsetCols3)[label].mean().reset_index().rename(columns={label:newlabel})
		agg_by_std_fall_test_df = pd.merge(agg_by_std_fall_test_df, get_avg_df, on=subsetCols3)
	
	fall_test_feats_aggragated_df = agg_by_std_fall_test_df[halstead_labels_fin]
	fall_test_final_df = pd.merge(fall_test_feats_aggragated_df, subject_test_fall_df, on="SubjectID")

	fall_test_final_df.to_csv('Fall_Test_Subject_With_Halstead.csv', encoding='utf-8', index=False)

	x_grades_pred_fall_linReg_df = trainLinearRegression(fall_train_final_df, fall_test_final_df, halstead_labels_4)
	x_grades_pred_fall_linReg_df[['SubjectID', 'X-Grade']].to_csv('Fall_Predicted_LinReg_Grades.csv', encoding='utf-8', index=False)

	x_grades_pred_fall_RidgeReg_df = trainRidgeRegression(fall_train_final_df, fall_test_final_df, halstead_labels_4)
	x_grades_pred_fall_RidgeReg_df[['SubjectID', 'X-Grade']].to_csv('Fall_Predicted_RidgeReg_Grades.csv', encoding='utf-8', index=False)

	x_grades_pred_fall_LassoReg_df = trainLassoRegression(fall_train_final_df, fall_test_final_df, halstead_labels_4)
	x_grades_pred_fall_LassoReg_df[['SubjectID', 'X-Grade']].to_csv('Fall_Predicted_LassoReg_Grades.csv', encoding='utf-8', index=False)
	
	x_grades_pred_fall_BayesRidgeReg_df = trainBayesRidgeRegression(fall_train_final_df, fall_test_final_df, halstead_labels_4)
	x_grades_pred_fall_BayesRidgeReg_df[['SubjectID', 'X-Grade']].to_csv('Fall_Predicted_BayesRidgeReg_Grades.csv', encoding='utf-8', index=False)

	x_grades_pred_fall_SVReg_df = trainSVRegression(fall_train_final_df, fall_test_final_df, halstead_labels_4)
	x_grades_pred_fall_SVReg_df[['SubjectID', 'X-Grade']].to_csv('Fall_Predicted_SVReg_Grades.csv', encoding='utf-8', index=False)

	x_grades_pred_fall_XGBReg_df = trainXGBRegression(fall_train_final_df, fall_test_final_df, halstead_labels_4)
	x_grades_pred_fall_XGBReg_df[['SubjectID', 'X-Grade']].to_csv('Fall_Predicted_XGBReg_Grades.csv', encoding='utf-8', index=False)

	x_grades_pred_fall_RFReg_df = trainRFRegression(fall_train_final_df, fall_test_final_df, halstead_labels_4)
	x_grades_pred_fall_RFReg_df[['SubjectID', 'X-Grade']].to_csv('Fall_Predicted_RFReg_Grades.csv', encoding='utf-8', index=False)

	x_grades_pred_fall_KNNReg_df = trainKNNRegression(fall_train_final_df, fall_test_final_df, halstead_labels_4)
	x_grades_pred_fall_KNNReg_df[['SubjectID', 'X-Grade']].to_csv('Fall_Predicted_KNNReg_Grades.csv', encoding='utf-8', index=False)

	x_grades_pred_fall_BaggingReg_df = trainBaggingRegression(fall_train_final_df, fall_test_final_df, halstead_labels_4)
	x_grades_pred_fall_BaggingReg_df[['SubjectID', 'X-Grade']].to_csv('Fall_Predicted_BagReg_SVR_Grades.csv', encoding='utf-8', index=False)

	x_grades_pred_fall_StackReg_df = trainStackingRegression(fall_train_final_df, fall_test_final_df, halstead_labels_4)
	x_grades_pred_fall_StackReg_df[['SubjectID', 'X-Grade']].to_csv('Fall_Predicted_StackReg_SVR_Grades.csv', encoding='utf-8', index=False)

	x_grades_pred_fall_VoteReg_df = trainVotingRegression(fall_train_final_df, fall_test_final_df, halstead_labels_4)
	x_grades_pred_fall_VoteReg_df[['SubjectID', 'X-Grade']].to_csv('Fall_Predicted_VoteReg_SVR_Grades.csv', encoding='utf-8', index=False)

	print("-------------------------------------------------------")

	# Spring Training Features, SubjectID, and Grade
	agg_by_prob_spring_train_df = spring_train_feats_df.drop_duplicates(subset=subsetCols, keep="last")
	for label in halstead_labels:
		newlabel = label+"Avg"
		get_avg_df = spring_train_feats_df.groupby(subsetCols)[label].mean().reset_index().rename(columns={label:newlabel})
		agg_by_prob_spring_train_df = pd.merge(agg_by_prob_spring_train_df, get_avg_df, on=subsetCols)
	
	agg_by_assign_spring_train_df = agg_by_prob_spring_train_df.drop_duplicates(subset=subsetCols2, keep="last")
	for label in halstead_labels_2:
		newlabel = label+"Fin"
		get_avg_df = agg_by_prob_spring_train_df.groupby(subsetCols2)[label].mean().reset_index().rename(columns={label:newlabel})
		agg_by_assign_spring_train_df = pd.merge(agg_by_assign_spring_train_df, get_avg_df, on=subsetCols2)
	
	agg_by_std_spring_train_df = agg_by_assign_spring_train_df.drop_duplicates(subset=subsetCols3, keep="last")
	for label in halstead_labels_3:
		newlabel = label+"Comp"
		get_avg_df = agg_by_assign_spring_train_df.groupby(subsetCols3)[label].mean().reset_index().rename(columns={label:newlabel})
		agg_by_std_spring_train_df = pd.merge(agg_by_std_spring_train_df, get_avg_df, on=subsetCols3)
	
	spring_train_feats_aggragated_df = agg_by_std_spring_train_df[halstead_labels_fin]
	spring_train_final_df = pd.merge(spring_train_feats_aggragated_df, subject_train_spring_df, on="SubjectID")

	spring_train_final_df.to_csv('Spring_Train_Subject_With_Halstead.csv', encoding='utf-8', index=False)

	# ## Spring Test Set Features and subject
	# agg_by_prob_spring_test_df = spring_test_feats_df.drop_duplicates(subset=subsetCols, keep="last")
	# for label in halstead_labels:
	# 	newlabel = label+"Avg"
	# 	get_avg_df = spring_test_feats_df.groupby(subsetCols)[label].mean().reset_index().rename(columns={label:newlabel})
	# 	agg_by_prob_spring_test_df = pd.merge(agg_by_prob_spring_test_df, get_avg_df, on=subsetCols)
	
	# agg_by_assign_spring_test_df = agg_by_prob_spring_test_df.drop_duplicates(subset=subsetCols2, keep="last")
	# for label in halstead_labels_2:
	# 	newlabel = label+"Fin"
	# 	get_avg_df = agg_by_prob_spring_test_df.groupby(subsetCols2)[label].mean().reset_index().rename(columns={label:newlabel})
	# 	agg_by_assign_spring_test_df = pd.merge(agg_by_assign_spring_test_df, get_avg_df, on=subsetCols2)
	
	# agg_by_std_spring_test_df = agg_by_assign_spring_test_df.drop_duplicates(subset=subsetCols3, keep="last")
	# for label in halstead_labels_3:
	# 	newlabel = label+"Comp"
	# 	get_avg_df = agg_by_std_spring_test_df.groupby(subsetCols3)[label].mean().reset_index().rename(columns={label:newlabel})
	# 	agg_by_std_spring_test_df = pd.merge(agg_by_std_spring_test_df, get_avg_df, on=subsetCols3)

	# spring_test_feats_aggragated_df = agg_by_std_spring_test_df[halstead_labels_fin]
	# spring_test_final_df = pd.merge(spring_test_feats_aggragated_df, subject_test_spring_df, on="SubjectID")

	# spring_test_final_df.to_csv('Spring_Test_Subject_With_Halstead.csv', encoding='utf-8', index=False)

	# ### Testing models
	# x_grades_pred_spring_linReg_df = trainLinearRegression(spring_train_final_df, spring_test_final_df, halstead_labels_4)
	# x_grades_pred_spring_linReg_df[['SubjectID', 'X-Grade']].to_csv('Spring_Predicted_LinReg_Grades.csv', encoding='utf-8', index=False)

	# x_grades_pred_spring_RidgeReg_df = trainRidgeRegression(spring_train_final_df, spring_test_final_df, halstead_labels_4)
	# x_grades_pred_spring_RidgeReg_df[['SubjectID', 'X-Grade']].to_csv('Spring_Predicted_RidgeReg_Grades.csv', encoding='utf-8', index=False)

	# x_grades_pred_spring_LassoReg_df = trainLassoRegression(spring_train_final_df, spring_test_final_df, halstead_labels_4)
	# x_grades_pred_spring_LassoReg_df[['SubjectID', 'X-Grade']].to_csv('Spring_Predicted_LassoReg_Grades.csv', encoding='utf-8', index=False)
	
	# x_grades_pred_spring_BayesRidgeReg_df = trainBayesRidgeRegression(spring_train_final_df, spring_test_final_df, halstead_labels_4)
	# x_grades_pred_spring_BayesRidgeReg_df[['SubjectID', 'X-Grade']].to_csv('Spring_Predicted_BayesRidgeReg_Grades.csv', encoding='utf-8', index=False)

	# x_grades_pred_spring_SVReg_df = trainSVRegression(spring_train_final_df, spring_test_final_df, halstead_labels_4)
	# x_grades_pred_spring_SVReg_df[['SubjectID', 'X-Grade']].to_csv('Spring_Predicted_SVReg_Grades.csv', encoding='utf-8', index=False)

	# x_grades_pred_spring_XGBReg_df = trainXGBRegression(spring_train_final_df, spring_test_final_df, halstead_labels_4)
	# x_grades_pred_spring_XGBReg_df[['SubjectID', 'X-Grade']].to_csv('Spring_Predicted_XGBReg_Grades.csv', encoding='utf-8', index=False)

	# x_grades_pred_spring_RFReg_df = trainRFRegression(spring_train_final_df, spring_test_final_df, halstead_labels_4)
	# x_grades_pred_spring_RFReg_df[['SubjectID', 'X-Grade']].to_csv('Spring_Predicted_RFReg_Grades.csv', encoding='utf-8', index=False)

	# x_grades_pred_spring_KNNReg_df = trainKNNRegression(spring_train_final_df, spring_test_final_df, halstead_labels_4)
	# x_grades_pred_spring_KNNReg_df[['SubjectID', 'X-Grade']].to_csv('Spring_Predicted_KNNReg_Grades.csv', encoding='utf-8', index=False)

####-------------------------------------------------------
	# Testing Models by combining both training sets for fitting 
	print("------------------------------------------------\n")
	spring_train_final_copy_df = spring_train_final_df
	spring_train_final_copy_df['X-Grade'] = spring_train_final_copy_df['X-Grade']*100
	spring_train_final_copy_df.to_csv('Spring_Train_Scaled_Grades.csv', encoding='utf-8', index=False)
	total_train_df = pd.concat([fall_train_final_df, spring_train_final_copy_df])

	grade_pred_combined_fall_SVReg_df = trainSVRegression(total_train_df, fall_test_final_df, halstead_labels_4)
	grade_pred_combined_fall_SVReg_df[['SubjectID', 'X-Grade']].to_csv('Combined_Fall_Predicted_SVReg_Grades.csv', encoding='utf-8', index=False)

	grade_pred_combined_fall_RFReg_df = trainRFRegression(total_train_df, fall_test_final_df, halstead_labels_4)
	grade_pred_combined_fall_RFReg_df[['SubjectID', 'X-Grade']].to_csv('Combined_Fall_Predicted_RFReg_Grades.csv', encoding='utf-8', index=False)

	grade_pred_combined_fall_KNNReg_df = trainKNNRegression(total_train_df, fall_test_final_df, halstead_labels_4)
	grade_pred_combined_fall_KNNReg_df[['SubjectID', 'X-Grade']].to_csv('Combined_Fall_Predicted_KNNReg_Grades.csv', encoding='utf-8', index=False)

	grade_pred_combined_fall_BagReg_df = trainBaggingRegression(total_train_df, fall_test_final_df, halstead_labels_4)
	grade_pred_combined_fall_BagReg_df[['SubjectID', 'X-Grade']].to_csv('Combined_Fall_Predicted_BagReg_Grades.csv', encoding='utf-8', index=False)

	# print(tune_model(fall_train_final_df, fall_test_final_df, halstead_labels_4))

	new_agg_final_df = pd.merge(new_agg_df, subject_train_fall_df, on='SubjectID')


	### Extended Features Datsets Fall Set -----
	#
	extended_feats_fall_train_df = pd.read_csv("data\\Features\\trainData_extended_features.csv")
	extended_feats_fall_test_df = pd.read_csv("data\\Features\\testData_extended_features.csv")
	
	fall_test_subject_col = fall_test_final_df['SubjectID']
	extended_feats_fall_test_df['SubjectID'] = fall_test_subject_col

	extended_feats_labels = ['problemAttempted', 'NumCorrectEventually', 'totalAttempts', 'NumCorrectFirstTry'] + halstead_labels_4

	x_grades_pred_fall_linReg_df = trainLinearRegression(extended_feats_fall_train_df, extended_feats_fall_test_df, extended_feats_labels)
	x_grades_pred_fall_linReg_df[['SubjectID', 'X-Grade']].to_csv('Predict_Fall_Extended_Train_LinReg.csv', encoding='utf-8', index=False)

	x_grades_pred_fall_Ridge_df = trainRidgeRegression(extended_feats_fall_train_df, extended_feats_fall_test_df, extended_feats_labels)
	x_grades_pred_fall_Ridge_df[['SubjectID', 'X-Grade']].to_csv('Predict_Fall_Extended_Train_RidgeReg.csv', encoding='utf-8', index=False)

	x_grades_pred_fall_RFReg_df = trainRFRegression(extended_feats_fall_train_df, extended_feats_fall_test_df, extended_feats_labels)
	x_grades_pred_fall_RFReg_df[['SubjectID', 'X-Grade']].to_csv('Predict_Fall_Extended_Train_RFReg.csv', encoding='utf-8', index=False)

	x_grades_pred_fall_KNN_df = trainKNNRegression(extended_feats_fall_train_df, extended_feats_fall_test_df, extended_feats_labels)
	x_grades_pred_fall_KNN_df[['SubjectID', 'X-Grade']].to_csv('Predict_Fall_Extended_Train_KNNReg.csv', encoding='utf-8', index=False)

	x_grades_pred_fall_SVR_df = trainSVRegression(extended_feats_fall_train_df, extended_feats_fall_test_df, halstead_labels_4)
	x_grades_pred_fall_SVR_df[['SubjectID', 'X-Grade']].to_csv('Predict_Fall_Extended_Train_SVReg.csv', encoding='utf-8', index=False)

	x_grades_pred_fall_XGB_df = trainXGBRegression(extended_feats_fall_train_df, extended_feats_fall_test_df, extended_feats_labels)
	x_grades_pred_fall_XGB_df[['SubjectID', 'X-Grade']].to_csv('Predict_Fall_Extended_Train_XGBReg.csv', encoding='utf-8', index=False)

	x_grades_pred_fall_BagReg_df = trainBaggingRegression(extended_feats_fall_train_df, extended_feats_fall_test_df, extended_feats_labels)
	x_grades_pred_fall_BagReg_df[['SubjectID', 'X-Grade']].to_csv('Predict_Fall_Extended_Train_BagReg.csv', encoding='utf-8', index=False)

### end of main()
###--------------------------------------------------
def trainLinearRegression(train_set_df, x_test_set_df, featureLabels):
	x_train_set = train_set_df[featureLabels]
	x_test_set = x_test_set_df[featureLabels]
	y_train_actual = train_set_df['X-Grade']
	lin_reg_model = LinearRegression()
	lin_reg_model.fit(x_train_set, y_train_actual)
	r_sq = lin_reg_model.score(x_train_set, y_train_actual)
	print(r_sq)
	y_pred_faux = lin_reg_model.predict(x_train_set)
	print(mean_squared_error(y_train_actual, y_pred_faux))
	print('')
	y_pred = lin_reg_model.predict(x_test_set)
	predict_x_grades_df = x_test_set_df
	predict_x_grades_df['X-Grade'] = y_pred 
	return predict_x_grades_df

def trainRidgeRegression(train_set_df, x_test_set_df, featureLabels):
	x_train_set = train_set_df[featureLabels]
	x_test_set = x_test_set_df[featureLabels]
	y_train_actual = train_set_df['X-Grade']
	ridge_reg_model = Ridge()
	ridge_reg_model.fit(x_train_set, y_train_actual)
	r_sq = ridge_reg_model.score(x_train_set, y_train_actual)
	print(r_sq)
	y_pred_faux = ridge_reg_model.predict(x_train_set)
	print(mean_squared_error(y_train_actual, y_pred_faux))
	print('')
	y_pred = ridge_reg_model.predict(x_test_set)
	predict_x_grades_df = x_test_set_df
	predict_x_grades_df['X-Grade'] = y_pred 
	return predict_x_grades_df

def trainLassoRegression(train_set_df, x_test_set_df, featureLabels):
	x_train_set = train_set_df[featureLabels]
	x_test_set = x_test_set_df[featureLabels]
	y_train_actual = train_set_df['X-Grade']
	lasso_reg_model = Lasso()
	lasso_reg_model.fit(x_train_set, y_train_actual)
	r_sq = lasso_reg_model.score(x_train_set, y_train_actual)
	print(r_sq)
	y_pred_faux = lasso_reg_model.predict(x_train_set)
	print(mean_squared_error(y_train_actual, y_pred_faux))
	print('')
	y_pred = lasso_reg_model.predict(x_test_set)
	predict_x_grades_df = x_test_set_df
	predict_x_grades_df['X-Grade'] = y_pred 
	return predict_x_grades_df

def trainBayesRidgeRegression(train_set_df, x_test_set_df, featureLabels):
	x_train_set = train_set_df[featureLabels]
	x_test_set = x_test_set_df[featureLabels]
	y_train_actual = train_set_df['X-Grade']
	bayes_reg_model = BayesianRidge()
	bayes_reg_model.fit(x_train_set, y_train_actual)
	r_sq = bayes_reg_model.score(x_train_set, y_train_actual)
	print(r_sq)
	y_pred_faux = bayes_reg_model.predict(x_train_set)
	print(mean_squared_error(y_train_actual, y_pred_faux))
	print('')
	y_pred = bayes_reg_model.predict(x_test_set)
	predict_x_grades_df = x_test_set_df
	predict_x_grades_df['X-Grade'] = y_pred 
	return predict_x_grades_df

def trainSVRegression(train_set_df, x_test_set_df, featureLabels):
	set_01236 = ['n1AvgFinComp', 'n2AvgFinComp', 'N1AvgFinComp', 'N2AvgFinComp', 'n_hatAvgFinComp']
	set_012368 = ['n1AvgFinComp', 'n2AvgFinComp', 'N1AvgFinComp', 'N2AvgFinComp', 'n_hatAvgFinComp', 'DAvgFinComp']
	x_train_set = train_set_df[featureLabels]
	x_test_set = x_test_set_df[featureLabels]
	y_train_actual = train_set_df['X-Grade']
	svr_reg_model = SVR()
	svr_reg_model.fit(x_train_set, y_train_actual)
	r_sq = svr_reg_model.score(x_train_set, y_train_actual)
	print("R^2 Value for SVR: {}".format(r_sq))
	y_pred_faux = svr_reg_model.predict(x_train_set)
	print(mean_squared_error(y_train_actual, y_pred_faux))
	print('')
	y_pred = svr_reg_model.predict(x_test_set)
	predict_x_grades_df = x_test_set_df
	predict_x_grades_df['X-Grade'] = y_pred 
	return predict_x_grades_df

def trainXGBRegression(train_set_df, x_test_set_df, featureLabels):
	x_train_set = train_set_df[featureLabels]
	x_test_set = x_test_set_df[featureLabels]
	y_train_actual = train_set_df['X-Grade']
	xgb_reg_model = XGBRegressor()
	xgb_reg_model.fit(x_train_set, y_train_actual)
	r_sq = xgb_reg_model.score(x_train_set, y_train_actual)
	print(r_sq)
	y_pred_faux = xgb_reg_model.predict(x_train_set)
	print(mean_squared_error(y_train_actual, y_pred_faux))
	print('')
	y_pred = xgb_reg_model.predict(x_test_set)
	predict_x_grades_df = x_test_set_df
	predict_x_grades_df['X-Grade'] = y_pred 
	return predict_x_grades_df

def trainRFRegression(train_set_df, x_test_set_df, featureLabels):
	x_train_set = train_set_df[featureLabels]
	x_test_set = x_test_set_df[featureLabels]
	y_train_actual = train_set_df['X-Grade']
	rf_reg_model = RandomForestRegressor(n_estimators = 100, random_state = 0)
	rf_reg_model.fit(x_train_set, y_train_actual)
	r_sq = rf_reg_model.score(x_train_set, y_train_actual)
	print(r_sq)
	y_pred_faux = rf_reg_model.predict(x_train_set)
	print(mean_squared_error(y_train_actual, y_pred_faux))
	print('')
	y_pred = rf_reg_model.predict(x_test_set)
	predict_x_grades_df = x_test_set_df
	predict_x_grades_df['X-Grade'] = y_pred 
	importance = rf_reg_model.feature_importances_
	for i,v in enumerate(importance):
		print('Feature: %0d, Score: %.5f' % (i,v))
	return predict_x_grades_df

def trainKNNRegression(train_set_df, x_test_set_df, featureLabels):
	x_train_set = train_set_df[featureLabels]
	x_test_set = x_test_set_df[featureLabels]
	y_train_actual = train_set_df['X-Grade']
	knn_reg_model = KNeighborsRegressor(n_neighbors=9, weights='distance')
	knn_reg_model.fit(x_train_set, y_train_actual)
	r_sq = knn_reg_model.score(x_train_set, y_train_actual)
	print(r_sq)
	y_pred_faux = knn_reg_model.predict(x_train_set)
	print(mean_squared_error(y_train_actual, y_pred_faux))
	print('')
	y_pred = knn_reg_model.predict(x_test_set)
	predict_x_grades_df = x_test_set_df
	predict_x_grades_df['X-Grade'] = y_pred 
	return predict_x_grades_df

def trainBaggingRegression(train_set_df, x_test_set_df, featureLabels):
	x_train_set = train_set_df[featureLabels]
	x_test_set = x_test_set_df[featureLabels]
	y_train_actual = train_set_df['X-Grade']
	bag_reg_model = BaggingRegressor(base_estimator=SVR(),
						n_estimators=10, random_state=0).fit(x_train_set, y_train_actual)
	y_pred = bag_reg_model.predict(x_test_set)
	predict_x_grades_df = x_test_set_df
	predict_x_grades_df['X-Grade'] = y_pred 
	return predict_x_grades_df

def trainStackingRegression(train_set_df, x_test_set_df, featureLabels):
	x_train_set = train_set_df[featureLabels]
	x_test_set = x_test_set_df[featureLabels]
	y_train_actual = train_set_df['X-Grade']
	estimators = [
		('rf', RandomForestRegressor()),
		('knr', SVR())
	]
	regmodel = StackingRegressor(
		estimators=estimators,
		final_estimator=BaggingRegressor(base_estimator=SVR(),
						n_estimators=10, random_state=0)
	)
	regmodel = regmodel.fit(x_train_set, y_train_actual)
	y_pred = regmodel.predict(x_test_set)
	predict_x_grades_df = x_test_set_df
	predict_x_grades_df['X-Grade'] = y_pred 
	return predict_x_grades_df

def trainVotingRegression(train_set_df, x_test_set_df, featureLabels):
	x_train_set = train_set_df[featureLabels]
	x_test_set = x_test_set_df[featureLabels]
	y_train_actual = train_set_df['X-Grade']
	r1 = SVR()
	r2 = BaggingRegressor(base_estimator=SVR(),
						n_estimators=10, random_state=0)
	votereg = VotingRegressor([('svrbase', r1), ('svrbag', r2)])
	votereg = votereg.fit(x_train_set, y_train_actual)
	
	y_pred = votereg.predict(x_test_set)
	predict_x_grades_df = x_test_set_df
	predict_x_grades_df['X-Grade'] = y_pred 
	return predict_x_grades_df

def trainPassiveAggressiveRegressor(train_set_df, x_test_set_df, featureLabels):
	x_train_set = train_set_df[featureLabels]
	x_test_set = x_test_set_df[featureLabels]
	y_train_actual = train_set_df['X-Grade']
	passagg_model = PassiveAggressiveRegressor(max_iter=500, random_state=0, tol=1e-3, early_stopping=False, validation_fraction=.1)
	passagg_model.fit(x_train_set, y_train_actual)
	r_sq = passagg_model.score(x_train_set, y_train_actual)
	print(r_sq)
	y_pred_faux = passagg_model.predict(x_train_set)
	print(mean_squared_error(y_train_actual, y_pred_faux))
	print('')
	y_pred = passagg_model.predict(x_test_set)
	predict_x_grades_df = x_test_set_df
	predict_x_grades_df['X-Grade'] = y_pred 
	return predict_x_grades_df

def trainHuberRegressor(train_set_df, x_test_set_df, featureLabels):
	x_train_set = train_set_df[featureLabels]
	x_test_set = x_test_set_df[featureLabels]
	y_train_actual = train_set_df['X-Grade']
	model = HuberRegressor()
	model.fit(x_train_set, y_train_actual)
	r_sq = model.score(x_train_set, y_train_actual)
	print(r_sq)
	y_pred = model.predict(x_test_set)
	predict_x_grades_df = x_test_set_df
	predict_x_grades_df['X-Grade'] = y_pred 
	return predict_x_grades_df

# tuning each model, will change for each one
def tune_model(train_set_df, x_test_set_df, featureLabels):
	X = train_set_df[featureLabels]
	y = train_set_df['X-Grade']
	gsc = GridSearchCV(
		estimator=SVR(kernel='rbf'),
		param_grid={
			'C': [0.1, 1, 2, 5, 10, 50, 100, 200, 500, 1000],
			'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
			'gamma': [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 1, 3, 5]
		},
		cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

	grid_result = gsc.fit(X, y)
	best_params = grid_result.best_params_
	print("Best parameters for rbf kernel: ")
	print(best_params)
	best_svr = SVR(kernel='rbf', C=best_params["C"], epsilon=best_params["epsilon"], gamma=best_params["gamma"],
				   coef0=0.1, shrinking=True,
				   tol=0.001, cache_size=200, verbose=False, max_iter=-1)

	scoring = {
			   'abs_error': 'neg_mean_absolute_error',
			   'squared_error': 'neg_mean_squared_error'}

	scores = cross_validate(best_svr, X, y, cv=10, scoring=scoring, return_train_score=True)
	return "MAE :", abs(scores['test_abs_error'].mean()), "| MSE :", abs(scores['test_squared_error'].mean())

main()

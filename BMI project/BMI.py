import numpy as np
import pandas as pd
from person import person
from BMIHelper import model
from BMIHelper import Read_Data
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline
import os
import time
from multiprocessing import Process
from scipy import interpolate


def convert_dict(np_BMI_agemos,np_BMI_id,np_BMI_bmi):
    dict_bmi = {}
    for i in range(np_BMI_id.shape[0]):
        if np_BMI_id[i] in dict_bmi: 
            dict_bmi[np_BMI_id[i]]["month"].append(np_BMI_agemos[i])
            dict_bmi[np_BMI_id[i]]["bmi"].append(np_BMI_bmi[i])
        else:
            dict_bmi[np_BMI_id[i]] = {"month":[np_BMI_agemos[i]],"bmi":[np_BMI_bmi[i]]}
    

    return dict_bmi

def convert_data(dict_bmi):
	for item in list(dict_bmi.keys()):
		month = np.array(dict_bmi[item]['month'])
		bmi = np.array(dict_bmi[item]['bmi'])
		dict_bmi[item]['data'] = np.vstack((bmi,month))
	return dict_bmi

def get_person_info(dict_bmi):
	people = []
	list_of_list = []
	for item in list(dict_bmi.keys()):
		ppl = person(dict_bmi[item]['data'])
		people.append(ppl)
		list_of_list.append([item,ppl.bmi,ppl.bmi_start,ppl.bmi_end,ppl.start,ppl.end,ppl.agemos])
	df = pd.DataFrame(list_of_list,columns =['id','bmi','bmi_start','bmi_end','start','end','agemos'])

	return people, df

def cubic_spline_equal_parse(agemos,bmi):
	try:
		spl = UnivariateSpline(agemos,bmi)
		dx = np.linspace(24,240,217)
		x = spl(dx)
	except:
		x = np.zeros(217)
		return x
	return x

def compute_matrix(agemos,bmi):
	f = interpolate.interp1d(agemos, bmi)
	xnew = np.linspace(55,219,165)
	ynew = f(xnew)
	return ynew


def do_Work(df_train,df_original,df_modified,subprocess):
	data = np.zeros((1,165))
	
	less_data_index = []
	counter = 0
	bmi_filled_dict = {}
	for i in df_train.index:
		ID = df_modified['id'][i]
		agemos, bmi = model.concatenate(df_original,df_modified,df_modified['agemos'][i],df_modified['bmi'][i],ID)
		if len(agemos) < 10:
			less_data_index.append(i+1)
		#bmi = cubic_spline_equal_parse(agemos,bmi)
		#data = np.vstack((data,bmi))
		bmi_filled_dict[ID] = [np.vstack((agemos,bmi))]
		counter += 1

		if max(agemos) < 220:
			continue
		bmi_interploted = compute_matrix(agemos,bmi)
		data = np.vstack((data,bmi_interploted))

		if counter % 50 == 0:
			print('---------------------------{}-------------------------'.format(counter))
			print("--- %s seconds ---" % (time.time() - start_time))
			print('------{}-----'.format(subprocess) )

		if counter > 5:
			break

	data = np.delete(data,less_data_index,0)
	np.save('data' + str(subprocess),data)
	
	df = pd.DataFrame.from_dict(bmi_filled_dict)
	file_name = 'bmi' + str(subprocess) + '.csv'
	df.T.to_csv(file_name, encoding='utf-8')

def parse_df(df,num):
	data_parsed_list = []
	for i in range(num):
		data_parsed_list.append(np.array_split(df, num)[i])
	return data_parsed_list


def distributed_computing(num_process,df_train,df_original,df_modified):
	num_subprocess = [i + 1 for i in range(num_process)]
	process = []
	for i in range(num_process):
		p = Process(target=do_Work, args=(df_train[i],df_original,df_modified,num_subprocess[i]))
		p.start()
		process.append(p)

	for i in range(len(process)):
		process[i].join()


if __name__ == '__main__':
	start_time = time.time()
	#read data
	BMIREF = pd.read_csv('CDCref_withClass_23Aug18.csv')
	BMI = pd.read_csv('class_assign_23aug18.csv')

	BMIREF = BMIREF.sort_values(by = ['id','agemos'])
	BMIREF = BMIREF.drop_duplicates(subset=['id', 'agemos'],keep = 'first')

	#split data by male and female
	BMIREFMale = BMIREF[BMIREF["sex"] == 1]
	BMIREFMale = BMIREFMale[["id","agemos","bmi"]]
	BMIREFFemale = BMIREF[BMIREF["sex"] == 2]
	BMIREFFemale = BMIREFFemale[["id","agemos","bmi"]]

	#Male
	np_BMI_Male_agemos = np.array(BMIREFMale["agemos"])
	np_BMI_Male_id = np.array(BMIREFMale["id"])
	np_BMI_Male_bmi = np.array(BMIREFMale["bmi"])

	#Female
	np_BMI_Female_agemos = np.array(BMIREFFemale["agemos"])
	np_BMI_Female_id = np.array(BMIREFFemale["id"])
	np_BMI_Female_bmi = np.array(BMIREFFemale["bmi"])

	#convert data into matrix
	dict_bmi_male = convert_dict(np_BMI_Male_agemos,np_BMI_Male_id,np_BMI_Male_bmi)
	dict_bmi_female = convert_dict(np_BMI_Female_agemos,np_BMI_Female_id,np_BMI_Female_bmi)


	dict_bmi_male = convert_data(dict_bmi_male)
	people_male, bmi_male_df = get_person_info(dict_bmi_male)

	dict_bmi_female = convert_data(dict_bmi_female)
	people_female, bmi_female_df = get_person_info(dict_bmi_female)


	model = model.trainning_model()
	
	bmi_male_df_train = bmi_male_df[(bmi_male_df['start'] > 24) & (bmi_male_df['start'] < 54)]
	bmi_female_df_train = bmi_female_df[(bmi_female_df['start'] > 24) & (bmi_female_df['start'] < 54)]


	# bmi_male_df_train_1 = np.array_split(bmi_male_df_train, 8)[0]
	# bmi_male_df_train_2 = np.array_split(bmi_male_df_train, 8)[1]
	# bmi_male_df_train_3 = np.array_split(bmi_male_df_train, 8)[2]
	# bmi_male_df_train_4 = np.array_split(bmi_male_df_train, 8)[3]
	# bmi_male_df_train_5 = np.array_split(bmi_male_df_train, 8)[4]
	# bmi_male_df_train_6 = np.array_split(bmi_male_df_train, 8)[5]
	# bmi_male_df_train_7 = np.array_split(bmi_male_df_train, 8)[6]
	# bmi_male_df_train_8 = np.array_split(bmi_male_df_train, 8)[7]
	
	# subprocess = 1
	# p1 = Process(target=do_Work, args=(bmi_male_df_train,BMIREFMale,bmi_male_df,subprocess))
	# p1.start()
	
	# subprocess = 2
	# p2 = Process(target=do_Work, args=(bmi_male_df_train,BMIREFMale,bmi_male_df,subprocess))
	# p2.start()
	
	# subprocess = 3
	# p3 = Process(target=do_Work, args=(bmi_male_df_train,BMIREFMale,bmi_male_df,subprocess))
	# p3.start()
	
	# subprocess = 4
	# p4 = Process(target=do_Work, args=(bmi_male_df_train,BMIREFMale,bmi_male_df,subprocess))
	# p4.start()




	data_list = parse_df(bmi_male_df_train,5)
	df_args = dict(df_train= data_list,df_original= BMIREFMale,df_modified=bmi_male_df)
	distributed_computing(5,**df_args)

	# Merge Data
	# data = Read_Data.read_data()
	# file_names = data.find_csv_filenames('Users/william/Desktop/PY36')
	# data = data.load_merge(file_names)
	# print(data.shape)
	
	# data = Read_Data.read_data()
	# file_names = data.find_npy_filenames('Users/william/Desktop/PY36')

	# data = data.load_np(file_names)
	# cap = int(data.shape[0]/3 * 2)
	# data_train = data[0:cap:1,::,None]
	# data_test = data[cap::1,::,None]
	# np.save('dat_train.npy',data_train)
	# np.save('dat_test.npy',data_test)






	# print(data['agemos_bmi'][1])
	# dat1 = data['agemos_bmi'][1]
	# # print(dat1[0])
	# # print(dat1[1])
	# f = interpolate.interp1d(dat1[0], dat1[1])
	# xnew = np.linspace(20,220)
	# ynew = f(xnew)
	# plt.plot(xnew,ynew)
	# plt.plot(dat1[0],dat1[1])
	# plt.show()













	# data = np.delete(data,less_data_index,0)
	# np.save('data.npy',data)



	# ID = bmi_male_df.iloc[408]['id']
	# i = list(bmi_male_df_train.index)[408]
	# x=bmi_male_df.iloc[408]['agemos']
	# y=bmi_male_df.iloc[408]['bmi']
	# # plt.plot(x,y)
	# # plt.show()
	# agemos, bmi = model.concatenate(BMIREFMale,bmi_male_df,bmi_male_df['agemos'][i],bmi_male_df['bmi'][i],ID)

	# bmi = cubic_spline_equal_parse(agemos,bmi)
	# data = np.vstack((data,bmi))
	# print(data)



	# df_train= bmi_male_df_train
	# df_original= BMIREFMale
	# df_modified=bmi_male_df
	# print(df_train)
	# for i in df_train.index:
	# 	print('hello')
	# 	ID = df_modified['id'][i]
	# 	agemos, bmi = model.concatenate(df_original,df_modified,df_modified['agemos'][i],df_modified['bmi'][i],ID)




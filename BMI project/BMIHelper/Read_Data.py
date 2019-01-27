import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from os import listdir

class read_data():
	
	def __init__(self):
		pass


	def find_csv_filenames(self,path_to_dir, suffix=".csv" ):
		filenames = listdir('/Users/william/Desktop/PY36')
		file_names = [ filename for filename in filenames if filename.endswith('.csv')]
		file_names = [ filename for filename in filenames if filename.startswith('bmi')]
		return file_names

	def find_npy_filenames(self,path_to_dir, suffix=".npy" ):
		filenames = listdir('/Users/william/Desktop/PY36')
		file_names = [ filename for filename in filenames if filename.endswith('.npy')]
		#file_names = [ filename for filename in filenames if filename.startswith('data')]
		return file_names

	def load_merge(self,file_names):

		for i in range(len(file_names)):
			if i == 0:		
				data_merge = pd.read_csv(file_names[i])
				data_merge.columns = ['index','agemos_bmi']
			else:
				data = pd.read_csv(file_names[i])
				data.columns = ['index','agemos_bmi']
				data_merge = data_merge.append(data,ignore_index=True)

		return data_merge

	def load_np(self,file_names):

		for i in range(len(file_names)):
			if i == 0:		
				data_merge = np.load(file_names[i])
				data_merge = np.delete(data_merge,0,0)
			else:
				data = np.load(file_names[i])
				data = np.delete(data,0,0)
				data_merge = np.vstack((data_merge,data))

		return data_merge

#if __name__ == '__main__':
# 	filenames = listdir('/Users/william/Desktop/PY36')
# 	file_names = [ filename for filename in filenames if filename.endswith('.csv')]
# 	file_names = [ filename for filename in filenames if filename.startswith('bmi')]


# 	for i in range(len(file_names)):
# 		if i == 0:		
# 			data_merge = pd.read_csv(file_names[i])
# 			data_merge.columns = ['index','agemos_bmi']
# 		else:
# 			data = pd.read_csv(file_names[i])
# 			data.columns = ['index','agemos_bmi']
# 			data_merge = data_merge.append(data)




	#data1 = np.load("data.npy1.npy")
	# data2 = np.load("data2.npy")
# data3 = np.load("data3.npy")
# data4 = np.load("data4.npy")
# data5 = np.load("data5.npy")
# data6 = np.load("data6.npy")
# data1 = np.delete(data1,0,0)
# data2 = np.delete(data2,0,0)
# data3 = np.delete(data3,0,0)
# data4 = np.delete(data4,0,0)
# data5 = np.delete(data5,0,0)
# data6 = np.delete(data6,0,0)
# print(data1[1,])
# print(data2[1,])
# print(data3[1,])
# print(data4[1,])
# print(data5[1,])
# print(data6[1,])
	# print(data1)
	


# data1 = np.vstack((data1,data2))
# data1 = np.vstack((data1,data3))
# data1 = np.vstack((data1,data4))
# data1 = np.vstack((data1,data5))
# data1 = np.vstack((data1,data6))
# data1 = data1[...,None]
# print(data1.shape)
# for i in range(data.shape[0]):
# 	if i % 300 ==0:
# 		plt.plot(range(217),data[i,])

# plt.show()

# np.save('data.npy',data1)

# pickle_in = open("dict.pickle","rb")
# example_dict = pickle.load(pickle_in)
# print(example_dict)
#abc['ID'].append(ID2)




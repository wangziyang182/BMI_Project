import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
from scipy.interpolate import CubicSpline
from scipy import optimize
from scipy.interpolate import UnivariateSpline


class trainning_model():
	
	def __init__(self):
		pass	

	def get_target(self,x,y,x_pred):
		try:
			spl = UnivariateSpline(x, y)
			return spl(x_pred),spl
		except:
			pass


	def merge_bmi_seqence(self,X_left,Y_left,X_right,Y_right):
		X = np.hstack((X_left,X_right))
		Y = np.hstack((Y_left,Y_right))
		dict_list = {'X':X,'Y':Y}
		df = pd.DataFrame(dict_list)
		df = df.sort_values(by = ['X'])
		df = df.drop_duplicates(subset = ['X'], keep = "first")
		return np.array(list(df['X'])),np.array(list(df['Y']))


	def compute_squared_error(self,X_left,Y_left,X_right,Y_right):
		if X_left[-1] >= X_right[0]:
			dx = np.linspace(X_right[0] - 6, X_left[-1] + 6, 10)
		else:
			dx = np.linspace(X_left[-1] - 6, X_right[0] + 6, 10)

		cs1,spl1 = self.get_target(X_left,Y_left,dx)
		cs2,spl2 = self.get_target(X_right,Y_right,dx)

		time_difference_penalty = (X_left[-1] - X_right[0]) ** 2 * 0.05
		slope_difference_penalty = spl1.derivative(1)(dx) - spl2.derivative(1)(dx)
		slope_difference_penalty = np.sum(slope_difference_penalty.dot(slope_difference_penalty))
		distance_penalty = np.sum((cs1 - cs2).dot(cs1 - cs2))

		error = distance_penalty + slope_difference_penalty * 20 + time_difference_penalty
		
		return error

	def concatenate(self,df_original,df_modified,agemos,bmi,id):
		last = 0
		data_Frame_original = df_original.copy()
		data_Frame_modified = df_modified.copy()
		if agemos[0] <=54 and agemos[-1] > 230:
			return agemos, bmi
		while last <= 230:
			error = float('inf')
			temp_error = float('inf')
			indexx = 0
			last = agemos[-1]
			range_Min = last - 6
			range_Max = last + 6
			
			data_Frame_original_train = data_Frame_original[(data_Frame_original['agemos'] >= range_Min) & (data_Frame_original['agemos'] <= range_Max)]
			data_Frame_original_train = data_Frame_original_train[data_Frame_original_train['id'] != id]
			id_list = list(data_Frame_original_train.id.unique())
			data_Frame_modified_train = data_Frame_modified[data_Frame_modified['id'].isin(id_list)]
			
			X_left = agemos
			Y_left = bmi

			for index in data_Frame_modified_train.index:

				X_right = data_Frame_modified_train['agemos'][index]
				Y_right = data_Frame_modified_train['bmi'][index]
				# print(X_right)

				if X_right[-1] <= last:
					continue
				if X_right[0] <= X_left[-1]-50:
					continue

				try:
					temp_error = self.compute_squared_error(X_left,Y_left,X_right,Y_right)
					
				except:
					pass

				# print(temp_error)
				if temp_error < error:
					indexx = index
					error = temp_error

			try:
				X_right = data_Frame_modified_train['agemos'][indexx]
				Y_right = data_Frame_modified_train['bmi'][indexx]
				plt.plot(X_left,Y_left)
				plt.plot(X_right,Y_right)
				plt.show()
				agemos,bmi = self.merge_bmi_seqence(agemos,bmi,X_right,Y_right)
				data_Frame_modified = data_Frame_modified[data_Frame_modified['id'] != data_Frame_modified_train['id'][indexx]]
				last = agemos[-1]

			except:
				break

		return agemos,bmi


	def train(self,df):
		pass
				
	def newton_interpolation():
		pass

	def find_derivative_end(self):
		pass

	def predict_trajectory(bmi_array,):
		pass




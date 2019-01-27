import numpy as np

class person():
	def __init__(self,bmi_array):
		self.bmi = bmi_array[0,:]
		self.bmi_start = bmi_array[0,0]
		self.bmi_end = bmi_array[0,bmi_array.shape[1] - 1]
		self.start = bmi_array[1,0]
		self.end = bmi_array[1,bmi_array.shape[1] - 1]
		self.agemos = bmi_array[1,:]

	def find_derivative_start(self):
		return np.gradient(self.bmi_array)[0]

	def find_derivative_end(self):
		return np.gradient(self.bmi_array)[-1]


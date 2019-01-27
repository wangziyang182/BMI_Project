import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def convert_dict(np_BMI_Male_agemos,np_BMI_Male_id,np_BMI_Male_bmi):
    dict_bmi = {}
    for i in range(np_BMI_Male_id.shape[0]):
        if np_BMI_Male_id[i] in dict_bmi:
            dict_bmi[np_BMI_Male_id[i]]["month"].append(np_BMI_Male_agemos[i])
            dict_bmi[np_BMI_Male_id[i]]["bmi"].append(np_BMI_Male_bmi[i])
        else:
            dict_bmi[np_BMI_Male_id[i]] = {"month":[np_BMI_Male_agemos[i]],"bmi":[np_BMI_Male_bmi[i]]}
            
    return dict_bmi
    

def convert_matrix(dict_bmi,np_BMI_agemos):
    matrix = np.zeros((len(dict_bmi.keys()),int(max((np_BMI_Male_agemos))) + 1))
    count = np.ones((len(dict_bmi.keys()),int(max((np_BMI_Male_agemos))) + 1))
    keys = list(dict_bmi.keys())
    for i in range(len(keys)):
        for j in range(len(dict_bmi[keys[i]]["month"])):
            if dict_bmi[keys[i]]["month"][j] >= int(dict_bmi[keys[i]]["month"][j]) + 0.5:
                if matrix[i][int(dict_bmi[keys[i]]["month"][j]) + 1] == 0:
                    matrix[i][int(dict_bmi[keys[i]]["month"][j]) + 1] = dict_bmi[keys[i]]["bmi"][j]
                else:
                    count[i][int(dict_bmi[keys[i]]["month"][j]) + 1] = count[i][int(dict_bmi[keys[i]]["month"][j]) + 1] + 1
                    matrix[i][int(dict_bmi[keys[i]]["month"][j]) + 1] = (matrix[i][int(dict_bmi[keys[i]]["month"][j]) + 1]\
                                                                        + dict_bmi[keys[i]]["bmi"][j])
            else:
                if matrix[i][int(dict_bmi[keys[i]]["month"][j])] == 0:
                     matrix[i][int(dict_bmi[keys[i]]["month"][j])] = dict_bmi[keys[i]]["bmi"][j]
                else:
                #matrix[i][int(dict_bmi[keys[i]]["month"][j])] != 0:
                    count[i][int(dict_bmi[keys[i]]["month"][j])] = count[i][int(dict_bmi[keys[i]]["month"][j])] + 1
                    matrix[i][int(dict_bmi[keys[i]]["month"][j])] = (matrix[i][int(dict_bmi[keys[i]]["month"][j])]\
                                                                        + dict_bmi[keys[i]]["bmi"][j])
    matrix = matrix/ count
    return matrix
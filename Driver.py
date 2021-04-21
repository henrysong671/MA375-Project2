import pandas as pd #requires openpyxl and xlrd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def find_RSS(y_original, y_modeled):
    return np.sum(np.square(y_original - y_modeled))

def import_data(path, sheet):
    return pd.read_excel(path, sheet_name=sheet)

# Imports data from specified excel spreadsheet
file_path = './data/MA375_SP21_Curve_Fitting_Data_samples.xlsx'     # filepath of excel spreadsheet
sheet_name = 'Length Wt of rabbitfish'      # sheetname of spreadsheet
d = import_data(file_path, sheet_name)
data = d.to_numpy()     # converts to 2D array from pandas dataframe

# separates 2D array into 2 separate lists
x = []
y = []
for i in range(len(data)): 
    x.append(data[i][0])
    y.append(data[i][1])


# plots
plt.title("Project #2")
plt.plot(x, y, 'o', label='original')
plt.legend()
plt.show()

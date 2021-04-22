import pandas as pd #requires openpyxl and xlrd
import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt

def find_RSS(y_original, y_modeled):
    return np.sum(np.square(y_original - y_modeled))

def import_data(path, sheet):
    return pd.read_excel(path, sheet_name=sheet)

def linear(m, b, x): return m*x + b
def quadratic(a, b, c, x): return a*(x**2)+b*x+c
def exponential(a, b, x): return a*np.exp(b*x)
def logarithmic(a, b, x): return a + b*np.log(1*x)

def model_data(x, var, f_type):
    array = []
    if f_type == "linear":
        for i in x: array.append(linear(var[0], var[1], i))
    elif f_type == "quadratic":
        for i in x: array.append(quadratic(var[0], var[1], var[2], i))
    elif f_type == "exponential":
        for i in x: array.append(exponential(var[0], var[1], i))
    elif f_type == "logarithmic":
        for i in x: array.append(logarithmic(var[0], var[1], i))
    return array

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

x = np.array(x)
y = np.array(y)

popt, pcov = sp.curve_fit(linear, x, y)
m, b = popt
linear_model = model_data(x, [b, m], f_type="linear")

popt, pcov = sp.curve_fit(quadratic, x, y)
a, b, c = popt
quadratic_model = model_data(x, [b, a, c], f_type="quadratic")

popt, pcov = sp.curve_fit(exponential, x, y)
a, b = popt
exponential_model = model_data(x, [a, b], f_type="exponential")

popt, pcov = sp.curve_fit(logarithmic, x, y)
a, b = popt
logarithmic_model = model_data(x, [a, b], f_type="logarithmic")

print("Sum of the Squares of the Residuals")
print("1. Linear: ", find_RSS(y, linear_model))
print("2. Quadratic: ", find_RSS(y, quadratic_model))
print("3. Exponential: ", find_RSS(y, exponential_model))
print("4. Logarithmic: ", find_RSS(y, logarithmic_model))

# plots
plt.title("Project #2")
plt.plot(x, y, 'o', label='original')
#plt.plot(x, linear_model, 'g--', label='linear')
plt.plot(x, quadratic_model, 'r--', label='quadratic')
#plt.plot(x, exponential_model, 'y--', label='exponential')
#plt.plot(x, logarithmic_model, 'b--', label='logarithmic')
plt.legend()
plt.show()

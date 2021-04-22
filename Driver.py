import pandas as pd #requires openpyxl and xlrd
import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt

def find_RSS(y_original, y_modeled):
    return np.sum(np.square(y_original - y_modeled))

def import_data(path, sheet):
    return pd.read_excel(path, sheet_name=sheet)

def linear(x, m, b): return m*x + b
def quadratic(x, a, b, c): return a*(x**2)+b*x+c
def cubic(x, a, b, c, d): return a*(x**3)+b*(x**2)+c*x+d
def fourth_degree(x, a, b, c, d, e): return a*(x**4)+b*(x**3)+c*(x**2)+d*x+e
def fifth_degree(x, a, b, c, d, e, f): return a*(x**5)+b*(x**4)+c*(x**3)+d*(x**2)+e*x+f
def sixth_degree(x, a, b, c, d, e, f, g): return a*(x**6)+b*(x**5)+c*(x**4)+d*(x**3)+e*(x**2)+f*x+g
def exponential(x, a, b): return a*np.exp(b*x)
def logarithmic(x, a, b): return a + b*np.log(1*x)

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

# converts python lists into numpy arrays
x = np.array(x)
y = np.array(y)

# linear curve fitting & data
popt, pcov = sp.curve_fit(linear, x, y)
linear_model = linear(x, *popt)

# quadratic curve fitting & data
popt, pcov = sp.curve_fit(quadratic, x, y)
quadratic_model = quadratic(x, *popt)

# cubic curve fitting & data
popt, pcov = sp.curve_fit(cubic, x, y)
cubic_model = cubic(x, *popt)

# fourth degree polynomial curve fitting & data
popt, pcov = sp.curve_fit(fourth_degree, x, y)
fourth_model = fourth_degree(x, *popt)

# fifth degree polynomial curve fitting & data
popt, pcov = sp.curve_fit(fifth_degree, x, y)
fifth_model = fifth_degree(x, *popt)

# sixth degree polynomial curve fitting & data
popt, pcov = sp.curve_fit(sixth_degree, x, y)
sixth_model = sixth_degree(x, *popt)

# exponential curve fitting & data
popt, pcov = sp.curve_fit(exponential, x, y)
exponential_model = exponential(x, *popt)

# logarithmic curve fitting & data
popt, pcov = sp.curve_fit(logarithmic, x, y)
logarithmic_model = logarithmic(x, *popt)

# prints RSS values in terminal
print()
print("Sum of the Squares of the Residuals")
print("1. Linear: \t\t", find_RSS(y, linear_model))
print("2. Quadratic: \t\t", find_RSS(y, quadratic_model))
print("3. Cubic: \t\t", find_RSS(y, cubic_model))
print("4. 4th˚ Polynomial: \t", find_RSS(y, fourth_model))
print("5. 5th˚ Polynomial: \t", find_RSS(y, fifth_model))
print("6. 6th˚ Polynomial: \t", find_RSS(y, sixth_model))
print("7. Exponential: \t", find_RSS(y, exponential_model))
print("8. Logarithmic: \t", find_RSS(y, logarithmic_model))
print()

# plots
plt.title("Project #2")
plt.plot(x, y, 'ko', label='original')

# linear model
# plt.plot(x, linear_model, 'g--', label='linear')
# plt.text(18, 80, 'Linear RSS = %0.4f' % find_RSS(y, linear_model), color='g')

# quadratic model
# plt.plot(x, quadratic_model, 'r--', label='quadratic')
# plt.text(17.5, 80, 'Quadratic RSS = %0.4f' % find_RSS(y, quadratic_model), color='r')

# cubic model
# plt.plot(x, cubic_model, 'm--', label='cubic')
# plt.text(18, 80, 'Cubic RSS = %0.4f' % find_RSS(y, cubic_model), color='m')

# fourth degree polynomial model
# plt.plot(x, fourth_model, 'm--', label='4th degree polynomial')
# plt.text(17.5, 80, '4th degree RSS = %0.4f' % find_RSS(y, fourth_model), color='m')

# fifth degree polynomial model
# plt.plot(x, fifth_model, 'm--', label='5th degree polynomial')
# plt.text(17.5, 80, '5th degree RSS = %0.4f' % find_RSS(y, fifth_model), color='m')

# sixth degree polynomial model
# plt.plot(x, sixth_model, 'm--', label='6th degree polynomial')
# plt.text(17.5, 80, '6th degree RSS = %0.4f' % find_RSS(y, sixth_model), color='m')

# exponential model
# plt.plot(x, exponential_model, 'y--', label='exponential')
# plt.text(17.5, 80, 'Exponential RSS = %0.4f' % find_RSS(y, exponential_model), color='y')

# logarithmic model
# plt.plot(x, logarithmic_model, 'c--', label='logarithmic')
# plt.text(17.5, 80, 'Logarithmic RSS = %0.4f' % find_RSS(y, logarithmic_model), color='c')
plt.legend()
plt.show()

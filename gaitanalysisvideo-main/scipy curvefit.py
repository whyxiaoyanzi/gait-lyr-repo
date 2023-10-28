import numpy as np
 
# curve-fit() function imported from scipy
from scipy.optimize import curve_fit
 
from matplotlib import pyplot as plt
 
# numpy.linspace with the given arguments
# produce an array of 40 numbers between 0
# and 10, both inclusive
x = [1,2,3,4,5]
 
 
# y is another array which stores 3.45 times
# the sine of (values in x) * 1.334. 
# The random.normal() draws random sample 
# from normal (Gaussian) distribution to make
# them scatter across the base line
y = [8,23,48,83,128]
 
# Test function with coefficients as parameters
def test(x, a, b, c):
    return a * x**2 + c
 
# curve_fit() function takes the test-function
# x-data and y-data as argument and returns 
# the coefficients a and b in param and
# the estimated covariance of param in param_cov
param, param_cov = curve_fit(test, x, y)

 
print("Sine function coefficients:")
print(param)
print("Covariance of coefficients:")
print(param_cov)
 
# ans stores the new y-data according to 
# the coefficients given by curve-fit() function
# ans = (param[0]*(np.sin(param[1]*x)))
 
'''Below 4 lines can be un-commented for plotting results 
using matplotlib as shown in the first example. '''
 
# plt.plot(x, y, 'o', color ='red', label ="data")
# plt.plot(x, ans, '--', color ='blue', label ="optimized data")
# plt.legend()
# plt.show()


coefficients = np.polyfit(x, y, 2)
poly_function = np.poly1d(coefficients)
print(poly_function)
x2 = [6,7,8]
y2 = poly_function(x2)
print(y2)
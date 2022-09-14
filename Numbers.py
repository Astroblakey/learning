import numpy as np
import matplotlib.pyplot as plt
import sys
# This code is for Blake to test out random variables and form a covariance matrix out of a randomized gaussian distribution
# Written by Blake, 9/14/2022

matrix = np.random.rand(100,100)
newmatrix = matrix
newmatrix[25:30,25:30] = 10
newmatrix[25:30,75:80] = 10
newmatrix[60:80,40:60] = 10
plt.matshow(newmatrix)
plt.xlabel("X-Axis ")
plt.ylabel("Y-Axis")
plt.show()
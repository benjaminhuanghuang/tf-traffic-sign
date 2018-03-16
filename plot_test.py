import numpy as np
from sklearn.linear_model import LinearRegression

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Pizza diameter
# X should be 2D array
X_train = np.array([6, 8, 10, 14, 18]).reshape(-1, 1)

# Pizza prices
y_train = [7, 9, 13, 17.5, 18]


plt.figure()
plt.title('Pizaa price plotted against diameter')
plt.xlabel('Diameter in inches')
plt.ylabel('Price in dollars')
plt.plot(X_train, y_train, 'ko')
plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.plot([0, 25], [1.97, 26.37], color='k', linestyle='-', linewidth=2)
plt.show()

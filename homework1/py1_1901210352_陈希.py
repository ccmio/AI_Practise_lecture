import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 画散点图
df_dot = pd.read_csv('./dot.csv', header=0)
x1 = df_dot['x1']
x2 = df_dot['x2']
y_c = df_dot['y_c']
Y_c = [['red' if y else 'blue'] for y in y_c]
plt.scatter(x1, x2, s=12, c=np.squeeze(Y_c))

# 画等高线图
df_prob = pd.read_csv('./probs.csv')
probs = df_prob.values
xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
plt.contour(xx, yy, probs, levels=[.5])

plt.show()

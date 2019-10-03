import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df_dot = pd.read_csv('./dot.csv', header=0)
x1 = df_dot['x1']
x2 = df_dot['x2']
y_c = df_dot['y_c']
Y_c = [['red' if y else 'blue'] for y in y_c]
plt.scatter(x1, x2, s=4, c=np.squeeze(Y_c))

df_prob = pd.read_csv('./probs.csv', header=0)
print(df_prob)
x = df_prob.index.values
y = x
plt.contour(y, x, df_prob, levels=0)
plt.show()

# plt.figure(figsize=(9, 5))
#
# plt.plot(absolute_error[:, 0], absolute_error[:, 1], label='绝对误差')
# plt.plot(relative_error[:, 0], relative_error[:, 1], label='相对误差')
#
# plt.title("拟合函数的误差", fontsize=14)
# plt.xlabel("数字2之间的距离", fontsize=14)
# plt.ylabel("误差", fontsize=14)
# plt.tick_params(axis='both', which='major', labelsize=14)
# plt.grid(ls='--')
# plt.legend()
# plt.show()
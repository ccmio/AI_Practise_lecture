import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('./SH_600519_high_low.csv', header=0)
high = df['high']
low = df['low']
x = np.arange(0, 47)

plt.xlabel('day')
plt.ylabel('price')
plt.title('Kweichow Moutai')
plt.plot(x, high, label='high')
plt.legend()
plt.show()

plt.xlabel('day')
plt.ylabel('price')
plt.title('Kweichow Moutai')
plt.plot(x, low, label='low', c='orange')
plt.legend()
plt.show()

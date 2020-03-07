import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd

x = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4095]
y = [62.89, 63.55, 61.57, 57.24, 56.76 , 58.74, 59.40, 59.89]

df = pd.DataFrame({'Random projection size':x, 'Target accuracy (%)':y})
sns.set()
a = sns.scatterplot(x='Random projection size', y='Target accuracy (%)', data=df)
plt.show()
	

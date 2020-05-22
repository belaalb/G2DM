import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


divergences = {'ERM_photo': np.asarray([[np.nan, 0.82875, 0.9425, 0.975625], [0.81375, np.nan, 0.936875, 0.980625], [0.94, 0.934375, np.nan, 0.975], [0.9775, 0.98625, 0.9775, np.nan]]),
		'photo': np.asarray([[np.nan, 0.725625, 0.934375, 0.995625], [0.7175, np.nan, 0.94625, 0.998125], [0.935, 0.945, np.nan, 0.990625], [0.995625, 0.998125, 0.9925, np.nan]]),
		'ERM_art': np.asarray([[np.nan, 0.831875, 0.9475, 0.975625], [0.830625, np.nan, 0.9425, 0.98375], [0.944375, 0.9475, np.nan, 0.97], [0.97125, 0.98375, 0.963125, np.nan]]),
		'art': np.asarray([[np.nan, 0.838125, 0.915, 0.961875], [0.83125, np.nan, 0.875625, 0.964375], [0.920625, 0.886875, np.nan, 0.924375], [0.96125, 0.955625, 0.916875, np.nan]]),
		'ERM_cartoon': np.asarray([[np.nan, 0.87375, 0.929375, 0.975], [0.860625, np.nan, 0.920625, 0.97125], [0.931875, 0.916875, np.nan, 0.951875], [0.976875, 0.97625, 0.94875, np.nan]]),
		'cartoon': np.asarray([[np.nan, 0.861875, 0.89125, 0.991875], [0.869375, np.nan, 0.838125, 0.983125], [0.884375, 0.8425, np.nan, 0.92875], [0.988125, 0.98625, 0.93375, np.nan]]), 
		'ERM_sketch': np.asarray([[np.nan, 0.87, 0.953125, 0.99625], [0.875625, np.nan, 0.938125, 0.995625], [0.944375, 0.940625, np.nan, 0.9875], [0.995625, 0.996875, 0.986875, np.nan]]),
		'sketch': np.asarray([[np.nan, 0.774375, 0.87125, 0.9925], [0.775, np.nan, 0.861875, 0.993125], [0.876875, 0.8525, np.nan, 0.95125 ], [0.99375, 0.99375, 0.954375, np.nan]])
		}		

cmap = sns.cubehelix_palette(as_cmap=True, light=.9)
cmap.set_under(".5")

for key in divergences.keys():

	matrix = divergences[key]
	matrix = 2*(1.-2.*(1-matrix))		
	mask = np.zeros_like(matrix, dtype=np.bool)
	mask[np.triu_indices_from(mask)] = True

	for x in range(matrix.shape[0]):
		for y in range (matrix.shape[0]):
			a=matrix[y,x]
			b=matrix[x,y]
			matrix[x,y] = max(a,b)
			matrix[y,x] = max(a,b)		

	matrix = pd.DataFrame(matrix)
	print(matrix.isnull())

	domains=['P', 'A', 'C', 'S']
	fmt = lambda x, pos: '{:.2f}'.format(x)
	ax = sns.heatmap(matrix, annot=True, fmt='.2f', cmap=cmap, cbar_kws={'format': FuncFormatter(fmt)}, cbar=False, xticklabels=domains, yticklabels=domains, annot_kws={"size": 25}, 
vmin=0.9, vmax=2, mask=matrix.isnull())
	ax.xaxis.tick_top() # x axis on top
	ax.xaxis.set_label_position('top')
	ax.tick_params(length=0)
	cax = plt.gcf().axes[-1]
	cax.tick_params(labelsize=20)
	plt.yticks(rotation=0) 
	plt.savefig(key+'_disparity_hmap_pink.png')
	plt.show()

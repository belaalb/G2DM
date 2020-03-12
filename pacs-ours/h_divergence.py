import numpy as np
import os
import glob
import torch
import torch.utils.data
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
from data_loader import *
import sys
import models as models
import torchvision.models as models_torch
import argparse
from matplotlib.ticker import FuncFormatter

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

class classifier(nn.Module):

	def __init__(self):
		super(classifier, self).__init__()
		self.classifier = nn.Sequential()
		self.classifier.add_module('t1_fc1', nn.Linear(4096, 2))
		self.classifier.add_module('sigmoid', nn.Sigmoid())

		self.initialize_params()

	def forward(self, input_data):
		output = self.classifier(input_data)
		return output

	def initialize_params(self):
		for layer in self.modules():
			if isinstance(layer, torch.nn.Linear):
				init.kaiming_uniform_(layer.weight)


parser = argparse.ArgumentParser(description='PACS')
parser.add_argument('--data-path', type=str, default='./data/pacs/prepared_data/', metavar='Path', help='Data path')
parser.add_argument('--source1', type=str, default='photo', metavar='Path', help='Path to source1 file')
parser.add_argument('--source2', type=str, default='art_painting', metavar='Path', help='Path to source2 file')
parser.add_argument('--source3', type=str, default='cartoon', metavar='Path', help='Path to source3 file')
parser.add_argument('--target', type=str, default='sketch', metavar='Path', help='Path to target data')
parser.add_argument('--encoder-path', type=str, default=None, metavar='Path', help='Path for encoder')
parser.add_argument('--batch-size', type=int, default=500, metavar='N', help='number of data points per domain')
parser.add_argument('--workers', type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--architecture', choices=['erm', 'adversarial'], default='erm', help='DG type')
args = parser.parse_args()

img_transform = transforms.Compose([transforms.Resize((225, 225)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

domains = [args.source1, args.source2, args.source3, args.target]

domain1_path = args.data_path + 'test_' + args.source1 + '.hdf'
domain2_path = args.data_path + 'test_' + args.source2 + '.hdf'
domain3_path = args.data_path + 'test_' + args.source3 + '.hdf'
domain4_path = args.data_path + 'test_' + args.target + '.hdf'
n_classes = 7

domain1_dataset = Loader_validation(hdf_path=domain1_path, transform=img_transform)
domain1_loader = torch.utils.data.DataLoader(dataset=domain1_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

domain2_dataset = Loader_validation(hdf_path=domain2_path, transform=img_transform)
domain2_loader = torch.utils.data.DataLoader(dataset=domain2_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

domain3_dataset = Loader_validation(hdf_path=domain3_path, transform=img_transform)
domain3_loader = torch.utils.data.DataLoader(dataset=domain3_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

domain4_dataset = Loader_validation(hdf_path=domain4_path, transform=img_transform)
domain4_loader = torch.utils.data.DataLoader(dataset=domain4_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)


feature_extractor = models.AlexNet(baseline=False)
ckpt = torch.load(args.encoder_path)
if args.dg_type=='erm':
	not_loaded = feature_extractor.load_state_dict(ckpt['model_state'], strict=False)
else:
	not_loaded = feature_extractor.load_state_dict(ckpt['feature_extractor_state'], strict=False)

data1, labels1, _ = next(iter(domain1_loader))
domain1 = torch.zeros_like(labels1)
data2, labels2, _ = next(iter(domain2_loader))
domain2 = torch.ones_like(labels2)
data3, labels3, _ = next(iter(domain3_loader))
domain3 = 2*torch.ones_like(labels3)
data4, labels4, _ = next(iter(domain4_loader))
domain4 = 3*torch.ones_like(labels4)

h_div_matrix = np.zeros([len(domains), len(domains)])
all_data = [data1, data2, data3, data4]
labels = [domain1, domain2, domain3, domain4]

for d1, name1 in enumerate(domains):
	for d2, name2 in enumerate(domains):
		if d1 != d2:
			print('Domain 1', name1)
			print('Domain 2', name2)

			data_d1 = all_data[d1]
			label_d1 = labels[d1].numpy()
			data_d2 = all_data[d2]
			label_d2 = labels[d2].numpy() 

			features_d1 = feature_extractor(data_d1).detach().view(data_d1.size(0), -1).numpy()
			features_d2 = feature_extractor(data_d2).detach().view(data_d2.size(0), -1).numpy()
			x = np.vstack((features_d1, features_d2))
			y = np.hstack((label_d1, label_d2))
			x, y = shuffle(x, y)
			
			model = RandomForestClassifier(n_estimators=100)	
			acc = cross_val_score(model, x, y.ravel(), cv=5, scoring='accuracy')

			print('Accuracy:', acc)
			h_div_matrix[d1, d2] = np.mean(acc)	
			
print(h_div_matrix)		

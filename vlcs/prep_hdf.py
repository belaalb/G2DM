import numpy as np
import scipy.io as sio
import h5py
import os
import sys
import glob
import torch
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import argparse
from sklearn.model_selection import train_test_split


def prep_hdf(dataset, out_path):
	print(dataset)
	
	data_file = './data/' + dataset + '.mat'
	data_dict = sio.loadmat(data_file)
				
	data = data_dict['data'][:, :-1]
	data = (data - np.mean(data, axis = 0)) / np.std(data, axis = 0)
	print(data.shape)
	labels = data_dict['data'][:, -1]	
	
	labels = labels - 1
	
	print(np.min(labels))
	print(np.max(labels))
	
	X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.30, random_state = 1)
	
	if dataset == 'Caltech101':
		domain_train = np.zeros(y_train.shape)
		domain_test = np.zeros(y_test.shape)
	if dataset == 'LabelMe':
		domain_train = np.ones(y_train.shape)
		domain_test = np.ones(y_test.shape)
	if dataset == 'SUN09':
		domain_train = 2*np.ones(y_train.shape)
		domain_test = 2*np.ones(y_test.shape)
	if dataset == 'VOC2007':
		domain_train = 3*np.ones(y_train.shape)
		domain_test = 3*np.ones(y_test.shape)
	
	hdf_train = h5py.File(out_path + 'train_' + dataset + '.hdf', 'w')
	hdf_train.create_dataset('X', data=X_train)
	hdf_train.create_dataset('y_task', data=y_train)
	hdf_train.create_dataset('y_domain', data=domain_train)
	hdf_train.close()
	
	hdf_test = h5py.File(out_path + 'test_' + dataset + '.hdf', 'w')
	hdf_test.create_dataset('X', data=X_test)
	hdf_test.create_dataset('y_task', data=y_test)
	hdf_test.create_dataset('y_domain', data=domain_test)
	hdf_test.close()		
	
if __name__== '__main__':

	all_datasets = ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']
	hdf_path = './prepared_data/'
	
	for dset in all_datasets:
		prep_hdf(dset, hdf_path)


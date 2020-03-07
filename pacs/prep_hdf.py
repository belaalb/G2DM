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


def prep_hdf(dataset, out_path, train_test = 'train'):
	print(dataset)		
	data_path = './prepared_data/' + dataset + '_' + train_test + '.hdf5'
	data_hdf = h5py.File(data_path, 'r')
	
	hf = h5py.File(out_path + train_test + '_' + dataset + '.hdf', 'w')

	images = data_hdf['images']
	labels = data_hdf['labels'][:].squeeze()
	labels = labels - 1
	
	if dataset == 'photo':
		domain = np.zeros(labels.shape)
	if dataset == 'artpainting':
		domain = np.ones(labels.shape)
	if dataset == 'cartoon':
		domain = 2*np.ones(labels.shape)
	if dataset == 'sketch':
		domain = 3*np.ones(labels.shape)

	hf.create_dataset('X', data=images)
	hf.create_dataset('y_task', data=labels)
	hf.create_dataset('y_domain', data=domain)
	hf.close()	

def merge_hdf(list_of_paths, final_path, train_test, combination):
	images_all = []
	y_all = []
	domain_all = []
	
	print(list_of_paths)
	
	for data_file in list_of_paths:
		print(data_file)
		hf = h5py.File(data_file, 'r')
		images_all.append(hf['X'][:,:,:,:])
		y_all.append(hf['y_task'][:])
		domain_all.append(hf['y_domain'][:])
		hf.close()
	
	images = np.vstack(images_all)
	labels = np.concatenate(y_all, axis=0)
	domain = np.hstack(domain_all)
		
	all_names = '_'.join(combination) 
	hf = h5py.File(final_path + train_test + '_' + all_names + '.hdf', 'w')	
	hf.create_dataset('X', data=images)
	hf.create_dataset('y_task', data=labels)
	hf.create_dataset('y_domain', data=domain)	
	
if __name__== '__main__':

	parser = argparse.ArgumentParser(description='Preparing datasets')
	parser.add_argument('--train-val-test', type=str, default='train', help='Preparing train or test data')
	args = parser.parse_args()

	all_datasets = ['photo', 'artpainting', 'cartoon', 'sketch']
	hdf_path = './prepared_data/'
	
	train_val_test = args.train_val_test
	
	for dset in all_datasets:
		prep_hdf(dset, hdf_path, train_val_test)

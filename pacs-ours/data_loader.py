import torch.utils.data as data
from PIL import Image
import os
import scipy.io as sio
import h5py
import torch
import numpy as np

from torchvision import transforms

import torchvision


class Loader_source(data.Dataset):
	def __init__(self, hdf_path, transform=None):
		self.hdf_path = hdf_path
		hf = h5py.File(self.hdf_path, 'r')
		y = hf['y_task']
		self.length = len(y)
		
		self.transform = transform

		hf.close()

	def __getitem__(self, idx):

		hf = h5py.File(self.hdf_path, 'r')
		y_task = hf['y_task'][idx]
		y_domain = hf['y_domain'][idx]
		data = Image.fromarray(hf['X'][idx, :, :, :].astype('uint8'), 'RGB')
		hf.close()

		if self.transform is not None:
			data = self.transform(data)
				
		return data, torch.tensor(y_task, dtype=torch.long).squeeze(), torch.tensor(y_domain, dtype=torch.long).squeeze()

	def __len__(self):
		return self.length
		

class Loader_validation(data.Dataset):
	def __init__(self, hdf_path, transform=None):
		self.hdf_path = hdf_path
		hf = h5py.File(self.hdf_path, 'r')
		y = hf['y_task']
		self.length = len(y)
		self.transform = transform

		hf.close()

	def __getitem__(self, idx):

		hf = h5py.File(self.hdf_path, 'r')
		y_task = hf['y_task'][idx]
		y_domain = hf['y_domain'][idx]
		data = Image.fromarray(hf['X'][idx, :, :, :].astype('uint8'), 'RGB')
		hf.close()

		if self.transform is not None:
			data = self.transform(data)
		else:
			data = torch.from_numpy(np.transpose(hf['X'][idx, :, :, :], (2, 0, 1)))
				
		return data, torch.tensor(y_task, dtype=torch.long).squeeze(), torch.tensor(y_domain, dtype=torch.long).squeeze()

	def __len__(self):
		return self.length

class Loader_unif_sampling(data.Dataset):
	def __init__(self, hdf_path1, hdf_path2, hdf_path3, transform=None):
		self.hdf_path_1 = hdf_path1
		self.hdf_path_2 = hdf_path2
		self.hdf_path_3 = hdf_path3
		
		hdf_1 = h5py.File(self.hdf_path_1, 'r')
		hdf_2 = h5py.File(self.hdf_path_2, 'r')
		hdf_3 = h5py.File(self.hdf_path_3, 'r')
		
		self.len_1 = len(hdf_1['y_task'])
		self.len_2 = len(hdf_2['y_task'])
		self.len_3 = len(hdf_3['y_task'])
		
		self.length = np.max([self.len_1, self.len_2, self.len_3])
		
		self.transform = transform

		hdf_1.close()
		hdf_2.close()
		hdf_3.close()

	def __getitem__(self, idx):

		idx_1 = idx % self.len_1
		idx_2 = idx % self.len_2
		idx_3 = idx % self.len_3

		hdf_1 = h5py.File(self.hdf_path_1, 'r')
		hdf_2 = h5py.File(self.hdf_path_2, 'r')
		hdf_3 = h5py.File(self.hdf_path_3, 'r')
		
		y_task_1 = hdf_1['y_task'][idx_1]
		#y_domain_1 = hdf_1['y_domain'][idx_1]
		y_domain_1 = 0.0
		data_1_pil = Image.fromarray(hdf_1['X'][idx_1, :, :, :].astype('uint8'), 'RGB')
		hdf_1.close()
		
		y_task_2 = hdf_2['y_task'][idx_2]
		#y_domain_2 = hdf_2['y_domain'][idx_2]
		y_domain_2 = 1.0
		data_2_pil = Image.fromarray(hdf_2['X'][idx_2, :, :, :].astype('uint8'), 'RGB')
		hdf_2.close()
		
		y_task_3 = hdf_3['y_task'][idx_3]
		y_domain_3 = 2.0
		data_3_pil = Image.fromarray(hdf_3['X'][idx_3, :, :, :].astype('uint8'), 'RGB')
		hdf_3.close()

		if self.transform is not None:
			data_1 = self.transform(data_1_pil)
			data_2 = self.transform(data_2_pil)
			data_3 = self.transform(data_3_pil)
		else:
			data_1 = torch.from_numpy(np.transpose(hdf_1['X'][idx_1, :, :, :], (2, 0, 1)))
			data_2 = torch.from_numpy(np.transpose(hdf_2['X'][idx_2, :, :, :], (2, 0, 1)))
			data_3 = torch.from_numpy(np.transpose(hdf_3['X'][idx_3, :, :, :], (2, 0, 1)))
						
		#y_task = np.vstack((y_task_1, y_task_2, y_task_3))
		#y_domain = np.vstack((y_domain_1, y_domain_2, y_domain_3))	
				
		return data_1, data_2, data_3, torch.tensor(y_task_1).long().squeeze(), torch.tensor(y_task_2).long().squeeze(), torch.tensor(y_task_3).long().squeeze(), torch.tensor(y_domain_1).long().squeeze(), torch.tensor(y_domain_2).long().squeeze(), torch.tensor(y_domain_3).long().squeeze()

	def __len__(self):
		return self.length
		
class Loader_unif_sampling_test_visualization(data.Dataset):
	def __init__(self, hdf_path1, hdf_path2, hdf_path3, transform=None):
		self.hdf_path_1 = hdf_path1
		self.hdf_path_2 = hdf_path2
		self.hdf_path_3 = hdf_path3
		
		hdf_1 = h5py.File(self.hdf_path_1, 'r')
		hdf_2 = h5py.File(self.hdf_path_2, 'r')
		hdf_3 = h5py.File(self.hdf_path_3, 'r')
		
		self.len_1 = len(hdf_1['y_task'])
		self.len_2 = len(hdf_2['y_task'])
		self.len_3 = len(hdf_3['y_task'])
		
		self.length = np.max([self.len_1, self.len_2, self.len_3])
		
		self.transform = transform

		hdf_1.close()
		hdf_2.close()
		hdf_3.close()

	def __getitem__(self, idx):

		idx_1 = idx % self.len_1
		idx_2 = idx % self.len_2
		idx_3 = idx % self.len_3

		hdf_1 = h5py.File(self.hdf_path_1, 'r')
		hdf_2 = h5py.File(self.hdf_path_2, 'r')
		hdf_3 = h5py.File(self.hdf_path_3, 'r')
		
		y_task_1 = hdf_1['y_task'][idx_1]
		y_domain_1 = hdf_1['y_domain'][idx_1]
		data_1_pil = Image.fromarray(hdf_1['X'][idx_1, :, :, :].astype('uint8'), 'RGB')
		data_1_tensor = torch.from_numpy(np.transpose(hdf_1['X'][idx_1, :, :, :], (2, 0, 1)))
		hdf_1.close()
		
		y_task_2 = hdf_2['y_task'][idx_2]
		y_domain_2 = hdf_2['y_domain'][idx_2]
		data_2_pil = Image.fromarray(hdf_2['X'][idx_2, :, :, :].astype('uint8'), 'RGB')
		data_2_tensor = torch.from_numpy(np.transpose(hdf_2['X'][idx_2, :, :, :], (2, 0, 1)))
		hdf_2.close()
		
		y_task_3 = hdf_3['y_task'][idx_3]
		y_domain_3 = hdf_3['y_domain'][idx_3]
		data_3_pil = Image.fromarray(hdf_3['X'][idx_3, :, :, :].astype('uint8'), 'RGB')
		data_3_tensor = torch.from_numpy(np.transpose(hdf_3['X'][idx_3, :, :, :], (2, 0, 1)))
		
		from torch.utils.tensorboard import SummaryWriter

		data_tensor = torch.cat((data_1_tensor.unsqueeze(0), data_2_tensor.unsqueeze(0), data_3_tensor.unsqueeze(0)), dim=0)
		writer = SummaryWriter()
		grid = torchvision.utils.make_grid(data_tensor)
		writer.add_image('images', grid, 0)
		writer.close()
		

		if self.transform is not None:
			data_1_transf = self.transform(data_1_pil)
			data_2_transf = self.transform(data_2_pil)
			data_3_transf = self.transform(data_3_pil)
					
		data = torch.cat((data_1_transf.unsqueeze(0), data_2_transf.unsqueeze(0), data_3_transf.unsqueeze(0)), dim=0)	
		writer = SummaryWriter()
		grid = torchvision.utils.make_grid(data)
		writer.add_image('images', grid, 0)
		writer.close()
		
		y_task = np.vstack((y_task_1, y_task_2, y_task_3))
		y_domain = np.vstack((y_domain_1, y_domain_2, y_domain_3))	
				
		return data, torch.tensor(y_task, dtype=torch.long).squeeze(), torch.tensor(y_domain, dtype=torch.long).squeeze()

	def __len__(self):
		return self.length
		
if __name__ == '__main__':

	source_1 = './prepared_data/train_photo.hdf'
	source_2 = './prepared_data/train_art_painting.hdf'
	source_3 = './prepared_data/train_sketch.hdf'
	
	img_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

	source_dataset = Loader_unif_sampling_test_visualization(hdf_path1=source_1, hdf_path2=source_2, hdf_path3=source_3, transform=img_transform)
	source_loader = torch.utils.data.DataLoader(dataset=source_dataset, batch_size=32, shuffle=True, num_workers=0)

	a, b, c = source_dataset.__getitem__(100)
			
	print(a.size(), a.min(), a.max())						

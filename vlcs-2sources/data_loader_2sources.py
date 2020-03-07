import torch.utils.data as data
from PIL import Image
import os
import scipy.io as sio
import h5py
import torch
import numpy as np
from torchvision import datasets
from torchvision import transforms
import torchvision
		
class Loader_validation(data.Dataset):
	def __init__(self, path1, transform=None):
		self.path = path1
		self.dataset = datasets.ImageFolder(path1, transform=transform)
		self.length = self.dataset.__len__()
		self.transform = transform
		
	def __getitem__(self, idx):
		data, y_task = self.dataset.__getitem__(idx)
		y_domain = 3.
				
		return data, torch.tensor(y_task).long().squeeze(), torch.tensor(y_domain).long().squeeze()

	def __len__(self):
		return self.length

class Loader_unif_sampling(data.Dataset):
	def __init__(self, path1, path2, transform=None):
		self.path_1 = path1
		self.path_2 = path2
		
		self.dataset_1 = datasets.ImageFolder(self.path_1, transform=transform)
		self.dataset_2 = datasets.ImageFolder(self.path_2, transform=transform)
		
		self.len_1 = self.dataset_1.__len__()
		self.len_2 = self.dataset_2.__len__()
		
		self.length = np.max([self.len_1, self.len_2])
		
		self.transform = transform

	def __getitem__(self, idx):

		idx_1 = idx % self.len_1
		idx_2 = idx % self.len_2

		data_1, y_task_1 = self.dataset_1.__getitem__(idx_1)
		y_domain_1 = 0.
		
		data_2, y_task_2 = self.dataset_2.__getitem__(idx_2)
		y_domain_2 = 1.
		
				
		return data_1, data_2, torch.tensor(y_task_1).long().squeeze(), torch.tensor(y_task_2).long().squeeze(), torch.tensor(y_domain_1).long().squeeze(), torch.tensor(y_domain_2).long().squeeze()

	def __len__(self):
		return self.length
		
				
if __name__ == '__main__':

	source_1 = './vlcs/CALTECH/train/'
	source_2 = './vlcs/LABELME/train/'
	source_3 = './vlcs/SUN/train/'
				
	img_transform = transforms.Compose([transforms.RandomResizedCrop(225, scale=(0.7, 1.0)), transforms.RandomHorizontalFlip(), transforms.ToTensor()])#, transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

	source_dataset = Loader_unif_sampling(path1=source_1, path2=source_2, path3=source_3, transform=img_transform)
	source_loader = torch.utils.data.DataLoader(dataset=source_dataset, batch_size=32, shuffle=True, num_workers=0)
	
	a, b, c, a_task, b_task, c_task, a_domain, b_domain, c_domain = source_dataset.__getitem__(500)

	from torch.utils.tensorboard import SummaryWriter			
	data_tensor = torch.cat((a.unsqueeze(0), b.unsqueeze(0), c.unsqueeze(0)), dim=0)
	writer = SummaryWriter()
	grid = torchvision.utils.make_grid(data_tensor)
	writer.add_image('images', grid, 0)
	writer.close()		
			
	print(a.size(), a_task, a_domain)
	print(b.size(), b_task, b_domain)
	print(c.size(), c_task, c_domain)		

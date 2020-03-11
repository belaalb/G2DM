import os
import torch
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
from data_loader import Loader_validation
import models as models

class OffTest(object):

	def __init__(self, checkpoint_path, cp_name, loader, cuda):

		self.cp_task = os.path.join(checkpoint_path, 'task'+cp_name+'.pt') if cp_name else os.path.join(checkpoint_path, 'task_checkpoint_{}ep.pt')
		self.cp_domain = os.path.join(checkpoint_path, 'Domain_{}'+cp_name+'.pt') if cp_name else os.path.join(checkpoint_path, 'Domain_{}.pt')
		self.dataloader = loader
		self.cuda = cuda
		self.feature_extractor = models.AlexNet(num_classes = 7, baseline = False)
		self.task_classifier = models.task_classifier()

	def load_checkpoint(self, epoch=None):
		self.cp_task = self.cp_task.format(epoch) if epoch else self.cp_task
		
		if os.path.isfile(self.cp_task):
			ckpt = torch.load(self.cp_task)
			not_loaded = self.feature_extractor.load_state_dict(ckpt['feature_extractor_state'])
			not_loaded = self.task_classifier.load_state_dict(ckpt['task_classifier_state'])
		else:
			print('No checkpoint found at: {}'.format(self.cp_task))


	def test(self):

		feature_extractor = self.feature_extractor.eval()
		task_classifier = self.task_classifier.eval()
		
		with torch.no_grad():

			if self.cuda:
				self.feature_extractor = self.feature_extractor.cuda()
				self.task_classifier = self.task_classifier.cuda()

			target_iter = tqdm(enumerate(self.dataloader))

			n_total = 0
			n_correct = 0
			predictions_domain = []
			labels_domain = []
			for t, batch in target_iter:
				x, y, _ = batch
				if self.cuda:
					x = x.cuda()
					y = y.cuda()
				
				features = self.feature_extractor.forward(x)
				task_out = self.task_classifier.forward(features)
				class_output = F.softmax(task_out, dim=1)
				pred_task = class_output.data.max(1, keepdim=True)[1]
				n_correct += pred_task.eq(y.data.view_as(pred_task)).cpu().sum()
				n_total += x.size(0)
				
			acc = n_correct.item() * 1.0 / n_total			
			
			return acc

if __name__ == '__main__':
	dataset = 'sketch'
	data_path =  './prepared_data/test_'+ dataset+'.hdf'
	img_transform = transforms.Compose([transforms.Resize((225, 225)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
	dataset = Loader_validation(hdf_path=data_path, transform=img_transform)
	loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=4)
	cp_path = './'
	cp_name = '_checkpoint_9ep'
	cuda = True
	test_object = OffTest(cp_path, cp_name, loader, cuda)
	test_object.load_checkpoint()
	acc = test_object.test()
	print(acc)

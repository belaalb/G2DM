import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch.optim as optim
import torch.nn.init as init
import torch
from collections import OrderedDict


class feature_extractor(nn.Module):

	def __init__(self, baseline = True):
		super(feature_extractor, self).__init__()
		self.feature_extractor = nn.Sequential()
		self.feature_extractor.add_module('f_fc1', nn.Linear(4096, 1024))
		self.feature_extractor.add_module('f_relu1', nn.ReLU())
		self.feature_extractor.add_module('f_fc2', nn.Linear(1024, 128))
		self.feature_extractor.add_module('f_relu2', nn.ReLU())
		
		if baseline:
			self.feature_extractor.add_module('t_fc1', nn.Linear(128, 5))	
		
		self.initialize_params()

	def forward(self, input_data):
		features = self.feature_extractor(input_data)

		return features

	def initialize_params(self):

		for layer in self.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
			elif isinstance(layer, torch.nn.Linear):
				init.kaiming_uniform_(layer.weight)
			elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()

		
class task_classifier(nn.Module):

	def __init__(self):
		super(task_classifier, self).__init__()
		self.task_classifier1 = nn.Sequential()
		self.task_classifier1.add_module('t1_fc1', nn.Linear(4096, 7))
		#self.task_classifier1.add_module('t1_relu1', nn.ReLU())
		#self.task_classifier1.add_module('t1_drop1', nn.Dropout())
		#self.task_classifier1.add_module('t1_fc2', nn.Linear(1024, 7))

		#self.task_classifier2 = nn.Sequential()
		#self.task_classifier2.add_module('t2_fc1', nn.Linear(2304, 1024))
		#self.task_classifier2.add_module('t2_relu2', nn.ReLU())
		#self.task_classifier2.add_module('t2_fc2', nn.Linear(1024, 7))

		self.initialize_params()

	def forward(self, input_data):
		task_output = self.task_classifier1(input_data).view(input_data.size(0), -1)
		#task_output = self.task_classifier2(task_output)

		return task_output

	def initialize_params(self):

		for layer in self.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
			elif isinstance(layer, torch.nn.Linear):
				init.kaiming_uniform_(layer.weight)
			elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()		
		
class domain_discriminator(nn.Module):

	def __init__(self, rp_size, optimizer, lr, momentum, weight_decay):
		super(domain_discriminator, self).__init__()

		self.domain_discriminator = nn.Sequential()
		self.domain_discriminator.add_module('d_fc1', nn.Linear(rp_size, 1024))
		self.domain_discriminator.add_module('d_relu1', nn.ReLU())
		self.domain_discriminator.add_module('d_drop1', nn.Dropout())
		self.domain_discriminator.add_module('d_fc2', nn.Linear(1024, 1))
		#self.domain_discriminator.add_module('d_relu2', nn.ReLU())
		#self.domain_discriminator.add_module('d_drop2', nn.Dropout())
		#self.domain_discriminator.add_module('d_fc3', nn.Linear(1024, 1))

		self.optimizer = optimizer(list(self.domain_discriminator.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay)

		self.initialize_params()

		# TODO Check the RP size
		self.projection = nn.Linear(4096, rp_size, bias=False)
		with torch.no_grad():
			self.projection.weight.div_(torch.norm(self.projection.weight, keepdim=True))

	def forward(self, input_data):
		#reverse_feature = ReverseLayer.apply(input_data, alpha)	# Make sure there will be no problem when updating discs params
		feature = input_data.view(input_data.size(0), -1)
		feature_proj = self.projection(feature)
		domain_output = self.domain_discriminator(feature_proj)

		return domain_output

	def initialize_params(self):

		for layer in self.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
			elif isinstance(layer, torch.nn.Linear):
				init.kaiming_uniform_(layer.weight)
			elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()
				
class domain_discriminator_ablation_RP(nn.Module):

	def __init__(self, optimizer, lr, momentum, weight_decay):
		super(domain_discriminator_ablation_RP, self).__init__()

		self.domain_discriminator = nn.Sequential()
		self.domain_discriminator.add_module('d_fc1', nn.Linear(4096, 1024))
		self.domain_discriminator.add_module('d_relu1', nn.ReLU())
		self.domain_discriminator.add_module('d_fc2', nn.Linear(1024, 1024))
		self.domain_discriminator.add_module('d_relu2', nn.ReLU())
		self.domain_discriminator.add_module('d_fc3', nn.Linear(1024, 1))

		self.optimizer = optimizer(list(self.domain_discriminator.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay)

		self.initialize_params()

	def forward(self, input_data):
		feature = input_data.view(input_data.size(0), -1)
		domain_output = self.domain_discriminator(feature)

		return domain_output

	def initialize_params(self):

		for layer in self.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
			elif isinstance(layer, torch.nn.Linear):
				init.kaiming_uniform_(layer.weight)
			elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()				
				
class AlexNet(nn.Module):
	def __init__(self, baseline = True, num_classes = 7):
		super(AlexNet, self).__init__()
		
		self.num_classes = num_classes
		self.baseline = baseline
		'''
		self.features = nn.Sequential(
			nn.Conv2d(3, 96, kernel_size=11, stride=4),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
			nn.LocalResponseNorm(5, 1e-4, 0.75),
			nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
			nn.LocalResponseNorm(5, 1e-4, 0.75),
			nn.Conv2d(256, 384, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
		)

		self.classifier = nn.Sequential(
			#nn.Dropout(),
			nn.Linear(256 * 6 * 6, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),			
			nn.Dropout(),
			nn.Linear(4096, self.num_classes),
		)

		#self.classifier_added = nn.Sequential(
			#nn.ReLU(),
		#	nn.Dropout(),
		#	nn.Linear(4096, self.num_classes),
		#)
		'''
		self.features = nn.Sequential(OrderedDict([
			("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
			("relu1", nn.ReLU(inplace=True)),
			("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
			("norm1", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
			("conv2", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
			("relu2", nn.ReLU(inplace=True)),
			("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
			("norm2", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
			("conv3", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
			("relu3", nn.ReLU(inplace=True)),
			("conv4", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
			("relu4", nn.ReLU(inplace=True)),
			("conv5", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
			("relu5", nn.ReLU(inplace=True)),
			("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
		]))
		
		if self.baseline:
			self.classifier = nn.Sequential(OrderedDict([
				("fc6", nn.Linear(256 * 6 * 6, 4096)),
				("relu6", nn.ReLU(inplace=True)),
				("drop6", nn.Dropout()),
				("fc7", nn.Linear(4096, 4096)),
				("relu7", nn.ReLU(inplace=True)),
				("drop7", nn.Dropout()),
				("fc8", nn.Linear(4096, self.num_classes))]))
				
		else:
			self.classifier = nn.Sequential(OrderedDict([
				("fc6", nn.Linear(256 * 6 * 6, 4096)),
				("relu6", nn.ReLU(inplace=True)),
				("drop6", nn.Dropout()),
				("fc7", nn.Linear(4096, 4096)),
				("relu7", nn.ReLU(inplace=True)),
				("drop7", nn.Dropout())]))
				
		self.initialize_params()
		
	def initialize_params(self):

		for layer in self.modules():
			#if isinstance(layer, torch.nn.Conv2d):
				#init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
				#layer.bias.data.zero_()	
			if isinstance(layer, torch.nn.Linear):
				init.xavier_uniform_(layer.weight, 0.1)
				layer.bias.data.zero_()	
			#elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
				#layer.weight.data.fill_(1)
				#layer.bias.data.zero_()	

	def forward(self, x):
		x = self.features(x*57.6)
		x = x.view(x.size(0), 256 * 6 * 6)
		x = self.classifier(x)
		return x
				
				
if __name__=='__main__':


	feature_extractor = feature_extractor(baseline=False)
	task_classifier = task_classifier()

	domain_discriminator_list = []
	for i in range(3):
        	disc = domain_discriminator(optim.SGD, 0.1, 0.9, 0.001).train()
	        domain_discriminator_list.append(disc)

	batch = torch.rand(3,4096)

	z = feature_extractor(batch)
	task_y = task_classifier(z)

	domain_y = []

	for disc in domain_discriminator_list:
		domain_y.append(disc(z))

	print(z.size())				

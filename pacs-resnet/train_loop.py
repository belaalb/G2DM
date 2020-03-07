import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from test import test
from utils import LabelSmoothingLoss, GradualWarmupScheduler


class TrainLoop(object):

	def __init__(self, models_dict, optimizer_task, source_loader, test_source_loader, target_loader, nadir_slack, alpha, patience, factor, label_smoothing, warmup_its, lr_threshold, verbose=-1, cp_name=None, save_cp=False, checkpoint_path=None, checkpoint_epoch=None, cuda=True, logging=False, ablation='no', train_mode='hv'):
		if checkpoint_path is None:
			# Save to current directory
			self.checkpoint_path = os.getcwd()
		else:
			self.checkpoint_path = checkpoint_path
			if not os.path.isdir(self.checkpoint_path):
				os.mkdir(self.checkpoint_path)

		self.save_epoch_fmt_task = os.path.join(self.checkpoint_path, 'task'+cp_name) if cp_name else os.path.join(self.checkpoint_path, 'task_checkpoint_{}ep.pt')
		self.save_epoch_fmt_domain = os.path.join(self.checkpoint_path, 'Domain_{}'+cp_name) if cp_name else os.path.join(self.checkpoint_path, 'Domain_{}.pt')
		
		self.cuda_mode = cuda
		self.feature_extractor = models_dict['feature_extractor']
		self.task_classifier = models_dict['task_classifier']
		self.domain_discriminator_list = models_dict['domain_discriminator_list']
		self.optimizer_task = optimizer_task
		self.source_loader = source_loader
		self.test_source_loader = test_source_loader
		self.target_loader = target_loader
		self.history = {'loss_task': [], 'hypervolume': [], 'loss_domain': [], 'accuracy_source':[], 'accuracy_target':[]}
		self.cur_epoch = 0
		self.total_iter = 0
		self.nadir_slack = nadir_slack
		self.alpha = alpha
		self.ablation = ablation
		self.train_mode = train_mode
		self.device = next(self.feature_extractor.parameters()).device

		its_per_epoch = len(source_loader.dataset)//(source_loader.batch_size) + 1 if len(source_loader.dataset)%(source_loader.batch_size)>0 else len(source_loader.dataset)//(source_loader.batch_size)
		patience = patience * (1+its_per_epoch)
		self.after_scheduler_task = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_task, factor=factor, patience=patience, verbose=True if verbose>0 else False, threshold=lr_threshold, min_lr=1e-7)
		self.after_scheduler_disc_list = [torch.optim.lr_scheduler.ReduceLROnPlateau(disc.optimizer, factor=factor, patience=patience, verbose=True if verbose>0 else False, threshold=lr_threshold, min_lr=1e-7) for disc in self.domain_discriminator_list]
		self.verbose = verbose
		self.save_cp = save_cp
	
		self.scheduler_task = GradualWarmupScheduler(self.optimizer_task, total_epoch=warmup_its, after_scheduler=self.after_scheduler_task)
		self.scheduler_disc_list = [GradualWarmupScheduler(self.domain_discriminator_list[i].optimizer, total_epoch=warmup_its, after_scheduler=sch_disc) for i, sch_disc in enumerate(self.after_scheduler_disc_list)]
		
		if checkpoint_epoch is not None:
			self.load_checkpoint(checkpoint_epoch)
		
		self.logging = logging
		if self.logging:	
			from torch.utils.tensorboard import SummaryWriter
			self.writer = SummaryWriter()
			
		if label_smoothing > 0.0:
			self.ce_criterion = LabelSmoothingLoss(label_smoothing, lbl_set_size = 7)
		else:
			self.ce_criterion = torch.nn.CrossEntropyLoss()	#torch.nn.NLLLoss()#

		#loss_domain_discriminator = F.binary_cross_entropy_with_logits(y_predict, curr_y_domain)
		weight = torch.tensor([2.0/3.0, 1.0/3.0]).to(self.device)
			#d_cr=torch.nn.CrossEntropyLoss(weight=weight)
		self.d_cr=torch.nn.NLLLoss(weight=weight)
		#self.d_cr=  F.binary_cross_entropy_with_logits()
	#### Edit####
	def adjust_learning_rate(self,optimizer, epoch=1,every_n=700,In_lr=0.01):
		"""Sets the learning rate to the initial LR decayed by 10 every n epoch epochs"""
		every_n_epoch=every_n#n_epoch/n_step
		lr = In_lr * (0.1** (epoch // every_n_epoch))
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr
			
		return lr
	######
	def train(self, n_epochs=1, save_every=1):

		'''
		##### Edit ####
		# init necessary objects
		num_steps = n_epochs #* (len(train_loader.dataset) / train_loader.batch_size)
		#yd = Variable(torch.from_numpy(np.hstack([np.repeat(1, int(batch_size / 2)), np.repeat(0, int(batch_size / 2))]).reshape(50, 1)))
		j = 0
		lambd=0
		max_lambd= 1 - self.alpha
		
		max_lr_t = list(self.optimizer_task.param_groups)[-1]['initial_lr']
		
		max_lr_d = list(self.domain_discriminator_list[0].optimizer.param_groups)[-1]['initial_lr']
		#0.005
		#lr_2=0
		#lr=In_lr
		#Domain_Classifier.set_lambda(lambd)
		#print(num_steps)

		'''
		max_lr_t = list(self.optimizer_task.param_groups)[-1]['initial_lr']
		
		max_lr_d = list(self.domain_discriminator_list[0].optimizer.param_groups)[-1]['initial_lr']
		
		###############
		
		while self.cur_epoch < n_epochs:


			cur_loss_task = 0
			cur_hypervolume = 0
			cur_loss_total = 0
			
			source_iter = tqdm(enumerate(self.source_loader), disable=False)

			'''
			## Edit ##
			## update lambda and learning rate as suggested in the paper
			p = float(j) / num_steps
			lambd = max_lambd * round(2. / (1. + np.exp(-10. * p)) - 1, 3) #I changed 10 to 7
			#lambd=lambd
			
			lr_t = max_lr_t / (1. + 10 * p)**0.75 # 
			lr_d = max_lr_d / (1. + 10 * p)**0.75 #
			
			lr_t= self.adjust_learning_rate(self.optimizer_task,In_lr=lr_t)
			for _, disc in enumerate(self.domain_discriminator_list):
				lr_d= self.adjust_learning_rate(disc.optimizer,In_lr=lr_d)
			
			
			
			self.alpha= 1 - lambd
			#print('Alpha = {}' .format(self.alpha))
			j += 1
			'''
			lr_t = max_lr_t #/ (1. + 10 * p)**0.75 # 
			lr_d = max_lr_d #/ (1. + 10 * p)**0.75 #
			'''
			lr_t= self.adjust_learning_rate(self.optimizer_task, epoch= self.cur_epoch, every_n=0.8 * n_epochs, In_lr=lr_t)
			for _, disc in enumerate(self.domain_discriminator_list):
				lr_d= self.adjust_learning_rate(disc.optimizer, epoch= self.cur_epoch, every_n=0.8 * n_epochs, In_lr=lr_d)
			'''
			##########
				
			print('Epoch {}/{} | Alpha = {:1.3} | Lr_task = {:1.4} | Lr_dis = {:1.4} '.format(self.cur_epoch + 1, n_epochs, self.alpha, lr_t , lr_d))

			for t, batch in source_iter:
				if self.ablation == 'all':
					cur_losses = self.train_step_ablation_all(batch)
				else:
					cur_losses = self.train_step(batch)
					
				self.scheduler_task.step(epoch = self.total_iter, metrics = 1.- self.history['accuracy_source'][-1] if self.cur_epoch>0 else np.inf)
				for sched in self.scheduler_disc_list:
					sched.step(epoch = self.total_iter, metrics = 1.- self.history['accuracy_source'][-1] if self.cur_epoch>0 else np.inf)	 

				cur_loss_task += cur_losses[0]
				cur_hypervolume += cur_losses[1]
				cur_loss_total += cur_losses[2]
				self.total_iter += 1
				
				if self.logging:
					self.writer.add_scalar('train/task_loss', cur_losses[0], self.total_iter)
					self.writer.add_scalar('train/hypervolume_loss', cur_losses[1], self.total_iter)
					self.writer.add_scalar('train/total_loss', cur_losses[2], self.total_iter)

			self.history['loss_task'].append(cur_loss_task/(t+1))
			self.history['hypervolume'].append(cur_hypervolume/(t+1))

			print('Current task loss: {}.'.format(cur_loss_task/(t+1)))
			print('Current hypervolume: {}.'.format(cur_hypervolume/(t+1)))
			
			self.history['accuracy_source'].append(test(self.test_source_loader, self.feature_extractor, self.task_classifier, self.domain_discriminator_list, self.device, source_target = 'source', epoch = self.cur_epoch, tb_writer = self.writer if self.logging else None))
			self.history['accuracy_target'].append(test(self.target_loader, self.feature_extractor, self.task_classifier, self.domain_discriminator_list, self.device, source_target = 'target', epoch = self.cur_epoch, tb_writer = self.writer if self.logging else None))

			idx = np.argmax(self.history['accuracy_source'])

			print('Valid. on SOURCE data - Current acc., best acc., best acc target, and epoch: {:0.4f}, {:0.4f}, {:0.4f}, {}'.format(self.history['accuracy_source'][-1], np.max(self.history['accuracy_source']), self.history['accuracy_target'][idx], 1+np.argmax(self.history['accuracy_source'])))
			print('Valid. on TARGET data - Current acc., best acc., and epoch: {:0.4f}, {:0.4f}, {}'.format(self.history['accuracy_target'][-1], np.max(self.history['accuracy_target']), 1+np.argmax(self.history['accuracy_target'])))

			if self.logging:
				self.writer.add_scalar('misc/LR-task', self.optimizer_task.param_groups[0]['lr'], self.total_iter)
				for i, disc in enumerate(self.domain_discriminator_list):
					self.writer.add_scalar('misc/LR-disc{}'.format(i), disc.optimizer.param_groups[0]['lr'], self.total_iter)

			self.cur_epoch += 1

			if self.save_cp and (self.cur_epoch % save_every == 0 or self.history['accuracy_target'][-1] > np.max([-np.inf]+self.history['accuracy_target'][:-1])):
				self.checkpointing()
		
		if self.logging:
			self.writer.close()
		
		idx_final = np.argmax(self.history['accuracy_source'])	

		return np.max(self.history['accuracy_target']), self.history['accuracy_target'][-1] 
		
	def train_step(self, batch):
		self.feature_extractor.train()
		self.task_classifier.train()
		for disc in self.domain_discriminator_list:
			disc = disc.train()
		
		x_1, x_2, x_3, y_task_1, y_task_2, y_task_3, y_domain_1, y_domain_2, y_domain_3 = batch

		x = torch.cat((x_1, x_2, x_3), dim=0)
		y_task = torch.cat((y_task_1, y_task_2, y_task_3), dim=0)
		y_domain = torch.cat((y_domain_1, y_domain_2, y_domain_3), dim=0)
		
		if self.cuda_mode:
			x = x.to(self.device)
			y_task = y_task.to(self.device)

		# COMPUTING FEATURES		
		features = self.feature_extractor.forward(x)	
		features_ = features.detach()

		 
		# DOMAIN DISCRIMINATORS (First)
		for i, disc in enumerate(self.domain_discriminator_list):
			y_predict = disc.forward(features_).squeeze()
			
			curr_y_domain = torch.where(y_domain == i, torch.ones(y_domain.size(0)), torch.zeros(y_domain.size(0)))
			curr_y_domain.type_as(y_domain)
			#print(y_domain.shape,curr_y_domain.shape,y_predict.shape)
			#print(sum(curr_y_domain))
			curr_y_domain = curr_y_domain.long()
			if self.cuda_mode:
				curr_y_domain = curr_y_domain.long().to(self.device)

			#loss_domain_discriminator = F.binary_cross_entropy_with_logits(y_predict, curr_y_domain)
			#weight = torch.tensor([2.0/3.0, 1.0/3.0]).to(self.device)
			#d_cr=torch.nn.CrossEntropyLoss(weight=weight)
			#d_cr=torch.nn.NLLLoss(weight=weight)
			loss_domain_discriminator = self.d_cr (y_predict, curr_y_domain)
			#print(loss_domain_discriminator)
		
			if self.logging:
				self.writer.add_scalar('train/D{}_loss'.format(i), loss_domain_discriminator, self.total_iter)
			
			disc.optimizer.zero_grad()
			loss_domain_discriminator.backward()
			disc.optimizer.step()
		 
		# UPDATE TASK CLASSIFIER AND FEATURE EXTRACTOR
		task_out = self.task_classifier.forward(features)

		loss_domain_disc_list = []
		loss_domain_disc_list_float = []
		for i, disc in enumerate(self.domain_discriminator_list):
			y_predict = disc.forward(features).squeeze()
			curr_y_domain = torch.where(y_domain == i, torch.zeros(y_domain.size(0)), torch.ones(y_domain.size(0)))
			
			curr_y_domain = curr_y_domain.long()
			if self.cuda_mode:
				curr_y_domain = curr_y_domain.long().to(self.device)
			  
			#loss_domain_disc_list.append(F.binary_cross_entropy_with_logits(y_predict, curr_y_domain))
			#y_predict= y_predict.long().to(self.device)
			loss_domain_disc_list.append(self.d_cr(y_predict, curr_y_domain))

			loss_domain_disc_list_float.append(loss_domain_disc_list[-1].detach().item())
		
		if self.train_mode == 'hv':
			self.update_nadir_point(loss_domain_disc_list_float)
			
		hypervolume = 0
		for loss in loss_domain_disc_list:			
			if self.train_mode == 'hv':
				hypervolume -= torch.log(self.nadir - loss + 1e-6)
				#hypervolume -= torch.log(loss)
			
			elif self.train_mode == 'avg':
				hypervolume -= loss

		task_loss = self.ce_criterion(task_out, y_task)
		loss_total = self.alpha*task_loss + (1-self.alpha)*hypervolume/len(loss_domain_disc_list)
		#loss_total = task_loss + self.alpha *hypervolume/len(loss_domain_disc_list)

		self.optimizer_task.zero_grad()
		loss_total.backward()
		self.optimizer_task.step()
		
		'''
		# DOMAIN DISCRIMINATORS (Second-After)
		for i, disc in enumerate(self.domain_discriminator_list):
			y_predict = disc.forward(features_).squeeze()
			
			curr_y_domain = torch.where(y_domain == i, torch.ones(y_domain.size(0)), torch.zeros(y_domain.size(0)))
			curr_y_domain.type_as(y_domain)
			#print(y_domain.shape,curr_y_domain.shape,y_predict.shape)
			#print(sum(curr_y_domain))
			if self.cuda_mode:
				curr_y_domain = curr_y_domain.long().to(self.device)

			
			loss_domain_discriminator = self.d_cr (y_predict, curr_y_domain)
			#print(loss_domain_discriminator)
		
			if self.logging:
				self.writer.add_scalar('train/D{}_loss'.format(i), loss_domain_discriminator, self.total_iter)
			
			disc.optimizer.zero_grad()
			loss_domain_discriminator.backward()
			disc.optimizer.step()

			####
		
		'''
		losses_return = task_loss.item(), hypervolume.item(), loss_total.item()
		return losses_return

	def train_step_ablation_all(self, batch):

		self.feature_extractor.train()
		self.task_classifier.train()
		for disc in self.domain_discriminator_list:
			disc = disc.train()
		
		x_1, x_2, x_3, y_task_1, y_task_2, y_task_3, _, _, _ = batch

		x = torch.cat((x_1, x_2, x_3), dim=0)
		y_task = torch.cat((y_task_1, y_task_2, y_task_3), dim=0)
		
		if self.cuda_mode:
			x = x.to(self.device)
			y_task = y_task.to(self.device)

		# COMPUTING FEATURES		
		features = self.feature_extractor.forward(x)	
		task_out = self.task_classifier.forward(features)
		task_loss = torch.nn.CrossEntropyLoss()(task_out, y_task)
		
		self.optimizer_task.zero_grad()
		task_loss.backward()
		self.optimizer_task.step()
		
		losses_return = task_loss.item(), 0

		return losses_return
		
	def checkpointing(self):
		if self.verbose>0:
			print(' ')	
			print('Checkpointing...')
			
		ckpt = {'feature_extractor_state': self.feature_extractor.state_dict(),
				'task_classifier_state': self.task_classifier.state_dict(),
				'optimizer_task_state': self.optimizer_task.state_dict(),
				'scheduler_task_state': self.scheduler_task.state_dict(),
				'history': self.history,
				'cur_epoch': self.cur_epoch}
		torch.save(ckpt, self.save_epoch_fmt_task.format(self.cur_epoch))

		for i, disc in enumerate(self.domain_discriminator_list):
			ckpt = {'model_state': disc.state_dict(),
					'optimizer_disc_state': disc.optimizer.state_dict(),
					'scheduler_disc_state': self.scheduler_disc_list[i].state_dict()}
			torch.save(ckpt, self.save_epoch_fmt_domain.format(i + 1))

	def load_checkpoint(self, epoch):
		ckpt = self.save_epoch_fmt_task.format(epoch)

		if os.path.isfile(ckpt):

			ckpt = torch.load(ckpt)
			# Load model state
			self.feature_extractor.load_state_dict(ckpt['feature_extractor_state'])
			self.task_classifier.load_state_dict(ckpt['task_classifier_state'])
			self.domain_classifier.load_state_dict(ckpt['domain_classifier_state'])
			# Load optimizer state
			self.optimizer.load_state_dict(ckpt['optimizer_task_state'])
			# Load scheduler state
			self.scheduler_task.load_state_dict(ckpt['scheduler_task_state'])
			# Load history
			self.history = ckpt['history']
			self.cur_epoch = ckpt['cur_epoch']

			for i, disc in enumerate(self.domain_discriminator_list):
				ckpt = torch.load(self.save_epoch_fmt_domain.format(i + 1))
				disc.load_state_dict(ckpt['model_state'])
				disc.optimizer.load_state_dict(ckpt['optimizer_disc_state'])
				self.scheduler_disc_list[i].load_state_dict(ckpt['scheduler_disc_state'])

		else:
			print('No checkpoint found at: {}'.format(ckpt))

	def print_grad_norms(self, model):
		norm = 0.0
		for params in list(filter(lambda p: p.grad is not None, model.parameters())):
			norm += params.grad.norm(2).item()
		print('Sum of grads norms: {}'.format(norm))

	def update_nadir_point(self, losses_list):
		self.nadir = float(np.max(losses_list) * self.nadir_slack + 1e-8)

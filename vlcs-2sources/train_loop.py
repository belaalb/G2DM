import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from test import test
import torchvision


class TrainLoop(object):

	def __init__(self, model, optimizer, source_loader, test_source_loader, target_loader, patience, l2, penalty_weight, penalty_anneal_epochs, checkpoint_path=None, checkpoint_epoch=None, cuda=True, logging=False):
		if checkpoint_path is None:
			# Save to current directory
			self.checkpoint_path = os.getcwd()
		else:
			self.checkpoint_path = checkpoint_path
			if not os.path.isdir(self.checkpoint_path):
				os.mkdir(self.checkpoint_path)

		self.save_epoch_fmt = os.path.join(self.checkpoint_path, 'IRM_{}ep.pt')

		self.cuda_mode = cuda
		self.model = model
		self.device = next(self.model.parameters()).device
		self.optimizer = optimizer
		self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=patience)
		self.source_loader = source_loader
		self.test_source_loader = test_source_loader
		self.target_loader = target_loader
		self.history = {'loss': [], 'accuracy_source':[], 'accuracy_target':[]}
		self.cur_epoch = 0
		self.dummy = torch.tensor(1.).to(self.device).requires_grad_()
		self.l2 = l2
		self.penalty_weight = penalty_weight
		self.penalty_anneal_epochs = penalty_anneal_epochs
		self.total_iter = 0 

		if checkpoint_epoch is not None:
			self.load_checkpoint(checkpoint_epoch)
			
		self.logging = logging
		if self.logging:	
			from torch.utils.tensorboard import SummaryWriter
			self.writer = SummaryWriter()	

	def train(self, n_epochs=1, save_every=1):

		while self.cur_epoch < n_epochs:

			print('Epoch {}/{}'.format(self.cur_epoch + 1, n_epochs))

			cur_loss = 0
			source_iter = tqdm(enumerate(self.source_loader))

			for t, batch in source_iter:
				
				loss_it = self.train_step(batch)
				self.total_iter += 1

				cur_loss += loss_it 

			self.history['loss'].append(cur_loss/(t+1))

			print('Current loss: {}.'.format(cur_loss/(t+1)))
			print('Current LR: {}'.format(self.optimizer.state_dict()['param_groups'][0]['lr']))
			
			if self.logging:
				self.writer.add_scalar('train/task_loss', cur_loss_task, self.total_iter)
				self.writer.add_scalar('train/hypervolume_loss', cur_hypervolume, self.total_iter)
				self.writer.add_scalar('misc/LR', self.optimizer_task.param_groups[0]['lr'], self.total_iter)

			self.history['accuracy_source'].append(test(self.test_source_loader, self.model, self.device, source_target = 'source', epoch = self.cur_epoch, tb_writer = self.writer if self.logging else None))
			self.history['accuracy_target'].append(test(self.target_loader, self.model, self.device, source_target = 'target', epoch = self.cur_epoch, tb_writer = self.writer if self.logging else None))

			print('Valid. on SOURCE data - Current acc., best acc., and epoch: {:0.4f}, {:0.4f}, {}'.format(self.history['accuracy_source'][-1], np.max(self.history['accuracy_source']), 1+np.argmax(self.history['accuracy_source'])))
			print('Valid. on TARGET data - Current acc., best acc., and epoch: {:0.4f}, {:0.4f}, {}'.format(self.history['accuracy_target'][-1], np.max(self.history['accuracy_target']), 1+np.argmax(self.history['accuracy_target'])))

			if self.cur_epoch % save_every == 0 or self.history['accuracy_target'][-1] > np.max([-np.inf]+self.history['accuracy_target'][:-1]):
				self.checkpointing()

			self.cur_epoch += 1
			self.scheduler.step()
			
		# saving final models
		print('Saving final model...')
		self.checkpointing()

		return 1. - np.max(self.history['accuracy_target'])

	def train_step(self, batch):
		self.model.train()
		loss_acc = 0
		penalty = 0
		for domain in range(3):
			x = batch[domain].to(self.device)
			y_task = batch[domain+3].to(self.device)

			out = self.model(x)
			loss_current = torch.nn.CrossEntropyLoss()(out*self.dummy, y_task)
			penalty += self.penalty(loss_current, self.dummy)
			loss_acc += loss_current
		
		weight_norm = torch.tensor(0.).to(self.device)
		
		for w in self.model.parameters():
			weight_norm += w.norm().pow(2)

		loss = loss_acc / 3
		#penalty = penalty / 3
		loss += self.l2 * weight_norm
		penalty_weight = (self.penalty_weight if self.cur_epoch >= self.penalty_anneal_epochs else 1.0)
		loss += penalty_weight * penalty
		
		if penalty_weight > 1.0:
		# Rescale the entire loss to keep gradients in a reasonable range
			loss /= penalty_weight
				
		self.optimizer.zero_grad()				
		loss.backward()
		self.optimizer.step()
			
		return loss.item()

	def checkpointing(self):
		# Checkpointing
		print('Checkpointing...')
		ckpt = {'model_state': self.model.state_dict(),
				'history': self.history,
				'cur_epoch': self.cur_epoch,
				'optimizer_state': self.optimizer.state_dict(),
				'scheduler_state': self.scheduler.state_dict()}
		torch.save(ckpt, self.save_epoch_fmt.format(self.cur_epoch))

	def load_checkpoint(self, epoch):
		ckpt = self.save_epoch_fmt_task.format(epoch)

		if os.path.isfile(ckpt):

			ckpt = torch.load(ckpt)
			# Load model state
			self.model.load_state_dict(ckpt['model_state'])
			# Load optimizer state
			self.optimizer.load_state_dict(ckpt['optimizer_state'])
			# Load scheduler state
			self.scheduler.load_state_dict(ckpt['scheduler_state'])
			# Load history
			self.history = ckpt['history']
			self.cur_epoch = ckpt['cur_epoch']

		else:
			print('No checkpoint found at: {}'.format(ckpt))

	def print_grad_norms(self, model):
		norm = 0.0
		for params in list(filter(lambda p: p.grad is not None, model.parameters())):
			norm += params.grad.norm(2).item()
		print('Sum of grads norms: {}'.format(norm))

	def penalty(self, loss, dummy):
		grad = torch.autograd.grad(loss, [dummy], create_graph=True)[0]
		return torch.sum(grad**2)



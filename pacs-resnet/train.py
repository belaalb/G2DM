import argparse
import os
import sys
import random
from tqdm import tqdm

import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
import PIL
import pandas

import models as models
from train_loop import TrainLoop
from data_loader import Loader_source, Loader_validation, Loader_unif_sampling
import torchvision.models as models_tv
import utils

parser = argparse.ArgumentParser(description='RP for domain generalization')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 50)')
parser.add_argument('--lr-task', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.0002)')
parser.add_argument('--lr-domain', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.0002)')
parser.add_argument('--lr-threshold', type=float, default=1e-4, metavar='LRthrs', help='learning rate (default: 1e-4)')
parser.add_argument('--momentum-task', type=float, default=0.9, metavar='m', help='momentum (default: 0.9)')
parser.add_argument('--momentum-domain', type=float, default=0.9, metavar='m', help='momentum (default: 0.9)')
parser.add_argument('--l2', type=float, default=1e-5, metavar='L2', help='Weight decay coefficient (default: 0.00001')
parser.add_argument('--factor', type=float, default=0.1, metavar='f', help='LR decrease factor (default: 0.1')
parser.add_argument('--checkpoint-epoch', type=int, default=None, metavar='N', help='epoch to load for checkpointing. If None, training starts from scratch')
parser.add_argument('--checkpoint-path', type=str, default='./', metavar='Path', help='Path for checkpointing')
parser.add_argument('--data-path', type=str, default= None, metavar='Path', help='Data path')
parser.add_argument('--source1', type=str, default='photo', metavar='Path', help='Path to source1 file')
parser.add_argument('--source2', type=str, default='cartoon', metavar='Path', help='Path to source2 file')
parser.add_argument('--source3', type=str, default='sketch', metavar='Path', help='Path to source3 file')
parser.add_argument('--target', type=str, default='art_painting', metavar='Path', help='Path to target data')
parser.add_argument('--seed', type=int, default=None, metavar='S', help='random seed (default: None)')
parser.add_argument('--nadir-slack', type=float, default=1.5, metavar='nadir', help='factor for nadir-point update. Only used in hyper mode (default: 1.5)')
parser.add_argument('--alpha', type=float, default=0.8, metavar='alpha', help='balance losses to train encoder. Should be within [0,1]')
parser.add_argument('--rp-size', type=int, default=3000, metavar='rp', help='Random projection size. Should be smaller than 4096')
parser.add_argument('--patience', type=int, default=20, metavar='N', help='number of epochs to wait before reducing lr (default: 20)')
parser.add_argument('--smoothing', type=float, default=0.2, metavar='l', help='Label smoothing (default: 0.2)')
parser.add_argument('--warmup-its', type=float, default=500, metavar='w', help='LR warm-up iterations (default: 500)')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--save-every', type=int, default=5, metavar='N', help='how many epochs to wait before logging training status. Default is 5')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
parser.add_argument('--no-logging', action='store_true', default=False, help='Deactivates logging')
parser.add_argument('--ablation', choices = ['all', 'RP', 'no'], default='no', help='Ablation study (removing only RPs (option: RP), RPs+domain classifier (option: all), (default: no))')
parser.add_argument('--train-mode', choices = ['hv', 'avg'], default='hv', help='Train mode (options: hv, avg), (default: hv))')
parser.add_argument('--train-model', choices = ['alexnet', 'resnet18'], default='alexnet', help='Train model (options: alexnet, resnet18), (default: alexnet))')

parser.add_argument('--n-runs', type=int, default=1, metavar='n', help='Number of repetitions (default: 3)')

args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False
args.logging = True if not args.no_logging else False

assert args.alpha>=0. and args.alpha<=1.

print('Source domains: {}, {}, {}'.format(args.source1, args.source2, args.source3))
print('Target domain:', args.target)
print('Cuda Mode: {}'.format(args.cuda))
print('Batch size: {}'.format(args.batch_size))
print('LR task: {}'.format(args.lr_task))
print('LR domain: {}'.format(args.lr_domain))
print('L2: {}'.format(args.l2))
print('Alpha: {}'.format(args.alpha))
print('Momentum task: {}'.format(args.momentum_task))
print('Momentum domain: {}'.format(args.momentum_domain))
print('Nadir slack: {}'.format(args.nadir_slack))
print('RP size: {}'.format(args.rp_size))
print('Patience: {}'.format(args.patience))
print('Smoothing: {}'.format(args.smoothing))
print('Warmup its: {}'.format(args.warmup_its))
print('LR factor: {}'.format(args.factor))
print('Ablation: {}'.format(args.ablation))
print('Train mode: {}'.format(args.train_mode))
print('Train model: {}'.format(args.train_model))
print('Seed: {}'.format(args.seed))


acc_runs = []
acc_blind = []
seeds = [1, 10, 100]

for run in range(args.n_runs):
	print('Run {}'.format(run))

	# Setting seed
	if args.seed is None:
		random.seed(seeds[run])
		torch.manual_seed(seeds[run])
		if args.cuda:
			torch.cuda.manual_seed(seeds[run])
		checkpoint_path = os.path.join(args.checkpoint_path, args.target+'_seed'+str(seeds[run]))
	else:
		seeds[run]=args.seed
		random.seed(args.seed)
		torch.manual_seed(args.seed)
		if args.cuda:
			torch.cuda.manual_seed(args.seed)
		checkpoint_path = os.path.join(args.checkpoint_path, args.target+'_seed'+str(args.seed))
	

	

	img_transform_train = transforms.Compose([transforms.RandomGrayscale(p=0.10),transforms.RandomResizedCrop(222, scale=(0.8,1.0)), transforms.RandomHorizontalFlip(),transforms.ColorJitter(brightness= 0.4, contrast= 0.4, saturation= 0.4, hue=min(0.5, 0.4)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
	img_transform_test = transforms.Compose([transforms.Resize(size=222), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#,transforms.ColorJitter(),transforms.CenterCrop(size=225),
#,transforms.CenterCrop(size=225)

	if args.data_path is None:  
		args.data_path = os.path.join('/',os.path.join(* os.getcwd().split('/')[0:-1]), 'data', 'pacs', 'prepared_data/')
		print(args.data_path)

	train_source_1 = args.data_path + 'train_' + args.source1 + '.hdf'
	train_source_2 = args.data_path + 'train_' + args.source2 + '.hdf'
	train_source_3 = args.data_path + 'train_' + args.source3 + '.hdf'
	test_source_1 = args.data_path + 'val_' + args.source1 + '.hdf'
	test_source_2 = args.data_path + 'val_' + args.source2 + '.hdf'
	test_source_3 = args.data_path + 'val_' + args.source3 + '.hdf'
	target_path = args.data_path + 'test_' + args.target + '.hdf'

	source_dataset = Loader_unif_sampling(hdf_path1=train_source_1, hdf_path2=train_source_2, hdf_path3=train_source_3, transform=img_transform_train)
	source_loader = torch.utils.data.DataLoader(dataset=source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

	test_source_dataset = Loader_unif_sampling(hdf_path1=test_source_1, hdf_path2=test_source_2, hdf_path3=test_source_3, transform=img_transform_test)
	test_source_loader = torch.utils.data.DataLoader(dataset=test_source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

	target_dataset = Loader_validation(hdf_path=target_path, transform=img_transform_test)
	target_loader = torch.utils.data.DataLoader(dataset=target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
		
	task_classifier = models.task_classifier()
	domain_discriminator_list = []
	for i in range(3):
		if args.rp_size == 4096:
			disc = models.domain_discriminator_ablation_RP(optim.SGD, args.lr_domain, args.momentum_domain, args.l2).train()
		else:
			disc = models.domain_discriminator(args.rp_size, optim.SGD, args.lr_domain, args.momentum_domain, args.l2).train()
		domain_discriminator_list.append(disc)	
		
	#feature_extractor = models.AlexNet(num_classes = 7, baseline = False)
	feature_extractor = models.get_pretrained_model(args.train_model)
	#state_dict = torch.load("./alexnet_caffe.pth.tar")
	#del state_dict["classifier.fc8.weight"]
	#del state_dict["classifier.fc8.bias"]
	#not_loaded = feature_extractor.load_state_dict(state_dict, strict = False)

	optimizer_task = optim.SGD(list(feature_extractor.parameters())+list(task_classifier.parameters()), lr=args.lr_task, momentum=args.momentum_task, weight_decay = args.l2)

	models_dict = {}
	models_dict['feature_extractor'] = feature_extractor
	models_dict['task_classifier'] = task_classifier
	models_dict['domain_discriminator_list'] = domain_discriminator_list

	if args.cuda:
		for key in models_dict.keys():
			if key != 'domain_discriminator_list':
				models_dict[key] = models_dict[key].cuda()
			else:
				for k, disc in enumerate(models_dict[key]):
					models_dict[key][k] = disc.cuda()
		torch.backends.cudnn.benchmark = True
			
	trainer = TrainLoop(models_dict, optimizer_task, source_loader, test_source_loader, target_loader, args.nadir_slack, args.alpha, args.patience, args.factor, args.smoothing, args.warmup_its, args.lr_threshold, checkpoint_path=checkpoint_path, checkpoint_epoch=args.checkpoint_epoch, cuda=args.cuda, ablation=args.ablation, logging=args.logging, train_mode=args.train_mode)
	err, err_blind = trainer.train(n_epochs=args.epochs, save_every=args.save_every)

	acc_runs.append(1-err)
	acc_blind.append(err_blind)

df = pandas.DataFrame(data={'Acc-{}'.format(args.target): acc_runs, 'Seed': seeds[:args.n_runs]})
df.to_csv('./accuracy_runs_'+args.target+'.csv', sep=',',mode='a', index = False)

df = pandas.DataFrame(data={'Acc-{}'.format(args.target): acc_runs, 'Seed': seeds[:args.n_runs]})
df.to_csv('./accuracy_runs_'+args.target+'_blind.csv', sep=',',mode='a', index = False)

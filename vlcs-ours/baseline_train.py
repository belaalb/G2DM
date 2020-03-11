import argparse
import os
import sys
import random

import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
import PIL
import pandas	

import models as models
from baseline_train_loop import TrainLoop
from data_loader import Loader_validation, Loader_unif_sampling
import utils


parser = argparse.ArgumentParser(description='VLCS baseline')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR', help='learning rate (default: 0.0002)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='m', help='momentum (default: 0.9)')
parser.add_argument('--l2', type=float, default=0.00005, metavar='m', help='L2 weight decay (default: 0.00005)')
parser.add_argument('--checkpoint-epoch', type=int, default=None, metavar='N', help='epoch to load for checkpointing. If None, training starts from scratch')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--data-path', type=str, default= None, metavar='Path', help='Data path')
parser.add_argument('--source1', type=str, default='CALTECH', metavar='Path', help='Path to source1 file')
parser.add_argument('--source2', type=str, default='LABELME', metavar='Path', help='Path to source2 file')
parser.add_argument('--source3', type=str, default='SUN', metavar='Path', help='Path to source3 file')
parser.add_argument('--target', type=str, default='PASCAL', metavar='Path', help='Path to target data')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--patience', type=int, default=400, metavar='N', help='number of epochs to wait before reducing lr (default: 20)')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--save-every', type=int, default=5, metavar='N', help='how many epochs to wait before logging training status. Default is 5')
parser.add_argument('--n-runs', type=int, default=1, metavar='n', help='Number of repetitions (default: 3)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')

args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

acc_runs = []
seeds = [1, 10, 100]

print('Source domains: {}, {}, {}'.format(args.source1, args.source2, args.source3))
print('Target domain:', args.target)
print('Cuda Mode: {}'.format(args.cuda))
print('Batch size: {}'.format(args.batch_size))
print('LR: {}'.format(args.lr))
print('L2: {}'.format(args.l2))
print('Momentum: {}'.format(args.momentum))
print('Patience: {}'.format(args.patience))

for run in range(args.n_runs):
	print('Run {}'.format(run))

	# Setting seed
	random.seed(seeds[run])
	torch.manual_seed(seeds[run])

	if args.cuda:
		torch.cuda.manual_seed(seeds[run])

	img_transform_train = transforms.Compose([transforms.RandomResizedCrop(225, scale=(0.7,1.0)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
	img_transform_test = transforms.Compose([transforms.Resize((225, 225)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

	if args.data_path is None: 
		args.data_path = os.path.join('/',os.path.join(* os.getcwd().split('/')[0:-1]), 'data', 'vlcs', 'prepared_data/')
		print(args.data_path)  

	train_source_1 = args.data_path + args.source1 + '/train/'
	train_source_2 = args.data_path + args.source2 + '/train/'
	train_source_3 = args.data_path + args.source3 + '/train/'
	test_source_1 = args.data_path + args.source1 + '/test/'
	test_source_2 = args.data_path + args.source2 + '/test/'
	test_source_3 = args.data_path + args.source3 + '/test/'
	target_path = args.data_path + args.target + '/test/'

	source_dataset = Loader_unif_sampling(path1=train_source_1, path2=train_source_2, path3=train_source_3, transform=img_transform_train)
	source_loader = torch.utils.data.DataLoader(dataset=source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

	test_source_dataset = Loader_unif_sampling(path1=test_source_1, path2=test_source_2, path3=test_source_3, transform=img_transform_test)
	test_source_loader = torch.utils.data.DataLoader(dataset=test_source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

	target_dataset = Loader_validation(path=target_path, transform=img_transform_test)
	target_loader = torch.utils.data.DataLoader(dataset=target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

	model = models.AlexNet(num_classes = 5, baseline = True)
	state_dict = torch.load("../alexnet_caffe.pth.tar")
	del state_dict["classifier.fc8.weight"]
	del state_dict["classifier.fc8.bias"]
	not_loaded = model.load_state_dict(state_dict, strict = False)

	optimizer = optim.SGD(list(model.features.parameters())+list(model.classifier.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.l2, nesterov=True)

	if args.cuda:
		model = model.cuda()

	torch.backends.cudnn.benchmark=True
		
	trainer = TrainLoop(model, optimizer, source_loader, test_source_loader, target_loader, args.patience, checkpoint_path=args.checkpoint_path, checkpoint_epoch=args.checkpoint_epoch, cuda=args.cuda)

	err = trainer.train(n_epochs=args.epochs, save_every=args.save_every)
	acc_runs.append(1-err)

print(acc_runs)
df = pandas.DataFrame(data={'Acc-{}'.format(args.target): acc_runs, 'Seed': seeds[:args.n_runs]})
df.to_csv('./baseline_accuracy_runs_'+args.target+'.csv', sep=',', index = False)



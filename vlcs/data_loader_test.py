from torchvision import datasets
from torchvision import transforms
import torch
import torchvision

def get_dataset(path, mode, image_size):
	if mode == "train":
		img_transform = transforms.Compose([
			transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # std=[1/256., 1/256., 1/256.] #[0.229, 0.224, 0.225]
		])
	else:
		img_transform = transforms.Compose([
			transforms.Resize(image_size),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # std=[1/256., 1/256., 1/256.]
		])
	return datasets.ImageFolder(path, transform=img_transform)
    
    
if __name__=='__main__':

	dataset = get_dataset('./vlcs/CALTECH/train/', 'train', 224)
	loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=4, shuffle=True)
	a1, b1 = dataset.__getitem__(500)
	print(a1)
	print(b1)
	a2, b2 = dataset.__getitem__(20)
	print(a2)
	print(b2)
	
	from torch.utils.tensorboard import SummaryWriter

	data_tensor = torch.cat((a1.unsqueeze(0), a2.unsqueeze(0)), dim=0)
	writer = SummaryWriter()
	grid = torchvision.utils.make_grid(data_tensor)
	writer.add_image('images', grid, 0)
	writer.close()
	

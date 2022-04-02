import os
import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset
import PIL

class VidhoiDataset(Dataset):
	def __init__(self, device='cuda', dirnum='0000/'):
		self.dir = '/home/senior/workspace/data/picture/' + dirnum
		names = sorted(os.listdir(self.dir))
		self.imglst = [f'{os.path.join(self.dir, name)}' for name in names]
		self.device = device

	def __len__(self):
		return len(self.imglst)

	def __getitem__(self, idx):
		pilimg = PIL.Image.open(self.imglst[idx])
		transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
		])
		img = transform(pilimg).to(self.device)
		print(f'path : {self.imglst[idx]}')
		print(f'shape : {img.shape}')
		return img

if __name__ == '__main__':
	data = VidhoiDataset()
	data[3]

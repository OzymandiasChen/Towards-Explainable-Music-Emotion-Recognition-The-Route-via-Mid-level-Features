
import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
with open(os.path.join(sys.path[0], 'config.json')) as json_file:
	config = json.load(json_file)

class JointVGG(nn.Module):
	def __init__(self):
		super().__init__()
		# (N, 1, 128, 431)
		self.conv0 = nn.Sequential(
			nn.Conv2d(1, 64, kernel_size = (5, 5), stride = (2, 2), padding = (2, 2)), nn.BatchNorm2d(64), nn.ReLU(), # (N, 64, 64, 216)
			nn.Conv2d(64, 64, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)), nn.BatchNorm2d(64), nn.ReLU(), # (N, 64, 64, 216)
			nn.MaxPool2d((2, 2)), nn.Dropout(p=0.3) #(N, 64, 32, 108)
			)
		self.conv1 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)), nn.BatchNorm2d(128), nn.ReLU(), # (N, 128, 32, 108)
			nn.Conv2d(128, 128, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)), nn.BatchNorm2d(128), nn.ReLU(), # (N, 128, 32, 108)
			nn.MaxPool2d((2, 2)), nn.Dropout(p=0.3) #(N, 128, 16, 54)
			)
		self.conv2 = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)), nn.BatchNorm2d(256), nn.ReLU(), # (N, 256, 16, 54)
			nn.Conv2d(256, 256, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)), nn.BatchNorm2d(256), nn.ReLU(), # (N, 256, 16, 54)
			nn.Conv2d(256, 384, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)), nn.BatchNorm2d(384), nn.ReLU(), # (N, 384, 16, 54)
			nn.Conv2d(384, 512, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)), nn.BatchNorm2d(512), nn.ReLU(), # (N, 512, 16, 54)
			nn.Conv2d(512, 256, kernel_size = (3, 3), stride = (1, 1), padding = (0, 0)), nn.BatchNorm2d(256), nn.ReLU(), # (N, 256, 14, 52)
			nn.AdaptiveAvgPool2d((1, 1)) # (N, 256, 1, 1)
			)
		self.linear2mid = nn.Sequential(
			nn.Linear(256, 7),
			nn.Sigmoid() #
			)
		self.linear_mid2emo = nn.Sequential(
			nn.Linear(7, 8),
			nn.Sigmoid() #
			)
		self.linear2emo = nn.Sequential(
			nn.Linear(256, 8),
			nn.Sigmoid() #
			)

	def forward(self, x):
		x = self.conv0(x)
		x = self.conv1(x)
		x = self.conv2(x)
		x = torch.flatten(x, start_dim = 1, end_dim = -1)
		mid = self.linear2mid(x)
		# emo = self.linear2emo(x)
		emo = self.linear_mid2emo(mid)
		return torch.cat((emo, mid), 1)

	def _init_weights(self, m) -> None:
		# As is, it does absolutely nothing. It is a type annotation for 
		# the main function that simply states that this function returns None.
		if(isinstance(m, nn.Conv2d)):
			nn.init.kaiming_uniform_(m.weight)
		elif(isinstance(m, nn.Linear)):
			nn.init.xavier_uniform_(m.weight)

if __name__ == '__main__':
	net = JointVGG()
	print(net)
	net = net.cuda()
	summary(net, (1, 128, 431))
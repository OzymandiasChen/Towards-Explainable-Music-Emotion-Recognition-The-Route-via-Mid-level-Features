
# coding: utf-8
import os
import json
import numpy as np
from models.JointVGG import JointVGG
import dataloader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter
from eval import Evaluator
from utils import LossLogger
with open("config.json") as json_file:
	config = json.load(json_file)


class Trainer():

	def __init__(self, expName):
		self.model = JointVGG() 
		self.optimizer = optim.Adam(self.model.parameters(), lr = 0.0005)
		self.criterion = nn.MSELoss(reduction = 'none')
		self.epochNum = config["EPOCH_NUM"]

		self.bestLoss = float("Inf")
		self.bestEpoch = -1
		self.expName = expName
		self.logPath = os.path.join(config["PROJECT_PATH"][config["ENV"]], 'logs', self.expName)
		if not os.path.exists(self.logPath):
			os.makedirs(self.logPath)
		self.writter = SummaryWriter(log_dir = os.path.join(self.logPath, 'tensorboard'))
		self.nonbetterCount = 0
		self.patience = config["EARLY_STOPPING_PATIENCE"]

		dataload = dataloader.MidEmoDataLoader()
		X_train, Y_train = dataload.datasetLoader('train')
		train_torch_dataset = Data.TensorDataset(X_train, Y_train)
		self.train_loader = Data.DataLoader(dataset = train_torch_dataset, batch_size = config["TRAIN_BATCH_SIZE"], shuffle = True)
		self.X_valid, self.Y_valid = dataload.datasetLoader('valid')
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.fo = open(os.path.join(self.logPath, 'trainLog.txt'), 'w+')
		self.tragetInfo = config["TRAGET_INFO"]

	def trainValidWritter(self, loss_train, loss_valid, loss_overall_train, loss_emo_train, loss_overall_valid, loss_emo_valid, epochIndex):
		for i in range(15):
			self.writter.add_scalars('{}/epoch'.format(self.tragetInfo[i]), {'train': loss_train[i].item(), 
																			'valid': loss_valid[i].item()}, epochIndex)
		self.writter.add_scalars('Loss_overall/epoch', {'train': loss_overall_train.item(), 
														'valid': loss_emo_train.item()}, epochIndex)
		self.writter.add_scalars('Loss_emo/epoch', {'train': loss_overall_valid.item(), 
														'valid': loss_emo_valid.item()}, epochIndex)
		self.writter.flush()

	def betterSaver(self, loss_valid):
		if(loss_valid <= self.bestLoss):
			self.bestLoss = loss_valid
			torch.save(self.model.cpu(), os.path.join(self.logPath, 'bestModel_loss.pkl'))
			print('[Model Saved!!]~~~~~~~~~~~~~~~~~~\n')
			self.fo.write('[Model Saved!!]~~~~~~~~~~~~~~~~~~\n')
			self.nonbetterCount = 0
		else:
			self.nonbetterCount = self.nonbetterCount + 1
		if(self.nonbetterCount == self.patience):
			print('[EARLY STOPPING!!]\n')
			self.fo.write('[EARLY STOPPING!!]\n')
			self.writter.close()
			self.fo.close()
			return True
		return False # continue flag

	def one_pass_train(self, epochIndex):
		epoch_loss = torch.zeros(15).cuda()
		self.model.train()
		self.model = self.model.to(self.device)
		for step, (batch_x, batch_y) in enumerate(self.train_loader):
			self.model.zero_grad()
			batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
			output_batch = self.model(batch_x)
			# 		X = torch.cat(X, axis=0)
			# torch.cat((x, x, x), 1)
			loss_element = self.criterion(output_batch, batch_y)
			loss_target = torch.mean(loss_element, 0) #
			loss_overall = torch.sum(loss_target)
			loss_overall.backward()
			self.optimizer.step()
			epoch_loss += torch.sum(loss_element, 0)
			if(step % (len(self.train_loader) // 4) == 0):
				loss_overall_batch, loss_emo_batch = LossLogger(self.fo, 'batch', loss_target, step, epochIndex)
				self.writter.add_scalar('Loss_overall/batch', loss_overall_batch.item(), epochIndex*len(self.train_loader)+step)
				self.writter.add_scalar('Loss_emo/batch', loss_emo_batch.item(), epochIndex*len(self.train_loader)+step)
		torch.save(self.model.cpu(), os.path.join(self.logPath, 'lastModel.pkl'))
		return epoch_loss / len(self.train_loader.dataset)

	def train(self):
		# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		valid_evaluator = Evaluator()
		self.model = self.model.to(self.device)
		for epochIndex in range(self.epochNum):
			loss_train = self.one_pass_train(epochIndex)
			loss_overall_train, loss_emo_train = LossLogger(self.fo, 'train', loss_train, '--', epochIndex)
			loss_valid = valid_evaluator.evaluation(self.X_valid, self.Y_valid, self.model)
			loss_overall_valid, loss_emo_valid = LossLogger(self.fo, 'valid', loss_valid, '--', epochIndex)
			self.trainValidWritter(loss_train, loss_valid, loss_overall_train, loss_emo_train, loss_overall_valid, loss_emo_valid, epochIndex)
			if(self.betterSaver(loss_emo_valid) == True):
				break
		self.writter.close()
		self.fo.close()

if __name__ == '__main__':
	trainer = Trainer("0104_mid2emo")
	trainer.train()
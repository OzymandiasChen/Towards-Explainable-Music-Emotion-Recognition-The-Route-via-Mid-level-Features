
# coding: utf-8
import os
import numpy as np
import dataloader
import json
from models.JointVGG import JointVGG
import torch
import torch.nn as nn
import torch.utils.data as Data
from utils import LossLogger
with open("config.json") as json_file:
	config = json.load(json_file)

class Evaluator():

	def __init__(self):
		self.criterion = nn.MSELoss(reduction = 'none')

	def evaluation(self, X, Y, model):
		torch_dataset = Data.TensorDataset(X, Y)
		dataLoader = Data.DataLoader(dataset = torch_dataset, batch_size = config["VALID_BATCH_SIZE"], shuffle = False)
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		model =model.to(device)
		model.eval()
		with torch.no_grad():
			output = []
			for _, (batch_x, batch_y) in enumerate(dataLoader):
				batch_x = batch_x.to(device)
				output_batch = model(batch_x)
				output.append(output_batch)

			output = torch.cat(output, axis = 0)
			output = output.cpu()
			# print(output.shape)
			loss_element = self.criterion(output, Y.float())
			# print(loss_element.shape)
			loss_target = torch.mean(loss_element, 0)
			# print(loss_target.shape)
		return loss_target


class Test():
	def __init__(self, expName):
		self.expName = expName
		self.logPath = os.path.join(config["PROJECT_PATH"][config["ENV"]], 'logs', self.expName)
		self.lastModel = torch.load(os.path.join(self.logPath, 'lastModel.pkl'))
		self.bestLossModel = torch.load(os.path.join(self.logPath, 'bestModel_loss.pkl'))
		self.fo_test = open(os.path.join(self.logPath, 'testLog.txt'), 'w+')

	def test(self):
		evaluator = Evaluator()
		dataload = dataloader.MidEmoDataLoader()
		X_test, Y_test = dataload.datasetLoader('test')
		loss_target_last = evaluator.evaluation(X_test, Y_test, self.lastModel)
		_, _ = LossLogger(self.fo_test, 'test_last', loss_target_last, '--', '--')
		loss_target_best = evaluator.evaluation(X_test, Y_test, self.bestLossModel)
		_, _ = LossLogger(self.fo_test, 'test_best', loss_target_best, '--', '--')

if __name__ == '__main__':
	testor = Test('0104_mid2emo')
	testor.test()

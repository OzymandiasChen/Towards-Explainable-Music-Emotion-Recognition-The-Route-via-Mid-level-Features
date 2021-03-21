
import torch
from torch.utils.tensorboard import SummaryWriter
import json
with open("config.json") as json_file:
	config = json.load(json_file)

def LossLogger(fo, phase, loss_target, batchIndex, epochIndex):
	'''
	phase: 'batch'/'train'/'valid'/test
	'''
	targetInfo = config["TRAGET_INFO"]
	loss_overall = torch.sum(loss_target)
	loss_emo = torch.sum(loss_target[:8])
	with torch.no_grad():
		# [[, , ,]
		fo.write('[{}] {}/{} loss_all:{}, loss_emo:{}\n'.format(phase, batchIndex, epochIndex, loss_overall.item(), loss_emo.item()))
		print('[{}] {}/{} loss_all:{}, loss_emo:{}\n'.format(phase, batchIndex, epochIndex, loss_overall.item(), loss_emo.item()), end = "")
		if(phase != 'batch'):
			for i in range(15):
				fo.write('{}:{:.3f},  '.format(targetInfo[i], loss_target[i]))
				print('{}:{:.3f},  '.format(targetInfo[i], loss_target[i]), end = "")
			fo.write('\n')
			print('\n')
	return loss_overall, loss_emo

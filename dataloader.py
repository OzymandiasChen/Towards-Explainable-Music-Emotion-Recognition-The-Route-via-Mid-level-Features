
import os
import sys
import json
import librosa
import numpy as np
from random import shuffle
import torch
import xlrd
import pandas as pd
with open("config.json") as json_file:
	config = json.load(json_file)

class MidEmoDataLoader():
	# info = ['valence', 'energy', 'tension', 'anger', 'fear', 'happy', 'sad', 'tender', 'TARGET',
	# 			'melody', 'articulation', 'rhythm_complexity', 'rhythm_stability', 'dissonance', 'atonality', 'mode']
	'''
		The class is structed with following paramaters and functions.
		Functions:
			(1). __init__(self): init function for midEmo datalaoder.
			(2). audio2Mel(self): load and save melspecto for audio.
			(3). TrainEvalTestSplit(self): split train valid test file.
			(4). loadDataSetFileName(self, train_test_valid):Getting dataset file name lsit for Train/Test/Valid.
			(5). loadMelMidFeaEmo_audio(self, audioFileName, EmoInfo, df_MidF_anno): load logSpectro, target for a certain song piece.
			(6). datasetLoader(self, train_test_valid): dataset loader for train/valid/test.
		Description:
			(a). (2) load logmelspectrogram, which will be calculated for a single time.
			(b). (3) and (4) are dataset file namelist setter and loader separately.
			(c). (6) calls (4) and (5) to load a certain dataset.
		Using:
			(a). If it is the first time to use, call 'MidEmoDataLoader().audio2Mel()' for loading and saving 10s mel.
			(b). If it is the first time to use, call 'MidEmoDataLoader().TrainEvalTestSplit()' for split data for train test valid.
			(c). call 'MidEmoDataLoader().TrainEvalTestSplit().dataloader.datasetLoader('')' to load a certain dataset.
	'''

	def __init__(self):
		'''
		init function for midEmo datalaoder.
		'''
		self.soundTrackPath = os.path.join(config["SOUNDTRACKS_PATH"][config["ENV"]], 'set1', 'set1')
		# home/chenxi/Documents/Reaserching_Work_(poorly)/MER/dataset/Soundtracks/set1/set1
		self.midLevelPath = config["MID_LEVEL_FEATURE_DATASET_PATH"][config["ENV"]]
		self.melFolderPath = os.path.join(self.soundTrackPath, config["AUDIO_PROCESSING_METHOD"])
		if not os.path.exists(self.melFolderPath):
			os.makedirs(self.melFolderPath)
		self.fileFolderPath = os.path.join(self.soundTrackPath, config["FILE_SPLIT_FOLDER"])
		if not os.path.exists(self.fileFolderPath):
			os.makedirs(self.fileFolderPath)
	
	def audio2Mel(self):
		'''
		load and save melspecto for audio.
		'''
		rawSoundAudioPath = os.path.join(self.soundTrackPath, 'mp3', 'Soundtrack360_mp3')
		# home/chenxi/Documents/Reaserching_Work_(poorly)/MER/dataset/Soundtracks/set1/set1/mp3/Soundtrack360_mp3
		for audioFileName in os.listdir(rawSoundAudioPath):
			if(audioFileName.split('.')[1] != 'mp3'):
				continue
			audioFilePath = os.path.join(rawSoundAudioPath, audioFileName)
			duration = librosa.get_duration(filename = audioFilePath)
			y, _ = librosa.load(audioFilePath, sr = config["SR"], offset = (duration-10.0)/2, duration = 10.0) # 10s info reserved
			melSpectro = librosa.feature.melspectrogram(y, sr = config["SR"])
			logMelSpectro = librosa.amplitude_to_db(melSpectro, amin=1e-07)
			torch.save(logMelSpectro, os.path.join(self.melFolderPath, audioFileName.split('.')[0]+'.pkl'))
			# print(audioFileName, ':  ', logMelSpectro.shape)

	def TrainEvalTestSplit(self):
		'''
		split train valid test file.
		'''
		fileNameList = os.listdir(os.path.join(self.soundTrackPath, 'mp3', 'Soundtrack360_mp3'))
		fileNameList = [x for x in fileNameList if x.split('.')[1] == 'mp3']
		shuffle(fileNameList)
		trainSize = int(len(fileNameList) * config["TRAIN_PERCENT"])
		validSize = int(len(fileNameList) * config["VALID_PERCENT"])
		with open(os.path.join(self.fileFolderPath, 'train.txt'), 'w+') as fo_train:
			for fileName in fileNameList[0: trainSize]:
				fo_train.write('{}\n'.format(fileName))
		with open(os.path.join(self.fileFolderPath, 'valid.txt'), 'w+') as fo_valid:
			for fileName in fileNameList[trainSize: trainSize + validSize]:
				fo_valid.write('{}\n'.format(fileName))
		with open(os.path.join(self.fileFolderPath, 'test.txt'), 'w+') as fo_test:
			for fileName in fileNameList[trainSize + validSize: ]:
				fo_test.write('{}\n'.format(fileName))

	def loadDataSetFileName(self, train_test_valid):
		'''
		Getting dataset file name lsit for Train/Test/Valid.
		Input:
			train_test_valid: 'Train', 'Test' or 'Valid'
		Output:
			nameList: dataset name list
		'''
		nameList = []
		with open(os.path.join(self.fileFolderPath, train_test_valid.lower() + '.txt'), 'r') as fo:
			for line in fo.readlines():
				nameList.append(line.strip('\n'))
		return nameList

	def loadMelMidFeaEmo_audio(self, audioFileName, EmoInfo, df_MidF_anno):
		'''
		load logSpectro, target for a certain song piece.
		Input: 
			audioFileName: song clip name in soundTrack dataset.
			EmoInfo: emotion target info loaded from xls file.
			df_Mid_anno: mid level feature info ,in dataframe mode, loaded form csv file.
		Output:
			logMelSpectro_audio: logspectrogram for a certain song.
			info_audio: target info for a certain song.
		'''
		# info = ['valence', 'energy', 'tension', 'anger', 'fear', 'happy', 'sad', 'tender', 'TARGET',
		# 			'melody', 'articulation', 'rhythm_complexity', 'rhythm_stability', 'dissonance', 'atonality', 'mode']
		logMelSpectro_audio = torch.load(os.path.join(self.melFolderPath, audioFileName.split('.')[0]+'.pkl'))
		sourceID = int(audioFileName.split('.')[0])
		MidFID = sourceID + 3575
		info_audio = []
		for i in range(8):
			info_audio.append(EmoInfo.cell(sourceID, i + 1).value * 0.1)
		for i in range(7):
			info_audio.append(df_MidF_anno.iloc[MidFID - 1, i + 1] * 0.1)
		# df_raw.iloc[index, 1]
		# print(audioFileName, info_audio)
		return logMelSpectro_audio, info_audio

	def datasetLoader(self, train_test_valid):
		'''
		dataset loader for train/valid/test.
		Input:
			train_test_valid: for which dataset you would like to load, 'train'/'valid'/'test'
		Output:
			X, Y: dataset
		'''
		# self.TrainEvalTestSplit()
		X = []
		Y = []
		EmoInfo = xlrd.open_workbook(os.path.join(self.soundTrackPath, 'mean_ratings_set1.xls')).sheet_by_index(0)
		df_MidF_anno = pd.read_csv(os.path.join(self.midLevelPath, 'metadata_annotations', 'annotations.csv'))
		# df_MidF_meta = pd.read_csv(os.path.join(self.midLevelPath, 'metadata_annotations', 'metadata.csv'), delimiter=";")s
		# df_MidF_meta = df_MidF_meta[df_MidF_meta.Source == 'soundtracks']
		# print(df_MidF_anno)
		# print(df_MidF_meta)
		# self.loadMelMidFeaEmo_audio('001.mp3', EmoInfo, df_MidF_anno)
		# return
		fileNameList = self.loadDataSetFileName(train_test_valid)
		for audioFileName in fileNameList:
			logMelSpectro_audio, info_audio = self.loadMelMidFeaEmo_audio(audioFileName, EmoInfo, df_MidF_anno)
			X.append(logMelSpectro_audio)
			Y.append(info_audio)
		X = torch.from_numpy(np.array(X))
		Y = torch.from_numpy(np.array(Y))
		X = X.unsqueeze(1)
		X = X.float()
		Y = Y.float()
		print('-------------Loading {}--------------\n'.format(train_test_valid), X.shape, '	', Y.shape)
		# print(X.shape, Y.shape)
		return X, Y


if __name__ == '__main__':
	dataloader = MidEmoDataLoader()
	# dataloader.audio2Mel()
	# dataloader.TrainEvalTestSplit()
	# X, Y = dataloader.datasetLoader('valid')




# def parse_args():
# 	'''
# 	'''
# 	description = 'options'
# 	parser = argparse.ArgumentParser(description=description)
# 	parser.add_argument('--option',help = '')
# 	args = parser.parse_args()
# 	return args


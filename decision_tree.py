#-*-coding:uft-8-*-

import numpy as np
import math

def getEntropy(DataSet):
	classes = DataSet[:,-1]
	classCount = {}
	for cla in classes:
		if cla in classCount:
			classCount[cla] += 1
		else:
			classCount[cla] = 1
	totalCount = len(classes)+ 0.0
	probs = [classCount[cla] / totalCount for cla in classes]
	entroy = 0
	for prob in probs:
		entroy -= prob * math.log(prob,2)
	return entroy

def selectBestAtt(DataSet):
	attNum = len(DataSet[0]) - 1
	dataCount = len(DataSet) + 0.0
	minConEntroy = math.log(dataCount,2) + 1 
	bestFeature = -1
	for i in xrange(attNum):
		conEntroy = 0
		attVals = DataSet[:,i]
		uniAttVals = set(attVals)
		for val in uniAttVals:
			selDataSet = [data for data in DataSet if data[i] == val]
			conEntroy += len(selDataSet) / dataCount *getEntropy(selDataSet)
		if conEntroy < minConEntroy:
			bestFeature = i
			minConEntroy = conEntroy
	return bestFeature

def splitData(DataSet,feature):
	i = feature
	featureVals = DataSet[:,i]
	sampleCount = len(featureVals)
	splitDataSets = dict.fromkeys(featureVals)
	splitDataSet = np.hstack((DataSet[:,:i],DataSet[:,i+1:]))
	for val in splitDataSets:
		index = [i for i in xrange(sampleCount) if featureVals[i] == val]
		splitDataSets[val] = [splitDataSet[i] for i in index]	
	return splitDataSets



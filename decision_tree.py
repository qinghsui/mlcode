#-*-coding:utf-8-*-

import numpy as np
import math
from collections import Counter

def getEntropy(DataSet):
	classes = DataSet[:,-1]
	classCount = {}
	for cla in classes:
		if cla in classCount:
			classCount[cla] += 1
		else:
			classCount[cla] = 1
	totalCount = len(classes)+ 0.0
	probs = [classCount[cla] / totalCount for cla in classCount]
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
			selDataSet = np.array([data for data in DataSet if data[i] == val])
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
		splitDataSets[val] = np.array([splitDataSet[i] for i in index])
	return splitDataSets

def majority(DataSet):
	classes = DataSet[:,-1]
	classCount = np.unique(classes, return_counts=True) #return two arrays:  one is unique element, the other is its frequency 
	return classCount[1][0]

def createDesicionTree(DataSet,lable):
	classes = DataSet[:,-1]
	if np.unique(classes, return_counts=True)[1][0] == len(classes):
		return classes[0]
	if len(DataSet[0]) == 1:
		return majority(DataSet)
	bestFeature = selectBestAtt(DataSet)	
	bestFeatureName = lable[bestFeature]
	tree = {bestFeatureName:{}}
	del lable[bestFeature]
	splitDataSets = splitData(DataSet,bestFeature)
	for val in splitDataSets:
		sublable = lable[:] #why?
		tree[bestFeatureName][val] = createDesicionTree(splitDataSets[val],sublable)
	return tree

def createDataSet():
	dataSet = [ [ 1,1,'yes'],
					[ 1,1,'yes'],
					[1,0,'no'],
					[0,1,'no'],
					[0,1,'no'] ]
	lables = ['no surfacing','flippers']
	return dataSet,lables


if __name__ == "__main__":
	dataSet,lables = createDataSet()
	print dataSet
	DataSet = np.array(dataSet)
	mytree = createDesicionTree(DataSet,lables)
	print mytree
	



import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision.models as models
import os
from PIL import Image
import PIL
import csv
from urllib.parse import urlparse
from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import simple_preprocess
from gensim.parsing.porter import PorterStemmer
from gensim import corpora
import csv


#only going to try to classify EN ones and throw out ones that is in other languages by taking a random guess between the 4

throwoutID = [] #keep track of all the tweets we are going to throw out due to different languages
throwoutUser = [] #keep track of all the username of tweets that we are throwing out
pathTrainX = os.path.join(os.getcwd(),"p_train_x.csv") #the file path for the training data
pathTrainY = os.path.join(os.getcwd(),"p_train_y.csv") 
pathTestX = os.path.join(os.getcwd(),"p_test_x.csv") 
pathModel = os.path.join(os.getcwd(),"model.pt")
pathFinal = os.path.join(os.getcwd(),"finalPred.csv")
trainDataX = []
processDataX = []
trainDataY = []
porter = PorterStemmer()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epoach = 30
labelCount = 4

#todo prediction for non english ones
class TokenClassification(nn.Module): #model for log regression with softmax instead of sigmod since it is not binary
	def __init__(self, yLabelSize, dictionarySize):
		super(TokenClassification,self).__init__()

		self.linear = nn.Linear(dictionarySize,yLabelSize)

	def forward(self,vector):
		return self.linear(vector)


def CreateLabel(yLabel):
	if (yLabel == 0):
		return torch.tensor([0],dtype=torch.long,device=device)
	elif(yLabel == 1):
		return torch.tensor([1],dtype=torch.long,device=device)
	elif(yLabel == 2):
		return torch.tensor([2],dtype=torch.long,device=device)
	else:
		return torch.tensor([3],dtype=torch.long,device=device)

def CreateTensor(tokenList , myDictionary):
	vector = torch.zeros(len(myDictionary),dtype=torch.float64)
	for token in tokenList:
		try: #this is used for words on testing set that has not been seen by the data yet
			tempIndex =myDictionary.token2id[token] 
			vector[tempIndex] += 1
		except:
			pass

	return vector.view(1,-1).float()

#helper function to me remove @ and # and stuff from twitter text
def removeTwitter(text):
	newString = ''
	for i in text.split():
		s,n,p,pa,q,f = urlparse(i)
		if s and n:
			pass
		elif i[:1] == "@":
			pass
		elif i[:1] == "#":
			pass
			#newString = newString.strip() + ' ' + i[1:] #change this depending on if i want to strip away the #. current set to remove all text following #
		else:
			newString = newString.strip() + ' ' + i

	return newString
nonEnUserDictionary = {} # a dictionary that will contain the freq of the likes of the user

trainingSplit = 100000000
countSplit = 0

testDatax = []
testDatay = []


with open(pathTrainX,encoding = 'utf8') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	for row in readCSV: 
		if (countSplit < trainingSplit):
			if (row[8] == "en"):
				#giving each text it's own token
				processingString = remove_stopwords(removeTwitter(row[7]))
				stringTokens = simple_preprocess(processingString,deacc=True, max_len=16)
				stringTokens.append(row[4])
			
				porterTokens = porter.stem_documents(stringTokens) #using porter stemming to remove the words
				processDataX.append(porterTokens) 
			elif (row[0] == "id" ):
				continue #igoring the inital lines in csv
			else:
				throwoutID.append(row[0])
				throwoutUser.append(row[4])
		else:
			testDatax.append(row)

		countSplit += 1


countSplit = 0
with open(pathTrainY,encoding='utf8') as csvfile:
	readCSV = csv.reader(csvfile,delimiter=',')
	for row in readCSV:
		if (row[0] == "id"):
			continue #skipping the first line
		label = 0
		try:
			label = int(row[1])
		except:
			print("should never come in here")
		if (countSplit < trainingSplit):
			if (not row[0] in throwoutID):
				trainDataY.append(label) #using all int for labels rather than strings
			else: # for non en tweets, going to predict purly based on the most common label for that user
				throwoutIndex = throwoutID.index(row[0])
				if (throwoutUser[throwoutIndex] in nonEnUserDictionary):
					nonEnUserDictionary[throwoutUser[throwoutIndex]][label] += 1
				else:
					nonEnUserDictionary[throwoutUser[throwoutIndex]] = [0,0,0,0] 
					nonEnUserDictionary[throwoutUser[throwoutIndex]][label] += 1
		else:
			testDatay.append(label)

		countSplit += 1




#doing tow for training
print("done processing files and strings")

dictionary = corpora.Dictionary(processDataX)

logmodel = TokenClassification(labelCount,len(dictionary))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(logmodel.parameters(),lr=0.01)
count = 1
for i in range(epoach):

	for index, tokenString in enumerate(processDataX):
		optimizer.zero_grad()

		inputVector = CreateTensor(tokenString,dictionary)
		outputVector = CreateLabel(trainDataY[index])
		prob = logmodel(inputVector)
		loss = criterion(prob,outputVector)
		loss.backward()
		optimizer.step()

	'''correctCount = 0
	
	with torch.no_grad():
		for index, row in enumerate(testDatax):
			prediction = 0
			if (row[8] == "en"):
				#giving each text it's own token
				processingString = remove_stopwords(removeTwitter(row[7]))
				stringTokens = simple_preprocess(processingString,deacc=True, max_len=16)
				stringTokens.append(row[4])
			
				tokenString = porter.stem_documents(stringTokens)
				inputVector = CreateTensor(tokenString,dictionary)
				prediction = torch.argmax(logmodel(inputVector),dim=1).cpu().numpy()[0]
			else:
				if ((row[4]) in nonEnUserDictionary):#if it is in the dictionary
					freqArray = nonEnUserDictionary[row[4]]
					prediction = freqArray.index(max(freqArray))
				else:
					prediction = 1 #this is simply the average of all the non en tweets

			if (testDatay[index] == prediction):
				correctCount += 1'''
	print(count)
	count += 1

	torch.save(logmodel,pathModel)

with torch.no_grad():
	with open(pathFinal,'w',newline='') as file:
		writer = csv.writer(file)
		writer.writerow(["id,label"])
		with open(pathTestX,encoding='utf8') as csvfile:
			readCSV = csv.reader(csvfile,delimiter=',')
			prediction = 0
			for row in readCSV:
				if (row[0] == "id" ):
					continue
				elif (row[8] == "en"):
					#giving each text it's own token
					processingString = remove_stopwords(removeTwitter(row[7]))
					stringTokens = simple_preprocess(processingString,deacc=True, max_len=16)
					stringTokens.append(row[4])
				
					tokenString = porter.stem_documents(stringTokens)
					inputVector = CreateTensor(tokenString,dictionary)
					prediction = torch.argmax(logmodel(inputVector),dim=1).cpu().numpy()[0]
				else:
					if ((row[4]) in nonEnUserDictionary):#if it is in the dictionary
						freqArray = nonEnUserDictionary[row[4]]
						prediction = freqArray.index(max(freqArray))
					else:
						prediction = 1 #this is simply the average of all the non en tweets
				writer.writerow([row[0],str(prediction)])

		








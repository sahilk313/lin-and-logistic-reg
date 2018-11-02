#in part 6, normalization is to be done
#change training set for each lambda 100 times

import sys
import random
import math
import numpy		#Note that numpy needs to be installed
import matplotlib.pyplot as plt
sys.setrecursionlimit(10**6)

linregdata1 = [line.rstrip('\n') for line in open('linregdata')]
linregdata = []
for item in linregdata1:
	lis = []
	splitted = item.split(",")
	if splitted[0]=='M':
		lis.append(0.0)
		lis.append(0.0)
		lis.append(1.0)
	elif splitted[0]=='F':
		lis.append(1.0)
		lis.append(0.0)
		lis.append(0.0)
	elif splitted[0]=='I':
		lis.append(0.0)
		lis.append(1.0)
		lis.append(0.0)
	index = 0
	for it in splitted:
		if index==0:
			index = 1
		else:
			lis.append(float(it))		#note that even the age to be predicted is stored in float format
	linregdata.append(lis)

linregg = linregdata.copy()

for i in range(0,10):
	alist = []
	for item in linregdata:
		alist.append(item[i])
	mean = numpy.mean(alist,axis=0)
	stdev = numpy.std(alist,axis=0)
	for item in linregdata:
		item[i] = (item[i]-mean)
		if not stdev==0:
			item[i] = item[i]/stdev


trainset = []
testset = []
linregdataDict = {}
i = 0
for line in linregdata:
	i = i + 1
	linregdataDict[i] = line

linreggdict = linregdataDict.copy()

randomIndices = random.sample(range(1,i+1),math.floor(i/5))		#separating 20% as test set
randomIndicesDict = {}
for item in randomIndices:
	randomIndicesDict[item] = 1
for key in linregdataDict:
	if key in randomIndicesDict:
		testset.append(linregdataDict[key])
	else:
		trainset.append(linregdataDict[key])

def mylinridgereg(X,Y,lambd):	
	a = X.shape
	x_m = a[1]
	I = numpy.eye(x_m)			#I is float type
	I = numpy.multiply(I,lambd)
	xt = X.transpose()
	ans = numpy.dot(xt,X)
	ans = numpy.add(ans,I)
	ans = numpy.linalg.inv(ans)
	temp = numpy.dot(xt,Y)
	return numpy.dot(ans,temp)

def mylinridgeregeval(X, weights):
	return numpy.dot(X,weights)

def meansquarederr(T,Tdash):
	ans = 0.0
	n = len(T)
	for i in range(0,n):
		ans = ans + (T[i]-Tdash[i])**2
	ans = ans/n
	return ans

def predictor(trainset1,testset1,lam):
	x_n = len(trainset1)
	x_m = 10+1
	X = numpy.zeros(shape=(x_n,x_m))
	Y = numpy.zeros(shape=(x_n,1))
	i=0				#remember to increment i and j
	for item in trainset1:
		j=0
		for it in item:
			if j==x_m:
				Y[i] = it
				break
			elif j==0:
				X[i][j] = 1
				X[i][j+1] = it
				j = j+1
			else:
				X[i][j] = it
			j = j+1
		i = i+1
	#check that output is being normalized or not during training
	#x = numpy.array([1.])
	#x = numpy.append(x,testset[0][:-1])
	X_test = numpy.zeros(shape=(len(testset1),x_m))
	Y_test = numpy.zeros(shape=(len(testset1),1))
	i=0
	for item in testset1:
		x = numpy.array([1.])
		x = numpy.append(x,testset1[i][:-1])
		for j in range(0,x_m):
			X_test[i][j] = x[j]
		Y_test[i] = testset1[i][-1]
		i = i+1
	#print("lambda =",lam)
	#print("Error for test data:\t",meansquarederr(Y_test,mylinridgeregeval(X_test,mylinridgereg(X,Y,lam))))
	#print("Error for training data:",meansquarederr(Y,mylinridgeregeval(X,mylinridgereg(X,Y,lam))))
	#print()
	return [meansquarederr(Y_test,mylinridgeregeval(X_test,mylinridgereg(X,Y,lam))),meansquarederr(Y,mylinridgeregeval(X,mylinridgereg(X,Y,lam)))]

print("for frac = 1,")
for item in [0.1,0.2,0.6,1,2,5,10]:
	testandtrainprediction = predictor(trainset,testset,item)
	print("lambda =",item)
	print("Error for test data:\t",testandtrainprediction[0])
	print("Error for training data:",testandtrainprediction[1])
	print()


def part6(frac,lambda_):
	
	'''for i in range(0,10):
		alist = []
		for item in trainset_1:
			alist.append(item[i])
		mean = numpy.mean(alist,axis=0)
		stdev = numpy.std(alist,axis=0)
		for item in trainset_1:
			item[i] = (item[i]-mean)
			if not stdev==0:
				item[i] = item[i]/stdev
		for item in testset_1:
			item[i] = (item[i]-mean)
			if not stdev==0:
				item[i] = item[i]/stdev'''
	#print("frac:",frac,"num=",numsampled)
	listoftrainerror = []
	listoftesterror = []	
	mintesterror = 1000
	mintesterrorlambda = 0
	for lmm in lambda_:
		avgmeanerrortrain = 0.0
		avgmeanerrortest = 0.0
		for itera in range(0,100):
			linregdata_ = linregg

			trainset_ = []
			testset_ = []
			linregdataDict_ = linreggdict

			randomIndices_ = random.sample(range(1,i+1),math.floor(i/5))		#separating 20% as test set
			randomIndicesDict_ = {}
			for item in randomIndices_:
				randomIndicesDict_[item] = 1
			for key in linregdataDict_:
				if key in randomIndicesDict_:
					testset_.append(linregdataDict_[key])
				else:
					trainset_.append(linregdataDict_[key])
			numsampled = math.floor(frac*len(trainset_))
			trainset_1 = random.sample(trainset_,numsampled)
			testset_1 = testset_.copy()
			anslis = predictor(trainset_1,testset_1,lmm)
			avgmeanerrortest = avgmeanerrortest+anslis[0]
			avgmeanerrortrain = avgmeanerrortrain+anslis[1]
		avgmeanerrortrain = avgmeanerrortrain/100
		avgmeanerrortest = avgmeanerrortest/100
		if avgmeanerrortest<mintesterror:
			mintesterror = avgmeanerrortest
			mintesterrorlambda = lmm
		listoftrainerror.append(avgmeanerrortrain)
		listoftesterror.append(avgmeanerrortest)
	plt.plot(listoflambda,listoftrainerror,color='green',label='trainerror')
	plt.plot(listoflambda,listoftesterror,color='red',label='testerror')
	plt.xlabel('lambda')
	ppt = 'error(for frac='+str(frac)+')'
	plt.ylabel(ppt)
	plt.legend()
	plt.show()
	return [mintesterror,mintesterrorlambda]
lisstoftesterror = []
lisstoflambda = []
listoffrac = [0.03,0.1,0.5,0.75]
for fract in listoffrac:
	listoflambda = numpy.linspace(0.1,2,3)
	templist = part6(fract,listoflambda)
	lisstoflambda.append(templist[1])
	lisstoftesterror.append(templist[0])

plt.plot(listoffrac,lisstoftesterror,color='green',label='minavgtesterror vs frac')
plt.plot(listoffrac,lisstoflambda,color='red',label='minavglambda vs frac')
plt.legend()
plt.show()

globallamda = 1
globalfrac = 0.75
linregdata_ = linregg
trainset_ = []
testset_ = []
linregdataDict_ = linreggdict

randomIndices_ = random.sample(range(1,i+1),math.floor(i/5))		#separating 20% as test set
randomIndicesDict_ = {}
for item in randomIndices_:
	randomIndicesDict_[item] = 1
for key in linregdataDict_:
	if key in randomIndicesDict_:
		testset_.append(linregdataDict_[key])
	else:
		trainset_.append(linregdataDict_[key])
numsampled = math.floor(globalfrac*len(trainset_))
trainset_1 = random.sample(trainset_,numsampled)
testset_1 = testset_.copy()
listofactual = []
listofprediction = []
trainset1 = linregg
x_n = len(trainset1)
x_m = 10+1
X = numpy.zeros(shape=(x_n,x_m))
Y = numpy.zeros(shape=(x_n,1))

i=0				#remember to increment i and j
for item in trainset1:
	j=0
	for it in item:
		if j==x_m:
			Y[i] = it
			break
		elif j==0:
			X[i][j] = 1
			X[i][j+1] = it
			j = j+1
		else:
			X[i][j] = it
		j = j+1
	i = i+1
finalweight = mylinridgereg(X,Y,globallamda)
aaanss = mylinridgeregeval(X,finalweight)
#print(aaanss.shape)
plt.scatter(list(aaanss),Y,color='green',label='actual value vs predicted value')
plt.show()
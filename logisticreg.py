#matplotlib is required
import matplotlib.pyplot as plt
import numpy
import math

def sigmoidfn(X,w):
	temp = numpy.zeros(shape=(X.shape[0],1))
	temp = numpy.dot(X,w)
	for i in range(0,temp.shape[0]):
		temp[i] = 1 / (1 + (math.exp((-temp[i]))))
	return temp

def gradientDescentWeight(X,w,alpha,y,numiter):
	for j in range(0,numiter):
		temp = numpy.subtract(sigmoidfn(X,w),y)
		temp = numpy.dot(X.transpose(),temp)
		temp = numpy.multiply(temp,1/len(temp))
		w = w - numpy.multiply(temp,alpha)
		#print(w)
	return w

def newtonRaphson(X,w,y,numiter):
	w = numpy.zeros(shape=(X.shape[1],1))
	w[:,0] = 0.1
	for j in range(0,numiter):
		R = numpy.zeros(shape=(y.shape[0],y.shape[0]))
		for m in range(0,y.shape[0]):
			for n in range(0,y.shape[0]):
				if m==n:
					tempmat = numpy.zeros(shape=(1,X.shape[1]))
					for k in range(0,X.shape[1]):
						#print(tempmat[0][k])
						tempmat[0][k] = X[m][k]
					fx = sigmoidfn(tempmat,w)
					R[m][m] = fx*(1-fx)
		H = numpy.dot(X.transpose(),R)
		H = numpy.dot(H,X)
		H = numpy.linalg.pinv(H)
		H = numpy.dot(H,X.transpose())
		w = w - numpy.dot(H,numpy.subtract(sigmoidfn(X,w),y))
	return w

def prediction(x,w):
	pred = sigmoidfn(x,w)[0][0]
	if(pred>=0.5):
		return 1
	return 0

def featureTransform(X,degree):
	size = int((degree+2)*(degree+1)/2)
	transform = numpy.zeros(shape=(X.shape[0],size))
	transform[:,0] = 1
	index = 1
	for i in range(1,degree+1):
		for j in range(0,i+1):
			aterm = X[:,0]
			bterm = X[:,1]
			if not X.shape[1]==2:
				aterm = X[:,1]
				bterm = X[:,2]
			aterm = numpy.power(aterm,i-j)
			bterm = numpy.power(bterm,j)
			transform[:,index] = numpy.multiply(aterm,bterm)
			index = index+1
	return transform

def plotDecisionBoundary(X,w,degree,y):
	plt.scatter(posx,posy,color='green',label='credit-approved')
	plt.scatter(negx,negy,color='red',label='credit-rejected')
	u = numpy.linspace(0, 6, 500)
	v = numpy.linspace(0, 7, 500)
	z = numpy.zeros(shape=(len(u),len(v)))
	for i in range(0,len(u)):
		for j in range(0,len(v)):
			matof2 = numpy.zeros(shape=(1,2))
			matof2[0][0] = u[i]
			matof2[0][1] = v[j]
			z[i][j] = numpy.dot(featureTransform(matof2,degree),w);
	z = z.transpose()
	plt.contour(u, v, z,[0,0])
	plt.show()

logisregdata1 = [line.rstrip('\n') for line in open('credit.txt')]
logisregdata = []
for item in logisregdata1:
	lis = []
	splitted = item.split(",")
	for it in splitted:
		lis.append(float(it))
	logisregdata.append(lis)

posx = []
posy = []
negx = []
negy = []
for item in logisregdata:
	if item[2]==0:
		negx.append(item[0])
		negy.append(item[1])
	else:
		posx.append(item[0])
		posy.append(item[1])

plt.scatter(posx,posy,color='green',label='credit-approved')
plt.scatter(negx,negy,color='red',label='credit-rejected')
plt.xlabel('x1-axis')
plt.ylabel('x2-axis')
plt.legend()
'''plt.show()'''

numex = len(logisregdata)
X = numpy.zeros(shape=(numex,3))
Y = numpy.zeros(shape=(numex,1))
i = 0
for item in logisregdata:
	X[i][0] = 1
	X[i][1] = item[0]
	X[i][2] = item[1]
	Y[i] = item[2]
	i = i + 1
w = numpy.zeros(shape=(3,1))

#learntweight = gradientDescentWeight(X,w,0.01,Y,5000)
'''Xtransformed = featureTransform(X,2)
learntweight = newtonRaphson(Xtransformed,w,Y,100)
plotDecisionBoundary(X,learntweight,2,Y)'''
learntweightlogistic = gradientDescentWeight(X,w,0.01,Y,5000)
learntweight = newtonRaphson(X,w,Y,5000)

meanerror = 0
#print(learntweight)
for item in logisregdata:
	x = numpy.zeros(shape=(1,3))
	x[0][0] = 1
	x[0][1] = item[0]
	x[0][2] = item[1]
	meanerror = meanerror+(prediction(x,learntweight)-item[2])*(prediction(x,learntweight)-item[2])
meanerror = meanerror
print("mean error for gradient descent: ",meanerror)

slope = (-learntweight[0]/learntweight[2])/(learntweight[0]/learntweight[1])
intercept = (-learntweight[0]/learntweight[2])
xaxis = numpy.linspace(min(X[:,1]),max(X[:,1]),50)
yaxiss = numpy.multiply(xaxis,slope)
yaxiss = numpy.add(yaxiss,intercept)
yaxis = []
for index in range(0,yaxiss.shape[0]):
	yaxis.append(yaxiss[index])
plt.plot(xaxis,yaxis,color='blue')
plt.ylim(top=6)
plt.ylim(bottom=0)

learntweight= learntweightlogistic
meanerror = 0
#print(learntweight)
for item in logisregdata:
	x = numpy.zeros(shape=(1,3))
	x[0][0] = 1
	x[0][1] = item[0]
	x[0][2] = item[1]
	meanerror = meanerror+(prediction(x,learntweight)-item[2])*(prediction(x,learntweight)-item[2])
meanerror = meanerror
print("mean error for newton Raphson: ",meanerror)

slope = (-learntweight[0]/learntweight[2])/(learntweight[0]/learntweight[1])
intercept = (-learntweight[0]/learntweight[2])
xaxis = numpy.linspace(min(X[:,1]),max(X[:,1]),50)
yaxiss = numpy.multiply(xaxis,slope)
yaxiss = numpy.add(yaxiss,intercept)
yaxis = []
for index in range(0,yaxiss.shape[0]):
	yaxis.append(yaxiss[index])
plt.plot(xaxis,yaxis,color='red')
plt.ylim(top=6)
plt.ylim(bottom=0)
print("red line is gradient descent")
print("blue line is newton raphson")
plt.show()
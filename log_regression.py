from numpy import loadtxt, where
from pylab import scatter, show, legend, xlabel, ylabel, plot
from math import exp, log, e
import scipy.optimize as opt
import numpy as np
from sklearn import datasets, neighbors, linear_model, metrics

def sigmoid(X):
	return 1 / (1 + np.exp(-X))

def compute_cost(theta, X, y):
	#number of training
	m = X.shape[0]
	#print('X = ', X)
	#print('y = ', y)
	#print('theta = ', theta)
	#input('Press enter ...')
	h = sigmoid(X.dot(theta))
	#Calculate the cost function
	J = (-1./m)*(np.dot(y,np.log(h)) 
		+ np.dot(1-y, np.log(1-h) ) )
	if np.isnan(J):
		return np.inf
	return J

def gradient(theta, X, y):
	#print(X)
	m = X.shape[0]
	h = sigmoid( np.dot(X,theta) )
	X_1 = np.matrix(X)
	error = -(y - h)/m
	theta = np.array(error*X_1)
	return theta[0]

#load the dataset


data = datasets.load_iris()
X = data.data[:100, :2]
y = data.target[:100]
X_1 = np.append( X, np.ones((X.shape[0], 1)), axis=1 )
#print(X_1)
#print(y)

print('------------------')
'''
data = np.loadtxt('ex2data1.txt',delimiter=",")
m,n = data.shape
X = np.array(data[:,:-1])
y = np.array(data[:,2].reshape(m,1))
y = y[:,0]
X_1 = np.append( X, np.ones((X.shape[0], 1)), axis=1 )
#print(X_1)
#print(y[:,0])
'''

#Use gradient descent

theta = np.array([21, -19, -53])
#theta = np.array([0,0,-4])
cost = compute_cost(theta, X_1, y)
cost_change = 1
precision = 1e-8
while (abs(cost_change) > precision):
	cost = compute_cost(theta, X_1, y)
	theta = theta + 0.01*gradient(theta, X_1, y)
	cost_change = cost - compute_cost(theta, X_1, y)

print('my pred = ', theta)
print('Cost for me = ', compute_cost(theta, X_1, y) )
#Use sklearn 
logreg = linear_model.LogisticRegression()
logreg.fit(X, y)
print(sum(y == logreg.predict(X)))

theta_s = logreg.coef_
theta_s = theta_s[0]
theta_s = np.append(theta_s, logreg.intercept_)
print('theta_s = ', theta_s)
print('Cost = ', compute_cost(theta_s, 
	X_1, y) )

from scipy.optimize import fmin_bfgs
#normalize data
#norm_X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
#norm_X = np.append( norm_X, np.ones((norm_X.shape[0], 1)), 
#	axis=1 )
arguments = (X_1, y)
x0 = np.array([0,0,0])
print('x0 = ', x0)
theta_d = fmin_bfgs(compute_cost, x0, 
	fprime=gradient, args=arguments)
print('fmin_bfgs = ', theta_d)



x1 = np.linspace(4, 8, 1000)
x2 = (-theta[2] - theta[0]*x1)/theta[1]
x3 = (-theta_s[2] - theta_s[0]*x1)/theta_s[1]
x4 = (-theta_d[2] - theta_d[0]*x1)/theta_d[1]
scatter(X[:50,0], X[:50,1], marker='o', c='b')
scatter(X[50:,0], X[50:,1], marker='x', c='r')
plot(x1, x2, c='g')
plot(x1, x3, c='y')
plot(x1, x4, c='r')
xlabel('Exam 1 score')
ylabel('Exam 2 score')
#legend(['Admitted', 'Not Admitted'])
show()

if __name__ == '__name__':
	main()
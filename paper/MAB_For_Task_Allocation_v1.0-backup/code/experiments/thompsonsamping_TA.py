#/usr/bin/env python
#-*- encoding:utf-8 -*-
import data_operation    #contactstat
import random
import task
from scipy import stats
import math
from scipy import linalg
import numpy as np
from matplotlib import pylab as plt
#contactdict = data_operation.contactdict   #user contacts


#tasks: 

#6个月的时刻
starttime = 24485668
#总的时间
Time = 35242046        #diff 10756378  约为4个月
task_generate_rate = 1800  # 每隔半个小时产生一个task, 总共会产生约5975个task



def getTaskValue(duration):
	#value~~N(mean, variance)
	mean = 21600 #6 hour
	variance = 1800
	value = (1.0/math.pow(2*math.pi*variance,1.0/2))*math.exp(-math.pow(duration-mean,2)/2.0*variance)
	return value

#nodetasks = dict()
#初步假定只有一个节点产生任务
tasksdict = dict()
tid = 1

# for t in range(starttime, Time, task_generate_rate):
# 	#node = random.randint(1,100)
# 	duration = random.randint(3600, 36000)           #one hour--ten hour
# 	value = getTaskValue(duration)

# 	task = Task(tid, duration, value)
# 	tasksdict[t] = task
# 	tid += 1
# print 'tid',tid

def init(D, ARMS):
    '''    
    Initialize algorithm with input dimension D, arms ARMS.    

    Arguments: 
     - ``D``: Input dimensionality as int
     - ``ARMS``: List of arm names       
    '''  
    init_a = np.diag(np.ones(D),0)
    init_b = np.zeros((D,1))   #D rows, 1 columns
    a_inv = linalg.inv(init_a)
    return [ {'name': a, 'a' : init_a, 
              'a_inv': a_inv, 
              'b' : init_b, 
              'total_reward' : 0.0
              } for a in ARMS ]



def  thompsonSampling(d, b):
	'''

	- d: feature dimension
	'''
	mu_true = np.random.rand(d)
	#单位矩阵
	B = np.identity(d)   #d*d
	mu_param = np.zeros(d)
	f = np.zeros(d)
	R = 1.0
	epsilon = 0.1
	delta = 0.1

	v = R*math.pow((24.0/epsilon)*d*math.log(1.0/delta),1/2)

	selectedArm = dict()
	regret = 0
	cumulateregret = []
	reward = []

	for t in range(1000):
		B_inv = linalg.inv(B) #inverse
		temp = v*v*B_inv
		
		deviation = np.sqrt(temp.sum(1))
		mu_sample = np.random.normal(mu_param, deviation)

		#对每一个context 计算 b*mu
		
		for i in range(len(b)):
			maxValue = 0.0
			selecteIndex = 0

			true_maxValue = 0.0
			true_index = 0
			if maxValue < np.dot(b[i].T,mu_sample):
				maxValue = np.dot(b[i].T, mu_sample)
				selecteIndex = i
			#true reward
			if true_maxValue < np.dot(b[i].T, mu_true):
				true_maxValue = np.dot( b[i].T, mu_true)
				true_index = i

			if true_index == selecteIndex:
				regret += 0
			else:
				regret += (true_maxValue-maxValue)
			cumulateregret.append(regret)



		B += np.dot(b[selecteIndex], b[selecteIndex].T)
		f += b[selecteIndex]*maxValue
		mu_param = np.dot(B_inv, f)
		reward.append(maxValue)
		#print 'selected arm',selecteIndex,'regret', cumulateregret


	plt.semilogy(reward)   #对数坐标图
	plt.title('reward')
	plt.ylabel('Cumulative reward')
	plt.xlabel('Round Index')
	plt.legend(('Reward'),loc='lower right')
	plt.show()

	plt.plot(cumulateregret)
	plt.show()


d=4
b=np.random.rand(10,4)
thompsonSampling(d,b)

# def thompsonTaskAllocation(T):

# 	for t in range(starttime,T):
# 		if tasksdict.has_key(t):
# 			#allocate task

# 	return 




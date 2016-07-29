#/usr/bin/env python
#-*- encoding:utf-8 -*-
#import data_operation    #contactstat
import random
from task import Task
from scipy import stats
import math
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
from data_operation import contactdict
from workerNode import Worker
from matplotlib.patches import ConnectionPatch
#contactdict = data_operation.contactdict   #user contacts

def getTaskValue(duration):
	if duration < 6 * 3600:
		value = round(duration / 36000.0, 3)
	elif duration < 3600 * 7:
		value = 0.6
	else:
		value = round(0.8 - duration / 36000.0, 3)

	return value if value > 0 else 0.1


def getContextArray(rid, task, workers):
	'''
	 Get the requester's all worker's context, return an array
	'''
	context_list = list()

	#the worker's buffer size is full then it can't accomplish more tasks
	for worker in workers:
		if worker.buffersize<=0:
			workers.remove(worker)

	for worker in workers:
		temp = worker.getContext(task.value)
		#temp.append(task.duration)
		temp.append(task.value)
		context_list.append(temp)
	return np.array(context_list)

def generate_true_data(K, num_samples):
	generated_data = np.tile(np.random.rand(K),(num_samples,1))   #tile重复的扩充数组，把第一个元素当成一个元素，后面指定几行几列
	true_rewards = np.random.rand(num_samples,K) < generated_data
	return true_rewards,generated_data

def generate_true_mu(k, num_samples):
	mu_true = []

	for i in range(num_samples):
		mean = np.random.randint(-100, 100, size=k)
		variance = np.random.random(size=k)
		mu_true.append(np.random.normal(mean, variance))

	return np.array(mu_true)


def generateTask(generate_rate):
	taskslist = []   #初步假设在某一个时刻只产生一个task
	tid = 1
	for t in range(starttime, Time, generate_rate):
		#node = random.randint(1,100)
		duration = random.randint(1800, 36000)           #half hour--ten hour
		value = getTaskValue(duration)

		task = Task(tid, duration, value, random.randint(1,3))
		taskslist.append(task)
		tid += 1
	print 'tid',tid
	return taskslist

'''
# def  thompsonSampling(d, b, armsNumber, task):
#
# 	- d: feature dimension
# 	- b: context
# 	- armsNumber: the number of arms
#
# 	task_number = len(task)
# 	true_rewards, generated_data = generate_true_data(armsNumber, task_number)
# 	nums_samples, k = true_rewards.shape
#
# 	mu_true = np.random.rand(d)
# 	#单位矩阵
# 	B = np.identity(d)   #d*d
# 	mu_param = np.zeros(d)
# 	f = np.zeros(d)
# 	R = 1.0
# 	epsilon = 0.1
# 	delta = 0.1
#
# 	v = round(R*math.pow((24.0/epsilon)*d*math.log(1.0/delta),1.0/2), 3)
#
# 	selectedArm = dict()
# 	#cumulateregret = []
# 	regret = np.zeros(task_number)
# 	b_array = np.array(b)
#
# 	#for t in range(0, num_samples):
# 	for t in task:
# 		B_inv = linalg.inv(B) #inverse
# 		temp = v*v*B_inv
#
# 		deviation = np.sqrt(temp.sum(1))
# 		mu_sample = np.random.normal(mu_param, deviation)
#
# 		#对每一个context 计算 b*mu
# 		selecteIndex = np.argmax(np.dot(b_array, mu_sample))
# 		# if true_rewards[t, selecteIndex] == 1:
# 		# 	reward = 0
# 		# else:
# 		# 	reward = 1
# 		reward = workers[selecteIndex].executeTask(task.tasktype)
#
# 		regret[t] = np.max(generated_data[t,:]) - generated_data[t,selecteIndex]
#
# 		B += np.dot(b[selecteIndex].T, b[selecteIndex])
# 		f += b[selecteIndex]*reward
# 		mu_param = np.dot(B_inv.T, f)
# 		#reward.append(maxValue)
# 	cum_regret = np.cumsum(regret)
# 	#print 'cum_regret',cum_regret
# 	return cum_regret
#
'''

def thompsonTaskAllocation(rid, dim, workers, tasklist):
	'''
	:param rid: requester's id
	:param dim: the dimension of context
	:param tasksdict: tasks
	:return:  cumulate regret
	'''

	regret = np.zeros(len(tasklist))
	reward_list = np.zeros(len(tasklist))
	#armsNumber = len(workers)

	#true_rewards, generated_data = generate_true_data(armsNumber, len(tasklist))

	#mu_true = np.random.random((len(tasklist),dim))
	mu_true = generate_true_mu(dim, len(tasklist))

	B = np.identity(dim)
	mu_param = np.zeros(dim)
	f = np.zeros(dim)
	R = 0.1
	epsilon = 1.0/math.log(len(tasklist))
	delta = 0.2

	v = round(R*math.pow((24.0/epsilon)*dim*math.log(1.0/delta),1.0/2), 3)

	index = 0
	for task in tasklist:
		context = getContextArray(rid, task, workers)
		if len(context) == 0:
			break
		B_inv = linalg.inv(B) #inverse
		temp = v*v*B_inv

		deviation = np.sqrt(temp.sum(1))
		mu_sample = np.random.normal(mu_param, deviation)

		#对每一个context 计算 b*mu
		selecteIndex = np.argmax(np.dot(context, mu_sample))

		reward = workers[selecteIndex].executeTask(task.tasktype)
		reward_list[index]=reward

		#regret[index] = np.max(generated_data[index,:]) - generated_data[index,selecteIndex]
		regret[index] = abs(np.max(np.dot(context, mu_true[index,:].T)) - np.dot(context[selecteIndex,:], mu_sample))
		index += 1

		#update parameters
		B += np.dot(context[selecteIndex].T, context[selecteIndex])
		f += context[selecteIndex]*reward
		mu_param = np.dot(B_inv.T, f)

	cum_regret = np.cumsum(regret)
	cum_reward = np.cumsum(reward_list)
	return cum_regret


d=6  #dimension of features
num_experiments = 100

#6个月的时刻
starttime = 24485668
#总的时间
Time = 35242046        #diff 10756378  约为4个月
# task_generate_rate_halfhour = 1800  # 每隔半个小时产生一个task, 总共会产生约5975个task

task_generate_rate = [1800, 3600, 3600 * 2, 3600 * 4, 3600 * 8, 3600 * 12, 3600 * 24]
task_generate_rate.reverse()
diff_rate = ['half hour', 'one hour', 'two hour', 'four hour', 'eight hour', "twelve hour", "one day"]
diff_rate.reverse()
#nodetasks = dict()
#初步假定只有一个节点产生任务

rid = '68'
workers = []

##used for plot
line_color = ["r","g","b","c","m","y","k"]
line_color.reverse()

# for wid in contactdict[rid]:
# 	workers.append(Worker(wid))
#regret_accumulator = np.zeros((num_samples, 7))

for i in range(len(task_generate_rate)):
	#cum_regret_sum[:,i] += thompsonTaskAllocation(Time, rid, generateTask(task_generate_rate[i]))
	tasklist = generateTask(task_generate_rate[i])
	regret_accumulator = np.zeros((len(tasklist), 1))
	for number in range(num_experiments):
		for wid in contactdict[rid]:
			workers.append(Worker(wid))
		regret_accumulator[:,0] += thompsonTaskAllocation(rid, d, workers, tasklist)  #allocate task
		workers = []
		print 'number',number
	print 'task generate rate', task_generate_rate[i]
	regret_accumulator /= num_experiments

	plt.semilogy(regret_accumulator, color=line_color[i], linewidth=2)   #对数坐标图
	#plt.plot(regret_accumulator, color=line_color[i], linewidth=2)   #对数坐标图
	plt.title('cum_regret_sum')
	plt.ylabel('Cumulative regret')
	plt.xlabel('Round Index')
	plt.grid(True)
	plt.legend(diff_rate,loc='lower right')
	#plt.show()
	plt.savefig('./fig_mu_true/'+diff_rate[i]+'.eps',format='eps')
	plt.savefig('./fig_mu_true/'+diff_rate[i]+'.png',format='png')



'''
###########################################################
#	draw the result picture
#	下面的代码实现了将左边图(p1)中局部的放大图在右边图(p2)中进行展示
###########################################################

# plt.figure(figsize=(16,8),dpi=75)
# p1 = plt.subplot(121, aspect=5/2.5)
# p2 = plt.subplot(122, aspect = 0.5/0.05)
#
# p1.semilogy(cum_regret_sum, linewidth=2)
# p1.set_title("Simulation thompson sampling for task allocation")
# p1.set_xlabel("Round Index")
# p1.set_ylabel('cumulative expected regret')
# p1.legend(('half hour', 'one hour', 'two hour', 'four hour', 'eight hour', 'twelve hour', 'one day'),loc='lower right')
# p1.grid(True)
#
# #p2 sub
# p2.axis([400, 600, 100, 500])  ##the range of x and y
# p2.semilogy(cum_regret_sum, linewidth=2)
# p2.set_title('Simulation thompson sampling for task allocation')
# p2.set_xlabel("Round Index")
# p2.set_ylabel('cumulative expected regret')
# p2.legend(('half hour', 'one hour', 'two hour', 'four hour', 'eight hour', 'twelve hour', 'one day'),loc='lower right')
# p2.grid(True)
#
# #plot the box
# tx0=400
# tx1=600
# ty0=180
# ty1=300
#
# sx = [tx0,tx1,tx1,tx0,tx0]
# sy = [ty0,ty0,ty1,ty1,ty0]
# p1.plot(sx,sy, 'purple')
#
# xy = (595, 298)
# xy2 = (402, 485)
# con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
# 					  axesA=p2,axesB=p1)
# p2.add_artist(con)
#
# xy=(595, 179)
# xy2=(402, 153)
# con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
# 					  axesA=p2,axesB=p1)
# p2.add_artist(con)
#
# plt.show()
# plt.semilogy(cum_regret_sum)   #对数坐标图
# plt.title('cum_regret_sum')
# plt.ylabel('Cumulative regret')
# plt.xlabel('Round Index')
# plt.grid(True)
# plt.legend(('half hour', 'one hour', 'two hour', 'four hour', 'eight hour', 'twelve hour', 'one day'),loc='lower right')
# plt.show()
'''
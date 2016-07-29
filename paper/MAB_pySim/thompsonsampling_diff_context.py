# /usr/bin/env python
# -*- encoding:utf-8 -*
'''
modify: add payment code, at one time, there will generate several tasks
'''
import random
from task import Task
import math
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
from data_operation import contactdict
from workerNode import Worker
import copy

##bid is random generate

def getTaskValue(duration):
	if duration < 6 * 3600:
		value = round(duration / 1000.0, 3)
	elif duration < 3600 * 7:
		value = 9.7
	else:
		value = round(8 - duration / 1000.0, 3)

	return value


def getContextArray(task, workers, d, flag):
	'''
	 Get the requester's all worker's context, return an array
	'''
	context_list = list()

	# the worker's buffer size is full then it can't accomplish more tasks
	for worker in workers:
		if worker.buffersize==0:
			workers.remove(worker)

	for worker in workers:
		temp = worker.getContext(task.value)
		if temp[1] > task.value:
			continue
		# temp.append(task.duration)
		temp.append(task.value)
		if flag:
			context_list.append(temp)
		else:
			context_list.append(temp[0:(d+1)])
	return np.array(context_list)

def generate_true_mu(k, num_samples):
	mu_true = []

	for i in range(num_samples):
		mean = np.random.randint(-100, 100, size=k)
		variance = np.random.random(size=k)
		mu_true.append(np.random.normal(mean, variance))

	return np.array(mu_true)

def generateTask(generate_rate):
	taskslist = []  # 初步假设在某一个时刻只产生一个task
	tid = 1
	for t in range(starttime, Time, generate_rate):
		# node = random.randint(1,100)
		duration = random.randint(1800, 36000)  # half hour--ten hour
		value = getTaskValue(duration)

		task = Task(tid, duration, value, random.randint(1, 3))
		taskslist.append(task)
		tid += 1
	print 'tid', tid
	return taskslist


# thompson sampling
def thompsonTaskAllocation(tasklist, flag, d, B, f, mu_param, index):
	'''
	:param rid: requester's id
	:param dim: the dimension of context
	:param tasksdict: tasks
	:return:  cumulate regret
	'''
	win_worker = dict()
	#worker_bid = np.zeros((len(workers),len(tasklist))) # all worker bid for all task

	payment = dict()  # worker:payment
	for task in tasklist:
	#while len(tasklist) != 0:
		reward = 0
		bid = []
		worker_bid = dict()
		context_array = getContextArray(task, workers, d, flag)
		# wid_list = context_array.T[:1]
		if flag:
			context = np.array((context_array.T[2:]).T, dtype=float)
		else:
			context = np.array((context_array.T[1:]).T, dtype=float)

		if len(context) == 0:
			break
		#i=0
		#bid = []
		for c in context_array:
			# worker_bid[i][task_index] = c[1]  # worker's bid
			worker_bid[(c[0],task)] = float(c[1]) # (wid,task): bid
			#bid.append(c[1])
			## i += 1
		bid = np.array(worker_bid.values())  #bid is random generate
		# select a worker who will accomplish the task success.
		while reward == 0:
			B_inv = linalg.inv(B) # inverse
			temp = v*v*B_inv

			deviation = np.sqrt(temp.sum(1))
			mu_sample = np.random.normal(mu_param, deviation)

			# 对每一个context 计算 b*mu
			if flag==True:
				selecteIndex = np.argmax(np.dot(context, mu_sample)/bid)
			else:
				selecteIndex = np.argmax(np.dot(context, mu_sample))


			reward = workers[selecteIndex].executeTask(task.tasktype)
			# reward_list[index]=reward

			B += np.dot(context[selecteIndex].T, context[selecteIndex])
			f += context[selecteIndex]*reward
			mu_param = np.dot(B_inv.T, f)
		if flag:
			column = 5
		else:
			column = d-1
		mu_sample = np.array(mu_sample)
		if d==1 and flag == False:
			regret[index, column] = abs(np.max(np.dot(context, mu_true[index])) - np.dot(context[selecteIndex], mu_sample))
		elif flag==True:
			regret[index, column] = abs(np.max(np.dot(context, mu_true[index, :].T)) - np.dot(context[selecteIndex, :], mu_sample))
		else:
			regret[index, column] = abs(np.max(np.dot(context, mu_true[index, :].T)) - np.dot(context[selecteIndex, :], mu_sample))

		#regret[index] = abs(np.max(np.dot(context, mu_true[index,:].T)) - np.dot(context[selecteIndex,:], mu_sample))

		wid=str(context_array[selecteIndex][0])
		workers.remove(Worker(wid))  # one worker just can acccomplish one task
		win_worker[Worker(wid)] = task

		### give the payment for the worker

		worker_bid_sorted = sorted(worker_bid.items(), key=lambda x:x[1])
		win_worker_bid = worker_bid[(wid, task)]
		for item in worker_bid_sorted:
			if(item[1] > win_worker_bid):
				payment[wid] = item[1]
				break

		if payment.has_key(wid)==False:
			payment[wid] = win_worker_bid


		u =task.value-payment[wid]
		utility[index, column] = u
		index += 1
	return B, f, mu_param, index


# cum_regret = np.cumsum(regret)
# cum_reward = np.cumsum(reward_list)


dim = [1,2,3,4,5]  # dimension of features
# num_experiments = 100

# 6个月的时刻
starttime = 24485668
# 总的时间
Time = 35242046  # diff 10756378  约为4个月

task_generate_rate = 3600 * 3  # 1800 #
tasksdict = dict()  # time:[task list]

# generate task
tid = 0  # count the amount of the task
#index = 0  # the index of task

for t in range(starttime, Time, task_generate_rate):
	# node = random.randint(1,100)
	task_number = random.randint(1, 10)

	tasklist = []
	for i in range(task_number):
		duration = random.randint(1800, 3600)  # half hour--ten hour
		value = getTaskValue(duration)
		task = Task(tid, duration, value, random.randint(1, 3))
		tasklist.append(task)
		tid += 1
	tasksdict[t] = tasklist

print 'task amount:', tid
rid = '68'

# buffersize, probability, centralize, bid, task_value
regret = np.zeros((tid, 6))
utility = np.zeros((tid, 6))

## #thompson sampling parameter##
for d in dim:
	print 'd is', d
	workers = []
	mu_true = generate_true_mu(d, tid)

	B = np.identity(d)
	mu_param = np.zeros(d)
	f = np.zeros(d)
	R = 0.1
	epsilon = 1.0 / math.log(tid)
	delta = 0.2

	v = round(R * math.pow((24.0 / epsilon) * d * math.log(1.0 / delta), 1.0 / 2), 3)

	index_1 = 0
	for time in range(starttime, Time):
		if tasksdict.has_key(time):
			# thompson sampling
			for wid in contactdict[rid]:
				workers.append(Worker(wid))
			B, f, mu_param, index_1 = thompsonTaskAllocation(tasksdict[time],False, d, B, f, mu_param, index_1)
			workers = []

print 'context not contain bid'
workers = []
mu_true = generate_true_mu(4, tid)

B = np.identity(4)
mu_param = np.zeros(4)
f = np.zeros(4)
R = 0.1
epsilon = 1.0 / math.log(tid)
delta = 0.2

v = round(R * math.pow((24.0 / epsilon) * 4 * math.log(1.0 / delta), 1.0 / 2), 3)

index_2 = 0
for time in range(starttime, Time):
	if tasksdict.has_key(time):
		# thompson sampling
		for wid in contactdict[rid]:
			workers.append(Worker(wid))
		B, f, mu_param, index_2 = thompsonTaskAllocation(tasksdict[time],True, 4, B, f, mu_param, index_2)
		workers = []

print 'thompson sampling is over.'

print 'algorithm run over, start plotting.'


regret[:, 0] = np.cumsum(regret[:, 0])
regret[:, 1] = np.cumsum(regret[:, 1])
regret[:, 2] = np.cumsum(regret[:, 2])
regret[:, 3] = np.cumsum(regret[:, 3])
regret[:, 4] = np.cumsum(regret[:, 4])
regret[:, 5] = np.cumsum(regret[:, 5])

plt.semilogy(regret, linewidth=2)
#plt.plot(regret, linewidth=2)
#plt.title('Cumulative Regret of Different Context',fontsize=18)
plt.xlabel('the Number of Tasks',fontsize=18)
plt.ylabel('Cumulative Regret',fontsize=18)
plt.legend(('TS-TA-1', 'TS-TA-2', 'TS-TA-3', 'TS-TA-4','TS-TA-5','TS-TA-6'), loc='lower right')
plt.grid(True)
plt.show()
# plt.savefig('./fig_mu_true/'+diff_rate[i]+'.eps',format='eps')
# plt.savefig('./fig_mu_true/'+diff_rate[i]+'.png',format='png')

utility[:,0] = np.cumsum(utility[:,0])
utility[:,1] = np.cumsum(utility[:,1])
utility[:,2] = np.cumsum(utility[:,2])
utility[:,3] = np.cumsum(utility[:,3])
utility[:,4] = np.cumsum(utility[:,4])
utility[:,5] = np.cumsum(utility[:,5])

plt.plot(utility, linewidth=2)
#plt.title('Requester Cumulative Utility of Different Context',fontsize=18)
plt.xlabel('the Number of Tasks',fontsize=18)
plt.ylabel('Cumulative Utility',fontsize=18)
plt.legend(('TS-TA-1', 'TS-TA-2', 'TS-TA-3', 'TS-TA-4','TS-TA-5','TS-TA-6'), loc='lower right')
plt.grid(True)
plt.show()

# f=open('./diff_cum_regret.txt', 'w')
# for i in range(6):
# 	for j in range(tid):
# 		f.write(''+utility[j,i])
# 	f.write('\n')
# f.close()
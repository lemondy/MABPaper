#/usr/bin/env python
#-*- encoding:utf-8 -*
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

## bid is use round(random.uniform(value/4.0, value*1.50), 3)

def getTaskValue(duration):
	if duration < 6 * 3600:
		value = round(duration / 2300.0, 3)
	elif duration < 3600 * 7:
		value = 9.7
	else:
		value = round(0.8 - duration / 2300.0, 3)

	return value if value > 0 else 0.1

def getContextArray(task, worker_list):
	'''
	 Get the requester's all worker's context, return an array
	'''
	context_list = list()

	for worker in workers:
		temp = worker.getContext(task.value)
		if temp[1]>task.value:
			continue
		temp.append(task.value)
		context_list.append(temp)
	return np.array(context_list)

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


# epsilon-greddy
def epsilon_greedy(tasklist, workers,index , column, epsilon=0.01):
	for task in tasklist:
		selectedIndex = -1
		reward = 0
		bid = []
		for worker in workers:
			bid.append(worker.getBid(task.value))
		while reward == 0:              # make sure the selected one finish the task success
			totals =  observed_data.sum(1)
			successes = observed_data[:,0]
			estimated_means = successes / totals
			best_mean = np.argmax(estimated_means)

			if np.random.rand()<epsilon:
				selectedIndex = np.random.randint(0,len(workers))
				while selectedIndex==best_mean:
					selectedIndex = np.random.randint(0,len(workers))
			else:     # actually, if the epsilon is bigger, the harder it can choose the best mean one.
				selectedIndex = best_mean
			reward = workers[selectedIndex].executeTask(1)
			if reward == 1:
				observed_data[selectedIndex,0] += 1
			else:
				observed_data[selectedIndex,1] += 1
		u = task.value - bid[selectedIndex]
		utility[index,column] = u
		win_worker_bid = bid[selectedIndex]
		payment = win_worker_bid
		bid = sorted(bid)
		for item in bid:
			if item > payment:
				payment = item
				break
		social_welfare[index, column] = u + (payment - win_worker_bid)

		# if utility[index,column] <0:
		# 	print 'epsilon-greedy:value-payment',str(utility[index,column])
		index += 1
	return index

# ucb
def ucb(tasklist, workers, index):
	for task in tasklist:
		reward = 0
		selectedIndex = -1
		bid = []
		for worker in workers:
			bid.append(worker.getBid(task.value))
		while reward == 0:
			t = float(observed_data.sum())  # total number of rounds so far
			totals = observed_data.sum(1)
			successes = observed_data[:,0]
			estimated_means = successes/totals
			estimated_variances = estimated_means - estimated_means**2
			UCB = estimated_means + np.sqrt(np.minimum(estimated_variances + np.sqrt(2*np.log(t)/totals), 0.25 ) * np.log(t)/totals )
			selectedIndex = np.argmax(UCB)
			reward = workers[selectedIndex].executeTask(1)
			if reward == 1:
				observed_data[selectedIndex, 0] += 1
			else:
				observed_data[selectedIndex, 1] += 1
		u = task.value-bid[selectedIndex]
		utility[index, 2] = u
		payment = bid[selectedIndex]
		win_worker_bid = bid[selectedIndex]
		bid = sorted(bid)
		for item in bid:
			if item>payment:
				payment = item
				break
		social_welfare[index, 2] = u + (payment-win_worker_bid)
		index += 1
	return index

def randomSelect(tasklist, workers, index):
	selectedIndex = -1

	for task in tasklist:
		reward = 0
		bid = []
		for worker in workers:
			bid.append(worker.getBid(task.value))
		while reward == 0:
			selectedIndex = np.random.randint(0, len(workers))
			reward = workers[selectedIndex].executeTask(1)
		u = (task.value - bid[selectedIndex])
		utility[index, 3] = u
		win_worker_bid = bid[selectedIndex]
		bid = sorted(bid)
		payment = win_worker_bid
		for item in bid:
			if item > win_worker_bid:
				payment = item
				break
		social_welfare[index, 3] = u + (payment - win_worker_bid)
		index += 1

	return index

# reference from the paper: truthful incentive mechanisms for crowdsourcing
# def crowdsourcingSelect(tasklist, workers, index_3):
# 	#workers_copy = copy.copy(workers)
# 	worker_bid = dict()
# 	for task in tasklist:
# 		for worker in workers:
# 			worker_bid[(worker, task)] = worker.getBid(task.value)
# 	#workers_bid_copy = copy.copy(worker_bid)
# 	worker_bid_sorted = sorted(worker_bid.items(),key=lambda x:x[1]) # after sort, it return a list
#
# 	win_worker_dict = dict()
# 	win_worker = []
# 	bid_sorted_copy = copy.copy(worker_bid_sorted)
# 	while len(worker_bid_sorted) != 0:
# 		key =  worker_bid_sorted[0][0]
# 		value = worker_bid_sorted[0][1]
# 		win_worker_dict[key] = value
# 		worker = key[0]
# 		task = key[1]
# 		win_worker.append(worker)
# 		items = filter(lambda x:x[0][0]==worker, worker_bid_sorted)
# 		for item in items:
# 			#del(worker_bid_sorted[item])
# 			worker_bid_sorted.remove(item)
#
# 		items = filter(lambda x:x[0][1]==task, worker_bid_sorted)
# 		for item in items:
# 			#del(worker_bid_sorted[item])
# 			worker_bid_sorted.remove(item)
#
# 	payments = dict() # (worker, task): payment
# 	for key in win_worker_dict.keys():
# 		task = key[1]
# 		items = filter(lambda x:x[0][1]==task, bid_sorted_copy)
# 		for item in items:
# 			if item[1] > win_worker_dict[key]:
# 				payments[key] = item[1]
# 				utility[index_3, 3] = task.value-item[1]
#
# 				break
# 		if payments.has_key(key)==False:
# 			payments[key] = win_worker_dict[key]
# 		utility[index_3,3] = task.value-win_worker_dict[key]
# 		index_3 += 1
#
#
# 	return index_3

# thompson sampling
def thompsonTaskAllocation(tasklist, B, f, mu_param, index):
	'''
	:param rid: requester's id
	:param dim: the dimension of context
	:param tasksdict: tasks
	:return:  cumulate regret
	'''
	win_worker = dict()

	payment = dict()  # worker:payment
	for task in tasklist:
	#while len(tasklist) != 0:
		reward = 0
		bid = None
		worker_bid = dict()  # (wid,task): bid
		context_array = getContextArray(task, workers)
		context = np.array((context_array.T[1:]).T, dtype=float)

		if len(context) == 0:
			break

		for c in context_array:
			# worker_bid[i][task_index] = c[1]  # worker's bid
			worker_bid[(c[0],task)] = float(c[1])


		bid = np.array(worker_bid.values())

		# select a worker who will accomplish the task success.
		while reward == 0:
			B_inv = linalg.inv(B) # inverse
			temp = v*v*B_inv

			deviation = np.sqrt(temp.sum(1))
			mu_sample = np.random.normal(mu_param, deviation)

			# 对每一个context 计算 b*mu
			selecteIndex = np.argmax(np.dot(context, mu_sample)/bid)

			reward = workers[selecteIndex].executeTask(task.tasktype)


			B += np.dot(context[selecteIndex].T, context[selecteIndex])
			f += context[selecteIndex]*reward
			mu_param = np.dot(B_inv.T, f)
		regret.append(abs(np.max(np.dot(context, mu_true[index, :].T)) - np.dot(context[selecteIndex, :], mu_sample)))

		wid=str(context_array[selecteIndex][0]) # worker id
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
		utility[index,4] = u
		worker_utility[wid].append(payment[wid]-win_worker_bid)
		social_welfare[index,4] = u + (payment[wid]-win_worker_bid)
		#print 'pwid',payment[wid]
		index += 1

	return B, f, mu_param, index

dim=5  # dimension of features

# 6个月的时刻
starttime = 24485668
# 总的时间
Time = 35242046        # diff 10756378  约为4个月

task_generate_rate = 3600*3 # three hours
tasksdict = dict()  # time:[task list]

#generate task
tid = 0  # count the amount of the task

for t in range(starttime, Time, task_generate_rate):
	#node = random.randint(1,100)
	task_number = random.randint(1,10)
	
	tasklist = []
	for i in range(task_number):
		duration = random.randint(1800, 36000)           #half hour--ten hour
		value = getTaskValue(duration)
		task = Task(tid, duration, value, random.randint(1,3))
		tasklist.append(task)
		tid += 1
	tasksdict[t] = tasklist

print 'tid',tid
rid = '68'  # requester id
workers = []

## #thompson sampling parameter##

regret = list()

mu_true = generate_true_mu(dim, tid)

B = np.identity(dim)
mu_param = np.zeros(dim)
f = np.zeros(dim)
R = 0.1
epsilon = 1.0/math.log(tid)
delta = 0.2

v = round(R*math.pow((24.0/epsilon)*dim*math.log(1.0/delta),1.0/2), 3)

# requester's utility
utility = np.zeros((tid, 5))
social_welfare = np.zeros((tid,5)) # Ur+Uw, the sum of the utility

worker_utility = dict()
for wid in contactdict[rid]:
	worker_utility[wid] = list()

## epsilon-greedy parameter
observed_data = np.zeros((len(contactdict[rid]),2))
observed_data[:,0] = 1.0  # sucesses
observed_data[:,1] = 1.0  # failure

print 'task amount:',tid

index_0=0; index_1=0; index_2=0; index_3=0; index_4=0

print 'epsilon-greedy start executing...'
for time in range(starttime, Time):
	if tasksdict.has_key(time):
		for wid in contactdict[rid]:
			workers.append(Worker(wid))
		index_0 = epsilon_greedy(tasksdict[time], workers, index_0, 0)
		workers = []
print 'epsilon-greedy is over.'

observed_data = np.zeros((len(contactdict[rid]),2))
observed_data[:,0] = 1.0  # sucesses
observed_data[:,1] = 1.0  # failure
print 'epsilon-greedy(1/sqrt(t)) start executing...'
for time in range(starttime, Time):
	if tasksdict.has_key(time):
		for wid in contactdict[rid]:
			workers.append(Worker(wid))
		if index_1<1000:
			index_1 = epsilon_greedy(tasksdict[time], workers, index_1, 1)
		else:
			index_1 = epsilon_greedy(tasksdict[time], workers, index_1,1, 1.0/math.sqrt(index_1))
		workers = []
print 'epsilon-greedy(1/sqrt(t)) is over.'

observed_data = np.zeros((len(contactdict[rid]),2))
observed_data[:,0] = 1.0  # sucesses
observed_data[:,1] = 1.0  # failure

workers = []
print 'UCB start executing...'
for time in range(starttime, Time):
	if tasksdict.has_key(time):
		for wid in contactdict[rid]:
			workers.append(Worker(wid))
		index_2 = ucb(tasksdict[time], workers, index_2)
		workers = []
print 'UCB is over.'

workers = []
print 'random select start executing...'
for time in range(starttime, Time):
	if tasksdict.has_key(time):
		for wid in contactdict[rid]:
			workers.append(Worker(wid))
		index_3 = randomSelect(tasksdict[time], workers, index_3)
		workers = []
print 'random select is over.'

workers = []

print 'thompson sampling start executing...'
for time in range(starttime, Time):
	if tasksdict.has_key(time):
		#thompson sampling
		for wid in contactdict[rid]:
			workers.append(Worker(wid))
		B, f, mu_param, index_4 = thompsonTaskAllocation(tasksdict[time], B, f, mu_param, index_4)
		workers= []
		print 'index',index_4

print 'thompson sampling is over.'

worker_utility_avg = dict()
for wid in worker_utility.keys():
	worker_utility_avg[wid] = sum(worker_utility[wid])

print 'algorithm run over, start plotting.'

utility[:,0] = np.cumsum(utility[:,0])
utility[:,1] = np.cumsum(utility[:,1])
utility[:,2] = np.cumsum(utility[:,2])
utility[:,3] = np.cumsum(utility[:,3])
utility[:,4] = np.cumsum(utility[:,4])
# #cum_utility_sum = np.cumsum(utility)
#plt.semilogy(cum_utility_sum, linewidth=2)
plt.plot(utility,  linewidth=2)
#plt.title('Requester Cumulative Utility',fontsize=18)
plt.xlabel('The Number of Tasks',fontsize=18)
plt.ylabel('Cumulative Utility',fontsize=18)
plt.legend(('epsilon-greedy','epoch-greedy','ucb','random select','TS-TA'),loc='upper left')
plt.grid(True)
plt.show()

social_welfare[:,0] = np.cumsum(social_welfare[:,0])
social_welfare[:,1] = np.cumsum(social_welfare[:,1])
social_welfare[:,2] = np.cumsum(social_welfare[:,2])
social_welfare[:,3] = np.cumsum(social_welfare[:,3])
social_welfare[:,4] = np.cumsum(social_welfare[:,4])

plt.plot(social_welfare,  linewidth=2)
#plt.title('Cumulative Social Welfare',fontsize=18)
plt.xlabel('The Number of Tasks',fontsize=18)
plt.ylabel('Cumulative Social Welfare',fontsize=18)
plt.legend(('epsilon-greedy','epoch-greedy','ucb','random select','TS-TA'),loc='upper left')
plt.grid(True)
plt.show()

## save the the plot pic to a file
# plt.savefig('./fig_mu_true/'+diff_rate[i]+'.eps',format='eps')
# plt.savefig('./fig_mu_true/'+diff_rate[i]+'.png',format='png')

# bar chart. Only plot the worker's utility in TS-TA

worker_utility_sorted = sorted(worker_utility_avg.items(), key=lambda x:int(x[0]))
groups = len(worker_utility_sorted)
xvalue = map(lambda x:x[0], worker_utility_sorted)
yvalue = map(lambda x:x[1], worker_utility_sorted)

x=[]
for i in range(len(xvalue)):
	if i%2==1:
		x.append(xvalue[i])
	else:
		x.append("")

fig, ax = plt.subplots()
index = np.arange(groups)
bar_width= 0.5
# opacity = 0.4 # 透明度

plt.bar(index, yvalue, bar_width)
#plt.title('Worker Utility Sum in TS-TA',fontsize=18)
plt.xlabel('Worker ID', fontsize=18)
plt.ylabel('The Sum of Utility', fontsize=18)
plt.xticks(index, x)

plt.legend()
plt.tight_layout()
plt.show()
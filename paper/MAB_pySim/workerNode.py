#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-06-30 19:59:38
# @Author  : lemondy
# @email   : lemondy9@gmail.com
# @Link    : https://github.com/lemondy
###########################

import random
import json

contactnode = open(r'.\data\contactnode.json','r')
data = contactnode.read()
contact = json.JSONDecoder().decode(data)

class Worker(object):
	'''
	when the requester want to choose a worker to finish a task,
	he must
	'''
	probability = 0.0
	bid = 0.0
	init_buffersize = 50
	centrality = 0
	numberOfTask1=0
	numberOfTask2=0
	numberOfTask3=0
	numberOfTask = 0

	threshold = round(random.random(), 2)  # every worker have different probability to accomplish the task. This doesn't works, because every worker have the same threshold.

	def __init__(self, wid, buffersize=50, numberOfTask=0):
		self.wid = wid  #worker id
		self.buffersize = buffersize
		self.numberOfTask = numberOfTask
		#self.threshold = round(random.random(), 2)

	def calprobability(self):
		return round(random.random(), 3)  #keep three number

	def getBid(self,value):
		return round(random.random(), 3)  # return bid
		#return round(random.uniform(value/4.0, value*1.50), 3)

	def getBuffersize(self):
		return self.buffersize/(self.init_buffersize*1.0)

	def calCentrality(self):
		return len(contact[self.wid])/(len(contact) * 1.0)   # statatistic how many neighbor who has. percentage

	def getContext(self, value):
		'''
		return current worker's context, which represent by a list
		'''
		probability = self.calprobability()

		buffersize = self.getBuffersize()
		centrality = self.calCentrality()
		bid = self.getBid(value)
		context = list()
		context.append(self.wid)
		context.append(bid)
		context.append(probability)
		context.append(buffersize)
		context.append(centrality)

		return context

	def executeTask(self,taskType):

		# the worker maybe not accomplish the task
		if random.random() > self.threshold:
			if taskType == 1:
				self.numberOfTask1 += 1
			elif taskType == 2:
				self.numberOfTask2 += 1
			elif taskType == 3:
				self.numberOfTask3 += 1
			else:
				raise ValueError("task type must be one of 1,2,3")
			self.buffersize -= 1
			self.numberOfTask += 1
			return 1
		else:
			return 0
	def __eq__(self, other):
		if self.wid == other.wid:
			return True
		else:
			return False
	def __str__(self):
		return 'wid'+str(self.wid)

'''
worker = Worker('1')
context = worker.getContext()
print 'context....'
print context.probability,context.bid, context.centrality,context.buffersize
print worker.executeTask(1)
print 'task:',worker.numberOfTask1,worker.numberOfTask2,worker.numberOfTask3
'''

# w1 = Worker('1',20)
# w2 = Worker('2',30)
# w3 = Worker('3',40)
# print w1.threshold, w2.threshold,w3.threshold
# #
# li = list()
# li.append(w1)
# li.append(w2)
# li.append(w3)
#
# li.remove(Worker('1'))
# for i in li:
# 	print i
# print len(li)
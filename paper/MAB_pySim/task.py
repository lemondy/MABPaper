#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-05-13 11:24:11
# @Author  : lemondy (zhangyunruanjian@163.com)
# @Link    : https://github.com/lemondy/MABPaper
# @Version : $Id$

#import os

class Task(object):
	'define the structure of task' #task类型

	def __init__(self, tid, duration, value, tasktype=1):
		self.tid = tid
		self.duration = duration
		self.value = value
		self.tasktype = tasktype
	def __eq__(self, other):     #重定义这个方法可以实现对象之间的比较
		if self.tid == other.tid:
			return True
		else:
			return False

	def __str__(self):
		return 'tid:'+str(self.tid)



# t1=Task(1,2,3,4)
# t2=Task(1,2,3,4)
# t3=Task(2,3,3,3)
#
# t = list()
# t.append(t1)
# t.append(t2)
# t.append(t3)
#
# t.remove(Task(1,2,3,4))
# print len(t)
#
# print 't1==t2',t1==t2
# print 't2==t3', t2==t3
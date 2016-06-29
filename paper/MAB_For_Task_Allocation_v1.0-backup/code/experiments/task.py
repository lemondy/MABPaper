#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-05-13 11:24:11
# @Author  : lemondy (zhangyunruanjian@163.com)
# @Link    : https://github.com/lemondy/MABPaper
# @Version : $Id$

#import os

class Task(object):
	'define the structure of task' #task类型

	def __init__(self, tid, duration, value):
		self.id = tid
		self.duration = duration
		self.value = value



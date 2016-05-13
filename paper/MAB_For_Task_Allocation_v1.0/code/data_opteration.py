#/usr/bin/env python
#-*- encoding: utf-8 -*-
import datetime
import json

datafile = open('./data/callspan.csv','r')
contactStat = open('./data/contactstat.txt','w')

#######################统计相遇记录##############################

dataset = datafile.readlines()
datafile.close()

firstline = dataset[0]
items = firstline.split(',')

timeformat = '%Y-%m-%d %H:%M:%S'
#starttime
begintime = datetime.datetime.strptime("2004-03-23 14:25:32",timeformat)

for line in dataset:
	items =line.split(',')
	starttime = datetime.datetime.strptime(items[1].strip('\"'),timeformat)
	##日期之间相隔的秒数
	difftime = int((starttime-begintime).total_seconds())
	source = items[2].strip('\"')
	des = items[5].strip('\n').strip('\"')
	#if des != '-1':
	#-1表示陌生人，不在联系人中
	if difftime<0 or des == '-1':
		continue
	#100个手机设备
	if int(source)>100 or int(des) > 100:
		continue
	if source == des:
		continue
	contactStat.write(source+','+des+','+str(difftime)+'\n')

contactStat.close()

############### 统计每个节点相遇过的节点（联系人）##############

contactfile = open('./data/contactstat.txt','r')
contactnode = open('./data/contactnode.json','w')

contactdict = {}

contacthis = contactfile.readlines()

#六个月的数据构建联系人
endtime = datetime.datetime.strptime("2005-01-01 00:00:00",timeformat)

durtime = int((endtime-begintime).total_seconds())
print 'six month durtime',durtime

for line in contacthis:
	items = line.split(',')
	##source, des
	if int(items[2]) <= durtime:
		if contactdict.has_key(items[0]):
			meetnodes = contactdict[items[0]]
			if items[1] not in meetnodes:
					contactdict[items[0]].append(items[1])		
		else:
			contactdict[items[0]] = list()
			contactdict[items[0]].append(items[1])
		##des, source
		if contactdict.has_key(items[1]):
			if items[0] not in contactdict[items[1]]:
				contactdict[items[1]].append(items[0])
		else:
			contactdict[items[1]]=list()
			contactdict[items[1]].append(items[0])

print 'len nodes',len(contactdict)
contactnode.write(json.dumps(contactdict))
contactnode.close()

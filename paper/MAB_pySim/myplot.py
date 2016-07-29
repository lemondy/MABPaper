#/usr/bin/env python
#-*- encoding:utf-8 -*

import matplotlib.pyplot as plt
import math
import numpy as np

# function: d*lnT*sqrt(TlnTln(1/delta))

x=range(1,1000)
y = []


for i in x:
	# math.log(i)

	y.append(5.0*math.log(i)*math.sqrt(i*math.log(i)*math.log(10)) + (math.log(i)*2)/i)
# y= np.cumsum(y)
plt.plot(y)
plt.show()
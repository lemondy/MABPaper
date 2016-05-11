#/usr/bin/env python
#-*- encoding: utf-8 -*-
import pylab as pl
import numpy as np
from scipy import stats

#scipy.stats.binom 为二项分布，用它来模拟n次抛硬币实验，出现k次正面的概率
n = 10
k = np.arange(n+1)   #k 的取值范围为[0,n+1]
pcoin = stats.binom.pmf(k,n,0.5)  # 正面的概率为0.5,二项分布

pl.stem(k, pcoin, basefmt="k-")
pl.margins(0.1)

#投掷6次骰子
n = 6
k = np.arange(n+1)
pdice = stats.binom.pmf(k, n, 1.0/6)

pl.stem(k, pdice, basefmt="k-")
pl.margins(0.1)

#beta 分布式一个连续分布，由于它描述概率 p 的分布，因此它取值范围为 0 到 1.
#beta 分布中的参数 alpha 和 beta 取值，初始都为0，试验中成功一次 alpha 就加1
# 失败一次 beta 就加1.
n = 10
k = 5
p = np.linspace(0, 1, 100)
pbeta = stats.beta.pdf(p, k+1, n-k+1)
plot(p, pbeta, label="k=5", lw=2)

k=4
pbeta = stats.beta.pdf(p, k+1, n-k+1)
plot(p, pbeta, label="k=4", lw=2)
xlabel("$p$")
legend(loc="best")





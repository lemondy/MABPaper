from beta_bandit import *

from numpy import *
from scipy.stats import beta
import random

theta = (0.25, 0.35)

def is_conversion(title):
    if random.random() < theta[title]:
        return True
    else:
        return False

conversions = [0,0]
trials = [0,0]

N = 100000
trials = zeros(shape=(N,2))
successes = zeros(shape=(N,2))

bb = BetaBandit()
for i in range(N):
    choice = bb.get_recommendation()
    trials[choice] = trials[choice]+1
    conv = is_conversion(choice)
    bb.add_result(choice, conv)

    trials[i] = bb.trials
    successes[i] = bb.successes

from pylab import *
subplot(211)
n = arange(N)+1
loglog(n, trials[:,0], label="title 0")
loglog(n, trials[:,1], label="title 1")
legend()
xlabel("Number of trials")
ylabel("Number of trials/title")

subplot(212)
semilogx(n, (successes[:,0]+successes[:,1])/n, label="CTR")
semilogx(n, zeros(shape=(N,))+0.35, label="Best CTR")
semilogx(n, zeros(shape=(N,))+0.30, label="Random chance CTR")
semilogx(n, zeros(shape=(N,))+0.25, label="Worst CTR")
axis([0,N,0.15,0.45])
xlabel("Number of trials")
ylabel("CTR")


legend()
show()
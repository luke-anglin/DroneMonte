import math

import numpy as np
from numpy.random import normal
from matplotlib import pyplot

pyplot.style.use('seaborn')
tau = 57
alpha = 1 / tau

# Parameters
mean = (1 / alpha) * (np.sqrt(np.pi / 2))
stdev = np.sqrt((4 - np.pi) / (2 * pow(alpha, 2)))
# runs = [10, 30, 50,100, 250, 500, 1000]
runs = [5, 10, 15, 30]
np.random.seed(1000)


def calc_means(samp_sizes, estimates):
    # 10, 30, . .. 1000
    # Initialize an empty dictionary which maps sample sizes to arrays of sample means
    sample_means = {new_list: [] for new_list in samp_sizes}
    for ss in samp_sizes:
        # 1, 2, 3 . . . 110
        for estimate in range(estimates):
            sample = np.random.normal(loc=mean, scale=stdev, size=(ss))
            sample_means[ss].append(round(sample.mean(), 4))
    return sample_means


# Holds our dict containing sample means
data = calc_means(runs, 550)
# print(data)
means = {'5': 0, '10': 0, '15': 0, '30': 0}
mu = []

# 2.1 calculate mean
def mean_est(default=data):
    for vals in data.keys():
        # print(vals)
        count = 0
        for item in data[vals]:
            count += item
        mu.append((count / 550))
    return mu


# 2.1 calculate variance
def var_est():
    itr = 0
    variance = []
    for vals in data.keys():
        var = 0
        for item in data[vals]:
            var += math.pow((item - mu[itr]), 2)
        itr += 1
        variance.append(var / 550)
    return variance


# 2.2 Calculate zn(k)
d = {}
def znk():
    itr = 0
    z = 0

    v = var_est()
    for vals in data.keys():
        # count =0
        for item in data[vals]:
            z = (item - mu[itr]) / (pow(v[itr], 0.5))
            d[item] = z
    return d  # ret a dict


# 2.3 estimate prob of 7 events
z_vals = {-1.4: 0.0808, -1: 0.1587, -0.5: 0.3085, 0: 0.5, 0.5: 0.6915, 1: 0.8413, 1.4: 0.9192}
est =[]
def estimate():
    for nums in z_vals.values(): #compare with z scores
        count = 0
        for vals in d.values():
            if vals < nums:
                count+=1
        est.append(count/550)
    return  est #

#2.4
zj =[-1.4,-1,-0.5,0,0.5,1,1.4]
mad_lst=[]
md_lst1=[]
def mad():
    for i in range(0,len(est)):
        for j in range(0,len(zj)):
            mad_lst.append(abs(est[i]-zj[j]))
        md_lst1.append(max(mad_lst))
        # md_lst1.clear()
        mad_lst.clear()
    return md_lst1

print("mean: " , mean_est())
print("variance: " , var_est())
print("zn(k) values: ", znk())
print("prob of 7 events", estimate())
print("mad",mad())


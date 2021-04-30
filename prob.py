
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
# print("data",data)

# print(len(data[5]))
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
x = {
    5: [],
    10: [],
    15: [],
    30: []
}
def znk():
    for item in data[5]:
        z =(item - mu[0])/ pow(mu[0],0.5)
        x[5].append(z)
    for item in data[10]:
        z =(item - mu[1])/ pow(mu[1],0.5)
        x[10].append(z)
    for item in data[15]:
        z =(item - mu[2])/ pow(mu[2],0.5)
        x[15].append(z)
    for item in data[30]:
        z =(item - mu[2])/ pow(mu[2],0.5)
        x[30].append(z)
    return x  # ret a dict


# 2.3 estimate prob of 7 events
z_vals = {-1.4: 0.0808, -1: 0.1587, -0.5: 0.3085, 0: 0.5, 0.5: 0.6915, 1: 0.8413, 1.4: 0.9192}
est = []
e_dict = {
    5: [],
    10: [],
    15: [],
    30: []
}
def estimate():
    x1,x2,x3,x4,x5,x6,x7=0,0,0,0,0,0,0
    for item in x[5]:
        if item < 0.0808:
            x1 += 1
        if item < 0.1587:
            x2 += 1
        if item < 0.3085:
            x3 += 1
        if item < 0.5:
            x4 += 1
        if item < 0.6915:
            x5 += 1
        if item < 0.8413:
            x6 += 1
        if item <  0.9192:
            x7 += 1
    e_dict[5].append(x1/ 550)
    e_dict[5].append(x2 / 550)
    e_dict[5].append(x3 / 550)
    e_dict[5].append(x4 / 550)
    e_dict[5].append(x5 / 550)
    e_dict[5].append(x6 / 550)
    e_dict[5].append(x7 / 550)

    y1, y2, y3, y4, y5, y6, y7 = 0, 0, 0, 0, 0, 0, 0
    for item in x[10]:
        if item < 0.0808:
            y1 += 1
        if item < 0.1587:
            y2 += 1
        if item < 0.3085:
            y3 += 1
        if item < 0.5:
            y4 += 1
        if item < 0.6915:
            y5 += 1
        if item < 0.8413:
            y6 += 1
        if item < 0.9192:
            y7 += 1
    e_dict[10].append(y1 / 550)
    e_dict[10].append(y2 / 550)
    e_dict[10].append(y3 / 550)
    e_dict[10].append(y4 / 550)
    e_dict[10].append(y5 / 550)
    e_dict[10].append(y6 / 550)
    e_dict[10].append(y7 / 550)

    z1, z2, z3, z4, z5, z6, z7 = 0, 0, 0, 0, 0, 0, 0
    for item in x[15]:
        if item < 0.0808:
            z1 += 1
        if item < 0.1587:
            z2 += 1
        if item < 0.3085:
            z3 += 1
        if item < 0.5:
            z4 += 1
        if item < 0.6915:
            z5 += 1
        if item < 0.8413:
            z6 += 1
        if item < 0.9192:
            z7 += 1
    e_dict[15].append(z1 / 550)
    e_dict[15].append(z2 / 550)
    e_dict[15].append(z3 / 550)
    e_dict[15].append(z4 / 550)
    e_dict[15].append(z5 / 550)
    e_dict[15].append(z6 / 550)
    e_dict[15].append(z7 / 550)

    a1, a2, a3, a4, a5, a6, a7 = 0, 0, 0, 0, 0, 0, 0
    for item in x[30]:
        if item < 0.0808:
            a1 += 1
        if item < 0.1587:
            a2 += 1
        if item < 0.3085:
            a3 += 1
        if item < 0.5:
            a4 += 1
        if item < 0.6915:
            a5 += 1
        if item < 0.8413:
            a6 += 1
        if item < 0.9192:
            a7 += 1
    e_dict[30].append(a1 / 550)
    e_dict[30].append(a2 / 550)
    e_dict[30].append(a3 / 550)
    e_dict[30].append(a4 / 550)
    e_dict[30].append(a5 / 550)
    e_dict[30].append(a6 / 550)
    e_dict[30].append(a7 / 550)

    # print(e_dict)
    return e_dict
#2.4 goodness of fit of cdf
zj =[-1.4,-1,-0.5,0,0.5,1,1.4]
mad_lst=[]
md_lst1={
    5:[], 10:[], 15:[], 30:[]
}
def mad():
    for item in e_dict[5]: #item is prob of event for key 5
        for items in zj:
            # print(items)
            mad_lst.append(abs(item-items))
        md_lst1[5].append(max(mad_lst))
        mad_lst.clear()
    for item in e_dict[10]: #item is prob of event for key 10
        for items in zj:
            mad_lst.append(abs(item-items))
        md_lst1[10].append(max(mad_lst))
        mad_lst.clear()
    for item in e_dict[15]: #item is prob of event for key 15
        for items in zj:
            mad_lst.append(abs(item-items))
        md_lst1[15].append(max(mad_lst))
        mad_lst.clear()
    for item in e_dict[30]: #item is prob of event for key 30
        for items in zj:
            mad_lst.append(abs(item-items))
        md_lst1[30].append(max(mad_lst))
        mad_lst.clear()
    return md_lst1


print("mean: " , mean_est())
print("variance: " , var_est())
print("zn(k) values: ", znk())
print("prob of 7 events", estimate())
print("mad",mad())

#graphs key: mad: green estimates: red normal cdf: blue
import numpy as np
import matplotlib.pyplot as plt

# No of data points used
N = 10000
data = np.random.randn(N)
x = np.sort(data)
y = np.arange(N) / float(N)
z = np
plt.xlabel('x-axis')
plt.ylabel('y-axis')

plt.figure(1)
plt.title('CDF n=5')
plt.xlim(-2.5, 2.5)
plt.plot(x,y, marker='o')
plt.plot([-1.4,-1,-0.5,0,0.5,1,1.4],[0.5, 0.5109090909090909, 0.5472727272727272, 0.5945454545454546, 0.6290909090909091, 0.66, 0.6709090909090909], 'ro')
plt.plot([-1.4,-1,-0.5,0,0.5,1,1.4],[1.9, 1.9109090909090907, 1.9472727272727273, 1.9945454545454546, 2.0290909090909093, 2.06, 2.070909090909091], 'ro',color='green')

plt.figure(2)
plt.title('CDF n=10')
plt.xlim(-2.5, 2.5)
plt.plot(x,y, marker='o')
plt.plot([-1.4,-1,-0.5,0,0.5,1,1.4],[0.5181818181818182, 0.5454545454545454, 0.58, 0.6418181818181818, 0.68, 0.7181818181818181, 0.7272727272727273], 'ro')
plt.plot([-1.4,-1,-0.5,0,0.5,1,1.4],[1.918181818181818, 1.9454545454545453, 1.98, 2.041818181818182, 2.08, 2.118181818181818, 2.127272727272727], 'ro',color='green')

plt.figure(3)
plt.title('CDF n=15')
plt.xlim(-2.5, 2.5)
plt.plot(x,y, marker='o')
plt.plot([-1.4,-1,-0.5,0,0.5,1,1.4],[0.52, 0.5436363636363636, 0.5963636363636363, 0.6636363636363637, 0.7436363636363637, 0.7854545454545454, 0.8127272727272727], 'ro')
plt.plot([-1.4,-1,-0.5,0,0.5,1,1.4],[1.92, 1.9436363636363634, 1.9963636363636361, 2.0636363636363635, 2.1436363636363636, 2.1854545454545455, 2.212727272727273], 'ro',color='green')

plt.figure(4)
plt.title('CDF n=30')
plt.xlim(-2.5, 2.5)
plt.plot(x,y, marker='o')
plt.plot([-1.4,-1,-0.5,0,0.5,1,1.4],[0.5236363636363637, 0.5672727272727273, 0.6563636363636364, 0.7472727272727273, 0.8018181818181818, 0.8327272727272728, 0.8727272727272727], 'ro')
plt.plot([-1.4,-1,-0.5,0,0.5,1,1.4],[1.9236363636363636, 1.9672727272727273, 2.056363636363636, 2.1472727272727274, 2.2018181818181817, 2.2327272727272724, 2.2727272727272725], 'ro',color='green')
plt.show()

# plt.plot([-1.4,-1,-0.5,0,0.5,1,1.4],[0.5, 0.5109090909090909, 0.5472727272727272, 0.5945454545454546, 0.6290909090909091, 0.66, 0.6709090909090909], 'ro')
# plt.plot([-1.4,-1,-0.5,0,0.5,1,1.4],[0.5181818181818182, 0.5454545454545454, 0.58, 0.6418181818181818, 0.68, 0.7181818181818181, 0.7272727272727273], 'ro')
# plt.plot([-1.4,-1,-0.5,0,0.5,1,1.4],[0.52, 0.5436363636363636, 0.5963636363636363, 0.6636363636363637, 0.7436363636363637, 0.7854545454545454, 0.8127272727272727], 'ro')
# plt.plot([-1.4,-1,-0.5,0,0.5,1,1.4],[0.5236363636363637, 0.5672727272727273, 0.6563636363636364, 0.7472727272727273, 0.8018181818181818, 0.8327272727272728, 0.8727272727272727], 'ro')

# 5:  [(-1.4,0.5), (-1,0.5109090909090909), (-0.5,0.5472727272727272), (0,0.5945454545454546), (0.5,0.6290909090909091), (1,0.66), (1.4,0.6709090909090909)]
# 10: [(-1.4,0.5181818181818182), (-1,0.5454545454545454), (-0.5,0.58), (0,0.6418181818181818), (0.5,0.68), (1,0.7181818181818181), (1.4,0.7272727272727273)]
# 15: [(-1.4,0.52), (-1,0.5436363636363636), (-0.5,0.5963636363636363), (0,0.6636363636363637), (0.5,0.7436363636363637), (1,0.7854545454545454), (1.4,0.8127272727272727)]
# 30: [(-1.4,0.5236363636363637), (-1,0.5672727272727273), (-0.5,0.6563636363636364), (0,0.7472727272727273), (0.5,0.8018181818181818), (1,0.8327272727272728), (1.4,0.8727272727272727)]

#table
from astropy.table import QTable, Table, Column
import numpy as np
chart = np.array([
["n=5",1.9, 1.9109090909090907, 1.9472727272727273, 1.9945454545454546, 2.0290909090909093, 2.06, 2.070909090909091],
["n=10",1.918181818181818, 1.9454545454545453, 1.98, 2.041818181818182, 2.08, 2.118181818181818, 2.127272727272727],
["n=15",1.92, 1.9436363636363634, 1.9963636363636361, 2.0636363636363635, 2.1436363636363636, 2.1854545454545455, 2.212727272727273],
["n=30",1.9236363636363636, 1.9672727272727273, 2.056363636363636, 2.1472727272727274, 2.2018181818181817, 2.2327272727272724, 2.2727272727272725]
])
t=Table(chart, names=("-",-1.4,-1,-0.5,0,0.5,1,1.4))
t.pprint(max_lines=1000, max_width=1000)

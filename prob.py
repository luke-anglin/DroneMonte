
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
        if item < -1.4:
            x1 += 1
        if item < -1:
            x2 += 1
        if item < -0.5:
            x3 += 1
        if item < 0:
            x4 += 1
        if item < 0.5:
            x5 += 1
        if item < 1:
            x6 += 1
        if item <  1.4:
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
        if item < -1.4:
            y1 += 1
        if item < -1:
            y2 += 1
        if item < -0.5:
            y3 += 1
        if item < 0:
            y4 += 1
        if item < 0.5:
            y5 += 1
        if item < 1:
            y6 += 1
        if item < 1.4:
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
        if item < -1.4:
            z1 += 1
        if item < -1:
            z2 += 1
        if item < -0.5:
            z3 += 1
        if item < 0:
            z4 += 1
        if item < 0.5:
            z5 += 1
        if item < 1:
            z6 += 1
        if item < 1.4:
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
        if item < -1.4:
            a1 += 1
        if item < -1:
            a2 += 1
        if item < -0.5:
            a3 += 1
        if item < 0:
            a4 += 1
        if item < 0.5:
            a5 += 1
        if item < 1:
            a6 += 1
        if item < 1.4:
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
# zj =[0.0808,0.1587,0.3085, 0.5, 0.6915, 0.8413, 0.9192]
mad_lst=[]

fv_est = [0.24727272727272728, 0.3018181818181818, 0.39636363636363636, 0.4909090909090909, 0.5945454545454546,
          0.6836363636363636, 0.76]
zj = [0.0808, 0.1587, 0.3085, 0.5, 0.6915, 0.8413, 0.9192]
mad_5 = -10000
mad_10 = -10000
mad_15 = -10000
mad_30 = -10000
md_lst1={
    5:[], 10:[], 15:[], 30:[]
}
all_mads =[]
def mad():
    fv_est = [0.24727272727272728, 0.3018181818181818, 0.39636363636363636, 0.4909090909090909, 0.5945454545454546, 0.6836363636363636, 0.76]
    ten_est= [0.15636363636363637, 0.24181818181818182, 0.37636363636363634, 0.5054545454545455, 0.6418181818181818, 0.7545454545454545, 0.8454545454545455]
    fift_est=[0.09454545454545454, 0.1690909090909091, 0.3236363636363636, 0.49272727272727274, 0.6636363636363637, 0.8381818181818181, 0.9145454545454546]
    thirty_est=[0.045454545454545456, 0.10727272727272727, 0.2781818181818182, 0.4781818181818182, 0.7472727272727273, 0.889090909090909, 0.9654545454545455]
    mad5 = -1
    mad10 = -1
    mad15 = -1
    mad30 = -1
    for i in range(7):
        x = abs(fv_est[i] - zj[i])
        md_lst1[5].append(x)
        if x > mad5:
            mad5 = x
    for i in range(7):
        y = abs(ten_est[i] - zj[i])
        md_lst1[10].append(y)
        if y > mad10:
            mad10 = y
    for i in range(7):
        z = abs(fift_est[i] - zj[i])
        md_lst1[15].append(z)
        if z > mad15:
            mad15 = z
    for i in range(7):
        a = abs(thirty_est[i] - zj[i])
        md_lst1[30].append(a)
        if a > mad30:
            mad30 = a
    all_mads.append(mad5)#,mad10,mad15,mad30)
    all_mads.append(mad10)#,mad10,mad15,mad30)
    all_mads.append(mad15)#,mad10,mad15,mad30)
    all_mads.append(mad30)#,mad10,mad15,mad30)

    return md_lst1,all_mads

# print(all_mads)

print("mean: " , mean_est())
print("variance: " , var_est())
print("zn(k) values: ", znk())
print("prob of 7 events", estimate())
print("mad",mad())

import numpy as np
import matplotlib.pyplot as plt
#graphs key: mad: green estimates: red normal cdf: blue

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
plt.plot([-1.4,-1,-0.5,0,0.5,1,1.4],[0.24727272727272728, 0.3018181818181818, 0.39636363636363636, 0.4909090909090909, 0.5945454545454546, 0.6836363636363636, 0.76], 'ro')
# plt.plot([-1.4,-1,-0.5,0,0.5,1,1.4],[0.16647272727272727, 0.1431181818181818, 0.08786363636363637, 0.009090909090909094, 0.0969545454545454, 0.15766363636363645, 0.1592], 'ro')
# plt.plot([-1.4],[0.16647272727272727], 'ro',color='green')
plt.axvline(x=-1.4)

# plt.show()#
plt.figure(2)
plt.title('CDF n=10')
plt.xlim(-2.5, 2.5)
plt.plot(x,y, marker='o')
plt.plot([-1.4,-1,-0.5,0,0.5,1,1.4],[0.15636363636363637, 0.24181818181818182, 0.37636363636363634, 0.5054545454545455, 0.6418181818181818, 0.7545454545454545, 0.8454545454545455], 'ro')
# plt.plot([1],[0.08675454545454553], 'ro',color='green')
plt.axvline(x=-1)
#
plt.figure(3)
plt.title('CDF n=15')
plt.xlim(-2.5, 2.5)
plt.plot(x,y, marker='o')
plt.plot([-1.4,-1,-0.5,0,0.5,1,1.4],[0.09454545454545454, 0.1690909090909091, 0.3236363636363636, 0.49272727272727274, 0.6636363636363637, 0.8381818181818181, 0.9145454545454546], 'ro')
# plt.plot([0.5],[0.027863636363636313], 'ro',color='green')
plt.axvline(x=0.5)
#
plt.figure(4)
plt.title('CDF n=30')
plt.xlim(-2.5, 2.5)
plt.plot(x,y, marker='o')
plt.plot([-1.4,-1,-0.5,0,0.5,1,1.4],[0.045454545454545456, 0.10727272727272727, 0.2781818181818182, 0.4781818181818182, 0.7472727272727273, 0.889090909090909, 0.9654545454545455], 'ro')
# plt.plot([0.5],[0.05577272727272731], 'ro',color='green')
plt.axvline(x=0.5)
plt.show()

#table
from astropy.table import QTable, Table, Column
import numpy as np
chart = np.array([
["n=5",0.16647272727272727, 0.1431181818181818, 0.08786363636363637, 0.009090909090909094, 0.0969545454545454, 0.15766363636363645, 0.1592,0.16647272727272727],
["n=10",0.07556363636363637, 0.08311818181818181, 0.06786363636363635, 0.00545454545454549, 0.04968181818181816, 0.08675454545454553, 0.07374545454545456,0.08675454545454553],
["n=15",0.013745454545454547, 0.01039090909090909, 0.015136363636363614, 0.007272727272727264, 0.027863636363636313, 0.0031181818181819088, 0.004654545454545467,0.027863636363636313],
["n=30",0.03534545454545454, 0.05142727272727274, 0.0303181818181818, 0.021818181818181792, 0.05577272727272731, 0.047790909090908995, 0.04625454545454544,0.05577272727272731]
])
t=Table(chart, names=("-",-1.4,-1,-0.5,0,0.5,1,1.4,"MADS"))
t.pprint(max_lines=1000, max_width=1000)

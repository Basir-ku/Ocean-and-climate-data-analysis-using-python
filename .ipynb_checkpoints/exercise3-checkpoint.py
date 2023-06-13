#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 20:22:22 2022

@author: mohammadbasiruddin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 15:25:57 2022

@author: mohammadbasiruddin
"""
#exercise 4
#import necessary libraries

import numpy as np
import matplotlib.pyplot as plt
import time
import math
import warnings

#returns sum using cycle
def myfunc(n):
    s=0
    for k in range(n):
        k=k+1
        s=s=(-1)**(k+1)/(k*np.exp(k*np.log(2)))+s
        return s
    
print(myfunc(2000))

#retruns sum using vector 
def myfunc2(n):
    s=0
    n=np.arange(1,n+1,1)
    s=np.sum((-1)**(n+1)/(n*np.exp(n*np.log(2))))
    return s
print(myfunc2(20000))

#last part
stime1 = time.time()
etime1 = time.time()-stime1
time_start = time.time()
etime2 = time.time()-stime1
print("time for using cycle is %s" % etime1)
print(" time for using vector is %s" % etime2)
m=np.zeros(20)
for i in range(20):
    m[i]=myfunc2(i+1)
    
x=np.linspace(1,20,20)
s=plt.plot(x,m,'b')
l=plt.axhline(y=math.log(1.5), color='r', linestyle='--')
plt.xlabel("n")
plt.ylabel("s")
plt.xticks(np.arange(0,20))
plt.legend([s,l],['s','y=log(1.5)'])
plt.show()
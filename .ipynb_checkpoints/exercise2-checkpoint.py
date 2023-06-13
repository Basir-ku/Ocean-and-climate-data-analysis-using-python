#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 00:32:25 2022

@author: mohammadbasiruddin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 14:54:32 2022

@author: mohammadbasiruddin
"""

#import libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
fig.suptitle('a sphere surface', fontsize=15)
ax = Axes3D(fig)

#constant and varibale
pi=np.pi
t=np.linspace(0, 2*pi,1000)
s=np.linspace(0,pi,1000)
t,s = np.meshgrid(t,s)

#equations
x=2 * np.cos(t) * np.cos(s)
y=2 * np.cos(t) * np.sin(s)
z = 2 * np.sin(t)
ax.plot_surface(x,y,z)
plt.show()

#plt.savefig('a sphere surface.png')
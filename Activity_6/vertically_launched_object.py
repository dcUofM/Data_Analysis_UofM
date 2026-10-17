#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 11:35:37 2026

@author: dcastro

Activity #6 Vertically Launched Object 

y = v0t - 0.5gt^2

"""

import numpy as np
from matplotlib import pyplot as plt


v0 = 5 ## inicial velocity 5m/s
g = 9.81 ## m/s^2 gravity
t0 = 0 ## start time
tf = 2  ## end time

ns = 2000 ## number of steps

t = np.linspace(t0,tf,ns)
t_max = 0.0
y_max = -1000

y=[]

for i in range(ns):
    t_current = t[i]
    y_current = v0 * t_current -0.5*g*(t_current **2)
    y.append(y_current)
    
    if y_current > y_max:
        y_max = y_current
        t_max  = t_current
    
print (f"Max. height {y_max} for loop")


i = 0
t_current = t[i]
y_current = v0 * t_current -0.5*g*(t_current **2)
while i < ns -1 :
    t_next = t[i+1]
    y_next = v0 * t_next -0.5*g*(t_next **2)
    
    if y_next < y_current:
        print (f"Max. height {y_max} while loop")
        break
    i=i+1
        

y_max=round(y_max,2)
t_max=round(t_max,3)

print (f"After the objet is lauched the max. Heigh is {y_max} m at {t_max} s")

## derivative:
dydt=v0-g*t

y0 = dydt[0]
yf = dydt[-1]
t0 = t[0]
tf = t[-1]

m = (yf - y0)/(tf - t0)


print (f"Inicial vel. {y0} m/s")
print (f"Final vel. {yf} m/s")
print (f"Initial time: {t0} s")
print (f"Final time: {tf} s")
print (f"Slope: {m} m/s**2 (aceleration)")

plt.figure(figsize=(10,5))
plt.suptitle("Vertically Launch Object")


plt.subplot(121)
plt.plot(t,dydt,color='red', label=f"(dv/dt) m = {m} m/s**2")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.grid(True,linestyle="--",alpha=0.3)
plt.legend()

plt.subplot(122)    
plt.plot(t,y,color='black',label=f"v(t) = {v0}*t - 0.5*{g}t**2")
plt.scatter(t_max,y_max,marker="*",color='green',label=f"Max. Height {y_max} m at {t_max} s")
plt.legend()
plt.grid(True,linestyle="--",alpha=0.3)
plt.xlabel("Time (s)")
plt.ylabel("Height (m)")

plt.tight_layout()
#plt.show()
plt.savefig(fname="verticallY-lauched_object.pdf")





import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Global param
Q = 0.5
R = 1

# initial values
r_l = 0.38
q_l = 0.6

r_r = 0.7
q_r = 0.3

# curves in (rho, q)- plane
q1 = lambda r: ((q_l - Q)/r_l)*r + Q   # 1st family wave
q2 = lambda r: (q_r/r_r)*((R- r_r)/(R-r))*r  #2nd family wave

r1s = np.linspace(0.01,1, 1000)
r2s = np.linspace(0.01,0.9, 1000)

plt.plot(r1s, q1(r1s), '-')
plt.plot(r2s, q2(r2s), '-')

#def findIntersection(fun1,fun2,x0):
# return fsolve(lambda x : fun1(x) - fun2(x),x0)

#result = findIntersection(q1, q2, 0)

idx = np.argwhere(np.diff(np.sign(q1(r1s) - q2(r2s)))).flatten()
print(idx)
r_m = r2s[idx]
print(r_m)
q_m = q1(r_m)
print(q_m)

plt.plot(r_m, q_m, 'ro')
plt.show()


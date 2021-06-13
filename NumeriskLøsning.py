import numpy as np
from matplotlib import pyplot as plt

##
##  HYPERBOLIC TRAFFIC MODEL WITH
##        PHASE TRANSITIONS
##

# FREE PHASE                    # CONGESTED PHASE
# r_t + (rv)_x = 0              # r_t + (rv)_x = 0
# v = vf = (1- r/R)V            # q_t + ((q-Q)v) = 0
# lambda = V(1- 2r/R)           # v = vc = (1- r/R)q/r
                                # max lambda = v_c

# parameters:
R = 1   # max density
V = 5   # max speed
Q = 0.5 # wide jam param.

xs = np.linspace(-1,1,1000)
ts = np.linspace(0,5, 500)

# Eigenvalues
lambda1 = lambda r,q: (2/R - 1/r)*(Q-q) - Q/R
lambda2 = lambda r,q: q/r*(1-r/R)


# Rarefaction
#w_r = lambda r, q, xi: 0.5*(r*(Q-q*r*xi - 2*r*(2*Q+1/R)))/(R*(Q-q)+ (q*r - 2*Q*r - Q/2) - 2*r/R)
#w_q = lambda r, q, xi: R/2*( xi + (Q-q)/r) - Q*(1+1/R)

w_r = lambda r, q, xi: 0.5*(r*(Q-q*r*xi - 2*r*(2*Q+1/R)))/(R*(Q-q)+ (q*r - 2*Q*r - Q/2) - 2*r/R)
w_q = lambda r, q, xi: R/2*( xi + (Q-q)/r) - Q*(1+1/R)

# Initial values, u_l, u_r
r_l = 0.6
q_l = 0.6

r_r = 0.2
q_r = 0.7

def plotInitialValues():
    sol_r = np.zeros(len(xs))
    sol_q = np.zeros(len(xs))

    t = 0
    i = 0
    for x in xs:
        if x < lambda1(r_l, q_l)*t:
            sol_r[i] = r_l
            sol_q[i] = q_l
            i += 1
        elif x > lambda2(r_r, q_r)*t:
            sol_r[i] = r_r
            sol_q[i] = q_r
            i += 1

    plt.plot(sol_r, 'r', label=r"$\rho$")
    plt.plot(sol_q, 'b', label=r"$q$")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.show()


def findMiddleState(plot):
    r1s = np.linspace(0.01, 1, 1000)
    r2s = np.linspace(0.01, 0.9, 1000)
    # wave curves in rho, q
    q1 = lambda r: ((q_l - Q) / r_l) * r + Q  # 1st family wave
    q2 = lambda r: (q_r / r_r) * ((R - r_r) / (R - r)) * r  # 2nd family wave
    # finding smallest point
    idx = np.argwhere(np.diff(np.sign(q1(r1s) - q2(r2s)))).flatten()
    r_m = r2s[idx]
    q_m = q1(r_m)
    if (plot):
        plt.plot(r1s, q1(r1s), '-')
        plt.plot(r2s, q2(r2s), '-')
        plt.plot(r_l, q_l, 'ro')
        plt.annotate(r"$u_l$", (r_l, q_l+.05))
        plt.plot(r_m, q_m, 'ro')
        plt.annotate(r"$u_m$", (r_m, q_m+0.05))
        plt.plot(r_r, q_r, 'ro')
        plt.annotate(r"$u_r$", (r_r, q_r+0.05))
        plt.xlabel(r"$\rho$")
        plt.ylabel("q")
        plt.title(r"$u(\rho, q)$")
        plt.show()
    return r_m, q_m

# Middle state u_m:
r_m, q_m = findMiddleState(plot=False)

def chooseEntropySol():
    if( q_l > Q):
        if(r_l > r_m):
            return "Rarefaction"
        else:
            return "Shock"
    else:
        if (r_l > r_m):
            return "Shock"
        else:
            return "Rarefaction"

##            ANALYTICAL SOLUTION
##
##            u_l    for  x < lambda1 (u_l) t
## u(x,t) =    w     for  lambda1(u_l)t < x < lambda1 (u_m)t
##            u_m    for  lambda1(u_m)t < x < lambda2(u_r)t
##            u_r    for  x > lambda2(u_r)t
##
##

def plotAnalyticalSolution():
    sol_r = np.zeros(len(xs))
    sol_q = np.zeros(len(xs))

    t = 1.5
    i = 0
    for x in xs:
        if x < lambda1(r_l, q_l) * t:
            sol_r[i] = r_l
            sol_q[i] = q_l
            i += 1
        elif (lambda1(r_l, q_l) * t < x) and (x < lambda1(r_m, q_m) * t):
            sol_r[i] = w_r(r_l, q_l, x/t)
            sol_q[i] = w_q(r_l, q_l, x/t)
            i += 1
        elif (lambda1(r_m, q_m) * t < x) and (x < lambda2(r_r, q_r) * t):
            sol_r[i] = r_m
            sol_q[i] = q_m
            i += 1
        elif x > lambda2(r_r, q_r) * t:
            sol_r[i] = r_r
            sol_q[i] = q_r
            i += 1

    plt.plot(sol_r, 'r', label=r"$\rho$")
    #plt.plot(sol_q, 'b', label=r"$q$")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.show()

print("r_l =", r_l)
print("r_m =", r_m[0])
print("r_r =", r_r)
print(chooseEntropySol())
plotAnalyticalSolution()
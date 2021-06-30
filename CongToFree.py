import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({
    "font.size":20})

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
V_c = 0.5
Q = 0.5 # wide jam param.
vf = lambda rho: (1-rho/R)*V
vc = lambda rho,q: (1-rho/R)*q/rho

xs = np.linspace(-1,1,1000)
ts = np.linspace(0,1, 1000)

# Eigenvalues
lambda1 = lambda r,q: (2/R - 1/r)*(Q-q) - Q/R
lambda2 = lambda r,q: q/r*(1-r/R)

# Speed of phase boundary
Lambda = lambda r_l,r_r, q_l, q_r: (r_l*vf(r_l) - r_r*vc(r_r, q_r))/(r_l - r_r)

# Rarefaction
#w_r = lambda r, q, xi: (R*(r*xi - q + 2*Q) -Q*r)/(2*(q- 2*Q + R/r*(Q-q)))
w_r = lambda r, q, xi: ( R*(r*xi - q +Q) + Q*r)/(2*(Q-q))
w_q = lambda r, q, xi: R/2*( (q-Q)/r - xi ) +Q/2

# Rarefaction wave in the free phase
w = lambda xi: R/2*(1-xi/V)
# Speed of rarefaction wave in the free phase
lmda = lambda rho: V*(1-2*rho/R)

# Initial values, u_l, u_r
def initialValuesfor1a():
    # 1-rarefaction followed by phase transition
    r_l = 0.79
    q_l = 0.9

    r_m1 = 0.24
    q_m1 = 0.62

    r_m2 = 0.24
    q_m2 = 0.62

    r_r = 0.11
    q_r = 0.56

def initialValuesfor6():
    #  phase transition followed by a shock in the free phase
    r_l = 0.79
    q_l = 0.2

    r_m1 = 0.09
    q_m1 = 0.46

    r_m2 = 0.12
    q_m2 = 0.61

    r_r = 0.12
    q_r = 0.61

def initialValuesfor4():
    #  phase transition followed by a shock in the free phase
    #V_c = 0.3, V = 5
    r_l = 0.79
    q_l = 1.06

    r_m1 = 0.16
    q_m1 = 0.61

    r_m2 = 0.12
    q_m2 = 0.58

    r_r = 0.04
    q_r = 0.2


r_l = 0.79
q_l = 1.06

r_m1 = 0.25
q_m1 = 0.68

r_m2 = 0.12
q_m2 = 0.58

r_r = 0.04
q_r = 0.2

# Speed of shock in the free phase
s = (r_r*vf(r_r) - r_l*vf(r_l))/(r_r - r_l)


def plotInitialValues():
    sol_r = np.zeros(len(xs))
    sol_q = np.zeros(len(xs))

    i = 0
    for x in xs:
        if x < 0:
            sol_r[i] = r_l
            sol_q[i] = q_l
            i += 1
        elif x > 0:
            sol_r[i] = r_r
            sol_q[i] = q_r
            i += 1

    plt.plot(xs, sol_r, 'r', label=r"$\rho$", color = "teal")
    plt.plot(xs, sol_q, 'b', label=r"$q$", color = "goldenrod")
    #plt.title("Inital values "+ r"$u(x,0): $" + r"$u_l = {},{}, u_r = {},{} $".format(r_l,q_l, r_r, q_r))
    plt.title("Inital values " + r"$u(x,0): $")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.show()




##            ANALYTICAL SOLUTION, u_l in Free and u_r in Cong
##
##            u_l    for  x < lambda1 (u_l) t
## u(x,t) =   w1     for  lambda1(u_l)t < x < lambda1 (u_m1)t
##            u_m1   for  lambda1(u_m1)t < x < Lambda(u_m2)t
##            u_m2   for  Lambda(u_m2)t < x < lambda(u_r)t
##            u_r    for  x > lambda(u_r)t
##
##

def plotAnalyticalSolutionCongToFree(t, plot):
    sol_r = np.zeros(len(xs))
    sol_q = np.zeros(len(xs))

    i = 0
    for x in xs:
        if x < lambda1(r_l, q_l) * t:
            sol_r[i] = r_l
            sol_q[i] = q_l
            i += 1
        elif (lambda1(r_l, q_l) * t < x) and (x < lambda1(r_m1, q_m1) * t):
            sol_r[i] = w_r(r_l, q_l, x/t)
            sol_q[i] = w_q(r_l, q_l, x/t)
            i += 1
        elif (lambda1(r_m1, q_m1) * t < x) and (x < Lambda(r_m1, r_m2, q_m1, q_m2) * t):
            sol_r[i] = r_m1
            sol_q[i] = q_m1
            i += 1
        elif (Lambda(r_m1, r_m2, q_m1, q_m2) * t < x) and (x < lmda(r_m2) * t):
            sol_r[i] = r_m2
            sol_q[i] = q_m2
            i += 1
        elif (lmda(r_m2) * t) and (x < lmda(r_r) * t):
            sol_r[i] = w(x/t)
            sol_q[i] = w(x/t)*V
            i += 1
        elif x > lmda(r_r) * t:
            sol_r[i] = r_r
            sol_q[i] = q_r
            i += 1
    if plot == True:
        plt.plot(xs, sol_r,  label=r"$\rho$", color = "teal")
        plt.plot(xs, sol_q,  label=r"$q$", color = "goldenrod")
        plt.title(r"$ u(x, t = {} ) $".format(t))
        plt.xlabel("x")
        plt.ylabel("u")
        plt.legend()
        plt.show()
    return sol_r, sol_q

def plot_xtSolCongToFree():
    sol_xtRho = np.zeros((len(ts), len(xs)))
    sol_xtQ = np.zeros((len(ts), len(xs)))
    j = 0
    X, T = np.meshgrid(xs, ts)

    for t in ts:
        sol_xtRho[j], sol_xtQ[j] = plotAnalyticalSolutionCongToFree(t, False)
        j += 1

    plt.contourf(X, T, sol_xtRho, cmap =  "GnBu", levels=20)
    plt.title(r"$ \rho(x,t) $")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("t")
    plt.show()

    plt.contourf(X, T, sol_xtQ, cmap = "YlOrBr", levels=20 )
    plt.title(r"$ q(x,t) $")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("t")
    plt.show()

print("r_l =", r_l)
#print("r_m =", r_m[0])
print("r_r =", r_r)
#print(chooseEntropySol())


plotInitialValues()
plotAnalyticalSolutionCongToFree(0.2, True)
plotAnalyticalSolutionCongToFree(0.4, True)
plotAnalyticalSolutionCongToFree(0.7, True)
plot_xtSolCongToFree()

print("Speed of phase boundary from u_l to u_m1: " , Lambda(r_l, r_m1, q_l, q_m1))
print("Speed of 1-wave at u_m2: " , lambda1(r_m2, q_m2))
print("Speed of 2-contact at u_r: ", lambda2(r_r, q_r))
print("Speed of rarefaction : " , lmda(r_r))
print("Speed of shock : " , s)



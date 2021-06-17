import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({
    "font.size":15})

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

# Initial values, u_l, u_r
r_l = 0.08
q_l = 0.38

r_m1 = 0.25
q_m1 = 0.38

r_m2 = 0.68
q_m2 = 0.16

r_r = 0.9
q_r = 0.65


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
##            u_l    for  x < Lambda (u_l) t
## u(x,t) =   u_m1   for  Lambda(u_l)t < x < lambda1 (u_m1)t
##            w1     for  lambda1(u_m1)t < x < lambda1(u_m2)t
##            u_m2   for  lambda1(u_m2)t < x < lambda2(u_r)t
##            u_r    for  x > lambda2(u_r)t
##
##

def plotAnalyticalSolutionFreeToCong(t, plot):
    sol_r = np.zeros(len(xs))
    sol_q = np.zeros(len(xs))

    i = 0
    for x in xs:
        if x < Lambda(r_l, r_m1, q_l, q_m1) * t:
            sol_r[i] = r_l
            sol_q[i] = q_l
            i += 1
        elif (Lambda(r_l, r_m1, q_l, q_m1) * t < x) and (x < lambda1(r_m1, q_m1) * t):
            sol_r[i] = r_m1
            sol_q[i] = q_m1
            i += 1
        elif (lambda1(r_m1, q_m1) * t < x) and (x < lambda1(r_m2, q_m2) * t):
            sol_r[i] = w_r(r_m1, q_m1, x / t)
            sol_q[i] = w_q(r_m1, q_m1, x / t)
            i += 1
        elif (lambda1(r_m2, q_m2) * t < x) and (x < lambda2(r_r, q_r) * t):
            sol_r[i] = r_m2
            sol_q[i] = q_m2
            i += 1
        elif x > lambda2(r_r, q_r) * t:
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

def plot_xtSolFreeToCong():
    sol_xtRho = np.zeros((len(ts), len(xs)))
    sol_xtQ = np.zeros((len(ts), len(xs)))
    j = 0
    X, T = np.meshgrid(xs, ts)

    for t in ts:
        sol_xtRho[j], sol_xtQ[j] = plotAnalyticalSolutionFreeToCong(t, False)
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
plotAnalyticalSolutionFreeToCong(0.7, True)
plot_xtSolFreeToCong()




import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({
    "font.size":15})

# CONGESTED PHASE
# r_t + (rv)_x = 0
# q_t + ((q-Q)v) = 0
# v = vc = (1- r/R)q/r
# max lambda = v_c


# parameters:
R = 1   # max density
V = 5   # max speed
Q = 0.5 # wide jam param.

xs = np.linspace(-1,1,1000)
ts = np.linspace(0,1, 1000)

# Eigenvalues
lambda1 = lambda r,q: (2/R - 1/r)*(Q-q) - Q/R
lambda2 = lambda r,q: q/r*(1-r/R)


# Rarefaction
#w_r = lambda r, q, xi: (R*(r*xi - q + 2*Q) -Q*r)/(2*(q- 2*Q + R/r*(Q-q)))
w_r = lambda r, q, xi: ( R*(r*xi - q +Q) + Q*r)/(2*(Q-q))
w_q = lambda r, q, xi: R/2*( (q-Q)/r - xi ) +Q/2

# Initial values, u_l, u_r
r_l = 0.8
q_l = 1

r_r = 0.2
q_r = 0.4

r_m = 0.3
q_m = 0.69

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



def findMiddleState(plot):
    r1s = np.linspace(0.01, 1, 1000)
    r2s = np.linspace(0.01, 0.9, 1000)
    # wave curves in rho, q
    q1 = lambda r: ((q_l - Q) / r_l) * r + Q  # 1st family wave
    q2 = lambda r: (q_r / r_r) * ((R - r_r) / (R - r)) * r  # 2nd family wave
    # finding smallest point
    idx = np.argwhere(np.diff(np.sign(q1(r1s) - q2(r2s)))).flatten()
    print(np.sign(q1(r1s) - q2(r2s)))
    r_m = r2s[idx]
    q_m = q1(r_m)
    if (plot):
        plt.plot(r1s, q1(r1s), '-',  color = "teal")
        plt.plot(r2s, q2(r2s), '-', color = "goldenrod")
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

def findMiddleState2():
    a = (q_l - Q)/r_l
    print("a: ", a)
    b = (R-r_r)*q_r/r_r
    print("b: ", b)




#findMiddleState2()
# Middle state u_m:
#r_m, q_m = findMiddleState(plot=False)


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

def plotAnalyticalSolution(t, plot):
    sol_r = np.zeros(len(xs))
    sol_q = np.zeros(len(xs))

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
    if plot == True:
        plt.plot(xs, sol_r,  label=r"$\rho$", color = "teal")
        plt.plot(xs, sol_q,  label=r"$q$", color = "goldenrod")
        plt.title(r"$ u(x, t = {} ) $".format(t))
        plt.xlabel("x")
        plt.ylabel("u")
        plt.legend()
        plt.show()
    return sol_r, sol_q

def plot_xtSol():
    sol_xtRho = np.zeros((len(ts), len(xs)))
    sol_xtQ = np.zeros((len(ts), len(xs)))
    j = 0
    X, T = np.meshgrid(xs, ts)

    for t in ts:
        sol_xtRho[j], sol_xtQ[j] = plotAnalyticalSolution(t, False)
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
print(chooseEntropySol())


plotInitialValues()
plotAnalyticalSolution(0.4, True)
plot_xtSol()

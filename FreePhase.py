import numpy as np
from matplotlib import pyplot as plt

# FREE PHASE
# r_t + (rv)_x = 0
# v = vf = (1- r/R)V

# parameters:
R = 1   # max density
V = 5   # max speed
vf = lambda rho: (1-rho/R)*V

# initial condition
rho_l = 0.2
rho_r = 0.9

# Shock speed
s = (rho_r*vf(rho_r) - rho_l*vf(rho_l))/(rho_r - rho_l)

# Rarefaction wave
w = lambda xi: R/2*(1-xi/V)
# Speed of rarefaction wave
lmda = lambda rho: V*(1-2*rho/R)

xs = np.linspace(-1,1,1000)
ts = np.linspace(0,1, 1000)

def plotInitialValues():
    sol_rho = np.zeros(len(xs))
    i = 0
    for x in xs:
        if x < 0:
            sol_rho[i] = rho_l
            i += 1
        elif x > 0:
            sol_rho[i] = rho_r
            i += 1

    plt.plot(xs, sol_rho, 'r', label=r"$\rho$", color="teal")
    plt.title("Inital values " + r"$\rho_l = {}, \rho_r = {} $".format(rho_l, rho_r))
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.show()

def plotAnalyticalSolution(t, plot):
    sol_rho = np.zeros(len(xs))
    i = 0
    if ( rho_l > rho_r):  # Rarefaction solution
        for x in xs:
            if x < lmda(rho_l)*t:
                sol_rho[i] = rho_l
                i += 1
            elif (x > lmda(rho_l)*t) and (x < lmda(rho_r)*t):
                sol_rho[i] = w(x/t)
                i += 1
            elif x > lmda(rho_r)*t:
                sol_rho[i] = rho_r
                i += 1
    else:                 # Shock solution
       for x in xs:
           if x < s*t:
               sol_rho[i] = rho_l
               i += 1
           elif x > s*t:
               sol_rho[i] = rho_r
               i += 1

    if plot == True:
        plt.plot(xs, sol_rho, 'r', label=r"$\rho$",color="teal")
        plt.title(r"$ \rho(x, t = {} ) $".format(t))
        plt.xlabel("x")
        plt.ylabel("u")
        plt.legend()
        plt.show()
    return sol_rho

def plot_xtSol():
    sol_xt = np.zeros((len(ts), len(xs)))
    j = 0
    X, T = np.meshgrid(xs, ts)

    for t in ts:
        sol_xt[j] = plotAnalyticalSolution(t, False)
        j += 1

    #plt.imshow(sol_xt, extent=[-1, 1, 0, 1])
    plt.contourf(X, T, sol_xt, cmap= "GnBu")
    plt.title(r"$ \rho(x,t) $")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("t")
    #plt.legend()
    plt.show()


#plotInitialValues()
#plotAnalyticalSolution(0.5, True)
#plot_xtSol()


def plotRarefactionWave():

    plotInitialValues()
    plotAnalyticalSolution(0.2, True)
    plot_xtSol()

def plotShockWave():

    plotInitialValues()
    plotAnalyticalSolution(0.5, True)
    plot_xtSol()

plotShockWave()
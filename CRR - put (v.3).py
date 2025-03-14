import time
import numpy as np
import matplotlib.pyplot as plt

t0 = time.time()
def CRR_put(T, N, r, S, sigma, K):
    dt = 1/N
    u = np.exp( sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    R = np.exp(r*dt)
    p = (R - d) / (u - d)
    #print (u,d, R)

    X = np.array([max(0, K - (S * (d ** k * u ** (T * N-k)))) for k in range(T * N+1)])

    XX = X.copy()

    X = np.delete(X, -1)
    XX = np.delete(XX, 0)


    Y = np.array([max(0, K - (S * (d ** k * u ** (T * N-k)))) for k in range(T * N)])

    for i in range(T * N, 0, -1):

        ev = Y
        bdv = R ** -1 * (p * X + (1-p) * XX)
        val = np.maximum(bdv, ev)

        X_temp = X.copy()
        X = np.delete(val.copy(), -1)
        XX = np.delete(val.copy(), 0)
        Y = np.delete(X_temp,0)
    return(val[0])

#Altering the CRR function s.t we instead get the optimal stopping boundary
def CRR_put_bound(T, N, r, S, sigma, K):
    dt = 1/N
    u = np.exp( sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    R = np.exp(r*dt)
    p = (R - d) / (u - d)
    #print (u,d, R)

    X = np.array([max(0, K - (S * (d ** k * u ** (T * N-k)))) for k in range(T * N+1)])

    XX = X.copy()

    X = np.delete(X, -1)
    XX = np.delete(XX, 0)

    cont_val = []

    Y = np.array([max(0, K - (S * (d ** k * u ** (T * N-k)))) for k in range(T * N)])

    for i in range(T * N, 0, -1):

        ev = Y
        bdv = R ** -1 * (p * X + (1-p) * XX)

        if i < T * N and np.sum(ev > bdv)>0:
            cont_val_element = ev[ev < bdv]
            cont_val_element_filt = np.max(cont_val_element) if cont_val_element.size > 0 else K
        elif i == T * N:
            cont_val_element_filt = 0
        else:
            cont_val_element_filt = np.nan

        val = np.maximum(bdv, ev)

        X_temp = X.copy()
        X = np.delete(val.copy(), -1)
        XX = np.delete(val.copy(), 0)
        Y = np.delete(X_temp,0)

        cont_val.append(K - cont_val_element_filt)
    return(cont_val[::-2])

X_points = np.linspace(10, 10**4 , 1000)
Y_points = [CRR_put(1, int(i) , 0.06, 36, 0.2, 40) for i in X_points]
y_ref = CRR_put(1, 10**4 , 0.06, 36, 0.2, 40)

plt.figure(facecolor="#fafafa") #colors outside the box
plt.grid(alpha=0.5, linestyle = '--', color = 'black', zorder = 1)
ax = plt.gca()
ax.set_facecolor("#fafafa") #
ax.set_axisbelow(True)
plt.plot(X_points, Y_points, color = 'tan', linestyle ='-', linewidth = 4, label = 'Option price')
plt.axhline(y = y_ref, color='black', linestyle = '--', label='Option price for N=10^4', linewidth = 4)
plt.xlabel("Amount of timepoints", fontsize=26)  # Bigger x-axis label
plt.ylabel("Option price", fontsize=26)
plt.legend(fontsize = 26)
#plt.title('Convergence of option price in Binomial model', fontsize=36)
plt.ylim((4.4848,4.4872))
plt.xticks(fontsize=20)  # Enlarge tick labels on x-axis
plt.yticks(fontsize=20)

###############################################3

y1 = CRR_put_bound(1,10**4,0.06,36,0.2,40)
x1 = np.linspace(0,1,len(y1))

#Binomial tree
dt = 1/10
sigma = 0.2
r = 0.06
u = np.exp( sigma * np.sqrt(dt))
d = np.exp(-sigma * np.sqrt(dt))
R = np.exp(r*dt)
p = (R - d) / (u - d)

fig = plt.figure(figsize=[5, 5], facecolor="#fafafa")
plt.grid(alpha=0.5, linestyle='--', color='black', zorder=1)
ax = plt.gca()
ax.set_facecolor("#fafafa")
ax.set_axisbelow(True)

for i in range(10):
    x = [dt, 0, dt]
    for j in range(i):
        x.append(0)
        x.append(dt)
    x = np.array(x) + dt*i
    y = 36 * u**np.arange(-(i+1), i+2)
    plt.plot(x, y, 'bo-', color = 'tan', markersize=10, linewidth = 2)
    if i == 10-1 and j == i-1:
        plt.plot(x, y, 'bo-', color='tan', label='Stock price paths', markersize=10, linewidth = 2)

plt.plot(x1,y1, color = 'black', label = 'Optimal stopping boundary', linewidth=4)
#plt.title('Binomial model and optimal stopping boundary', fontsize=36)
plt.xlabel("Time", fontsize=26)  # Bigger x-axis label
plt.ylabel("Stock price", fontsize=26)
plt.legend(fontsize=26)
plt.xticks(fontsize=20)  # Enlarge tick labels on x-axis
plt.yticks(fontsize=20)
plt.tight_layout()
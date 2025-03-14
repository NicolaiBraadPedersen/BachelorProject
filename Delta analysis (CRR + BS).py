import numpy as np
from scipy import stats as stats
import matplotlib.pyplot as plt


def CRR_put_Greeks(T, N, r, S, sigma, K, t):
    dt = 1/N
    u = np.exp( sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    R = np.exp(r*dt)
    p = (R - d) / (u - d)
    #print (u, d, R)
    Timesteps = int((T-t)*N)

    X = np.array([max(0, K - (S * (d ** k * u ** (Timesteps-k)))) for k in range(Timesteps+1)])

    XX = X.copy()

    X = np.delete(X, -1)
    XX = np.delete(XX, 0)

    Y = np.array([max(0, K - (S * (d ** k * u ** (Timesteps-k)))) for k in range(Timesteps)])

    for i in range(Timesteps, 0, -1):

        if i == 1:
            delta = (X-XX)/(S*(u-d))

        ev = Y
        bdv = R ** -1 * (p * X + (1-p) * XX)
        val = np.maximum(bdv, ev)

        X_temp = X.copy()
        X = np.delete(val.copy(), -1)
        XX = np.delete(val.copy(), 0)
        Y = np.delete(X_temp,0)

    return(val[0], delta)

def BS_EU_put(T, r, S, sigma, K, t):

    d1 = 1 / (sigma * np.sqrt(T-t)) * ( np.log(S / K) + (r + sigma ** 2 / 2) * (T-t) )
    d2 = d1 - sigma * np.sqrt(T - t)

    price = S * (stats.norm(0,1).cdf(d1) - 1) - np.exp(-r * (T-t)) * K * (stats.norm(0,1).cdf(d2) - 1)

    delta = stats.norm(0,1).cdf(d1)-1

    return price, delta

x = np.linspace(10,90,200)

y_EU_0_price = [BS_EU_put(1, 0.06, s, 0.2, 40, 0)[0] for s in x]
y_EU_1_price = [BS_EU_put(1, 0.06, s, 0.2, 40, 0.99)[0] for s in x]
y_EU_0_delta = [BS_EU_put(1, 0.06, s, 0.2, 40, 0)[1] for s in x]
y_EU_1_delta = [BS_EU_put(1, 0.06, s, 0.2, 40, 0.99)[1] for s in x]

y_AMR_0_price = [CRR_put_Greeks(1, 10**4, 0.06, s, 0.2, 40,0)[0] for s in x]
y_AMR_1_price = [CRR_put_Greeks(1, 10**4, 0.06, s, 0.2, 40,0.99)[0] for s in x]
y_AMR_0_delta = [CRR_put_Greeks(1, 10**4, 0.06, s, 0.2, 40,0)[1] for s in x]
y_AMR_1_delta = [CRR_put_Greeks(1, 10**4, 0.06, s, 0.2, 40,0.99)[1] for s in x]

#Price - change for t=0 and t=0.9
plt.figure(facecolor="#fafafa") #colors outside the box
plt.grid(alpha=0.5, linestyle = '--', color = 'black', zorder = 1)
ax = plt.gca()
ax.set_facecolor("#fafafa")
ax.set_axisbelow(True)
plt.plot(x,y_EU_0_price, label = 'EU option price', linewidth = 4, color='grey')
plt.plot(x,y_AMR_0_price, label = 'AMR option price', linewidth = 4, color='tan', linestyle = '--')
#plt.title('Difference in Price for AMR and EU option at t=0', fontsize=36)
plt.xticks(fontsize=20)  # Enlarge tick labels on x-axis
plt.yticks(fontsize=20)
plt.xlabel("Stock price", fontsize=26)  # Bigger x-axis label
plt.ylabel("Option price", fontsize=26)
plt.legend(fontsize=26)
plt.tight_layout()

plt.figure(facecolor="#fafafa") #colors outside the box
plt.grid(alpha=0.5, linestyle = '--', color = 'black', zorder = 1)
ax = plt.gca()
ax.set_facecolor("#fafafa")
ax.set_axisbelow(True)
plt.plot(x,y_EU_1_price, label = 'EU option price', linewidth = 4, color='grey')
plt.plot(x,y_AMR_1_price, label = 'AMR option price', linewidth = 4, color='tan', linestyle = '--')
#plt.title('Difference in Price for AMR and EU option at t=0', fontsize=36)
plt.xticks(fontsize=20)  # Enlarge tick labels on x-axis
plt.yticks(fontsize=20)
plt.xlabel("Stock price", fontsize=26)  # Bigger x-axis label
plt.ylabel("Option price", fontsize=26)
plt.legend(fontsize=26)
plt.tight_layout()

#Delta - change for t=0 and t=0.9
plt.figure(facecolor="#fafafa") #colors outside the box
plt.grid(alpha=0.5, linestyle = '--', color = 'black', zorder = 1)
ax = plt.gca()
ax.set_facecolor("#fafafa")
ax.set_axisbelow(True)
plt.plot(x,y_EU_0_delta, label = 'EU option delta', linewidth = 4, color='grey')
plt.plot(x,y_AMR_0_delta, label = 'AMR option delta', linewidth = 4, color='tan', linestyle = '--')
#plt.title('Difference in Delta for AMR and EU option at t=0', fontsize=36)
plt.xticks(fontsize=20)  # Enlarge tick labels on x-axis
plt.yticks(fontsize=20)
plt.xlabel("Stock price", fontsize=26)  # Bigger x-axis label
plt.ylabel("Delta", fontsize=26)
plt.legend(fontsize=26)
plt.tight_layout()

plt.figure(facecolor="#fafafa") #colors outside the box
plt.grid(alpha=0.5, linestyle = '--', color = 'black', zorder = 1)
ax = plt.gca()
ax.set_facecolor("#fafafa")
ax.set_axisbelow(True)
plt.plot(x,y_EU_1_delta, label = 'EU option delta', linewidth = 4, color='grey')
plt.plot(x,y_AMR_1_delta, label = 'AMR option delta', linewidth = 4, color='tan', linestyle = '--')
#plt.title('Difference in Delta for AMR and EU option at t=0', fontsize=36)
plt.xticks(fontsize=20)  # Enlarge tick labels on x-axis
plt.yticks(fontsize=20)
plt.xlabel("Stock price", fontsize=26)  # Bigger x-axis label
plt.ylabel("Delta", fontsize=26)
plt.legend(fontsize=26)

#Change in smoothness of Delta function, based on number of points (n)

x_50 = np.linspace(30,60,51)
x_20 = np.linspace(30,60,21)
x_8 = np.linspace(30,60,9)

y_50_0 = [CRR_put_Greeks(1, 10**4, 0.06, s, 0.2, 40, 0)[1] for s in x_50]
y_20_0 = [CRR_put_Greeks(1, 10**4, 0.06, s, 0.2, 40, 0)[1] for s in x_20]
y_8_0 = [CRR_put_Greeks(1, 10**4, 0.06, s, 0.2, 40, 0)[1] for s in x_8]

y_50_1 = [CRR_put_Greeks(1, 10**4, 0.06, s, 0.2, 40, 0.99)[1] for s in x_50]
y_20_1 = [CRR_put_Greeks(1, 10**4, 0.06, s, 0.2, 40, 0.99)[1] for s in x_20]
y_8_1 = [CRR_put_Greeks(1, 10**4, 0.06, s, 0.2, 40, 0.99)[1] for s in x_8]

#Change the date with t = 0, and t = 0.99
plt.figure(facecolor="#fafafa") #colors outside the box
plt.grid(alpha=0.5, linestyle = '--', color = 'black', zorder = 1)
ax = plt.gca()
ax.set_facecolor("#fafafa")
ax.set_axisbelow(True)
plt.plot(x_50,y_50_0, color='black', linewidth=4, label = 'n = 50', linestyle = '-')
plt.plot(x_20,y_20_0, color='pink', linewidth=4, label = 'n = 20',linestyle = '--')
plt.plot(x_8,y_8_0, color='tan', linewidth=4, label = 'n = 8', linestyle = ':')
#plt.title('Smoothness of Delta, with different amount of interpolation points (n), at t = 0  ', fontsize=32)
plt.xticks(fontsize=20)  # Enlarge tick labels on x-axis
plt.yticks(fontsize=20)
plt.xlabel("Stock price", fontsize=26)  # Bigger x-axis label
plt.ylabel("Delta", fontsize=26)
plt.legend(fontsize=26)

plt.figure(facecolor="#fafafa") #colors outside the box
plt.grid(alpha=0.5, linestyle = '--', color = 'black', zorder = 1)
ax = plt.gca()
ax.set_facecolor("#fafafa")
ax.set_axisbelow(True)
plt.plot(x_50,y_50_1, color='black', linewidth=4, label = 'n = 50', linestyle = '-')
plt.plot(x_20,y_20_1, color='pink', linewidth=4, label = 'n = 20',linestyle = '--')
plt.plot(x_8,y_8_1, color='tan', linewidth=4, label = 'n = 8', linestyle = ':')
#plt.title('Smoothness of Delta, with different amount of interpolation points (n), at t = 0.99  ', fontsize=32)
plt.xticks(fontsize=20)  # Enlarge tick labels on x-axis
plt.yticks(fontsize=20)
plt.xlabel("Stock price", fontsize=26)  # Bigger x-axis label
plt.ylabel("Delta", fontsize=26)
plt.legend(fontsize=26)


import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import math
import time

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

def BM(T, N, r, S, sigma, Omega):
    dt = 1 / N
    Z = np.random.normal(0, np.sqrt(dt), size=(Omega, T * N))
    x = np.zeros((Omega, T * N + 1))
    x[:, 0] = S
    x[:, 1:] = S * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * dt + sigma * Z, axis=1))
    return x

def Lag_Pol(X):
    L_0 = np.exp(-X / 2)
    L_1 = np.exp(-X / 2) * (1 - X)
    L_2 = np.exp(-X / 2) * (1 - 2 * X + X ** 2 / 2)
    return np.column_stack((L_0, L_1, L_2))

def Lag_Pol_2(X):
    L_0 = np.exp(-X / 2)
    L_1 = np.exp(-X / 2) * (1 - X)
    return np.column_stack((L_0, L_1))

def Lag_Pol_5(X):
    L_0 = np.exp(-X / 2)
    L_1 = np.exp(-X / 2) * (1 - X)
    L_2 = np.exp(-X / 2) * (1 - 2 * X + X ** 2 / 2)
    L_3 = np.exp(-X / 2) * (1 - 3 * X + 3/2 * X ** 2 - 1/6 * X ** 3)
    L_4 = np.exp(-X / 2) * (1 - 4 * X + 3 * X ** 2 - 2/3 * X ** 3 + 1/24 * X ** 4)
    return np.column_stack((L_0, L_1, L_2, L_3, L_4))


def LSM_put(T, N, r, S, sigma, Omega, K, sim):
    dt = 1 / N
    data = sim[:Omega,:] / K  # Normalize data
    CFM = np.maximum(1 - data, 0)
    CFM[:, 0] = 0
    A_CFM = CFM.copy()

    disc = np.exp(-r * dt)

    for i in range(T*N, 1, -1):
        ITM = CFM[:, i - 1] > 0

        Y = disc * A_CFM[ITM, i]
        X = data[ITM, i - 1]
        Z = Lag_Pol(X)

        model = LinearRegression()
        model.fit(Z, Y)
        a = model.intercept_
        b, c, d = model.coef_

        Q = Lag_Pol(data[:, i - 1])
        E = a + b * Q[:, 0] + c * Q[:, 1] + d * Q[:, 2]

        CFM[:, i - 1] = np.where(CFM[:, i - 1] < E, 0, CFM[:, i - 1])
        CFM[CFM[:, i - 1] > 0, i:] = 0
        A_CFM[:, i - 1] = np.where(CFM[:, i - 1] == 0, A_CFM[:, i] * disc, A_CFM[:, i - 1])

    disc_vec = np.vander([disc], T * N + 1, increasing = True).T

    disc_CashFlows = CFM @ disc_vec

    price = sum(disc_CashFlows) * K / Omega
    se = np.sqrt(sum((disc_CashFlows * K - price) ** 2)) / Omega

    return (price, se)

def LSM_put_2(T, N, r, S, sigma, Omega, K, sim):
    dt = 1 / N
    data = sim[:Omega,:] / K  # Normalize data
    CFM = np.maximum(1 - data, 0)
    CFM[:, 0] = 0
    A_CFM = CFM.copy()

    disc = np.exp(-r * dt)

    for i in range(T*N, 1, -1):
        ITM = CFM[:, i - 1] > 0

        Y = disc * A_CFM[ITM, i]
        X = data[ITM, i - 1]
        Z = Lag_Pol_2(X)

        model = LinearRegression()
        model.fit(Z, Y)

        Q = Lag_Pol_2(data[:, i - 1])
        E = model.predict(Q)

        CFM[:, i - 1] = np.where(CFM[:, i - 1] < E, 0, CFM[:, i - 1])
        CFM[CFM[:, i - 1] > 0, i:] = 0
        A_CFM[:, i - 1] = np.where(CFM[:, i - 1] == 0, A_CFM[:, i] * disc, A_CFM[:, i - 1])

    disc_vec = np.vander([disc], T * N + 1, increasing = True).T

    disc_CashFlows = CFM @ disc_vec

    price = sum(disc_CashFlows) * K / Omega
    se = np.sqrt(sum((disc_CashFlows * K - price) ** 2)) / Omega

    return (price, se)

def LSM_put_5(T, N, r, S, sigma, Omega, K, sim):
    dt = 1 / N
    data = sim[:Omega,:] / K  # Normalize data
    CFM = np.maximum(1 - data, 0)
    CFM[:, 0] = 0
    A_CFM = CFM.copy()

    disc = np.exp(-r * dt)

    for i in range(T*N, 1, -1):
        ITM = CFM[:, i - 1] > 0

        Y = disc * A_CFM[ITM, i]
        X = data[ITM, i - 1]
        Z = Lag_Pol_5(X)

        model = LinearRegression()
        model.fit(Z, Y)

        Q = Lag_Pol_5(data[:, i - 1])
        E = model.predict(Q)

        CFM[:, i - 1] = np.where(CFM[:, i - 1] < E, 0, CFM[:, i - 1])
        CFM[CFM[:, i - 1] > 0, i:] = 0
        A_CFM[:, i - 1] = np.where(CFM[:, i - 1] == 0, A_CFM[:, i] * disc, A_CFM[:, i - 1])

    disc_vec = np.vander([disc], T * N + 1, increasing = True).T

    disc_CashFlows = CFM @ disc_vec

    price = sum(disc_CashFlows) * K / Omega
    se = np.sqrt(sum((disc_CashFlows * K - price) ** 2)) / Omega

    return (price, se)

def LSM_put_pol(T, N, r, S, sigma, Omega, K, sim):
    M = 9
    dt = 1 / N
    data = sim[:Omega,:] / K
    CFM = np.maximum(1 - data, 0)
    CFM[:, 0] = 0
    A_CFM = CFM.copy()

    cont_bound = []  # Start with the final boundary value (strike price)
    disc = np.exp(-r * dt)

    for i in range(T * N, 0, -1):
        ITM = CFM[:, i - 1] > 0
        if not ITM.any():  # If no valid points, append strike price
            continue

        Y = disc * A_CFM[ITM, i]
        X = data[ITM, i - 1]
        Z = np.vander(X, M ,increasing=True)[:,1:]

        model = LinearRegression()
        model.fit(Z, Y)

        Q = np.vander(data[:, i - 1],M,increasing=True)[:,1:]

        E = model.predict(Q)

        CFM[:, i - 1] = np.where(CFM[:, i - 1] < E, 0, CFM[:, i - 1])
        CFM[CFM[:, i - 1] > 0, i:] = 0
        A_CFM[:, i - 1] = np.where(CFM[:, i - 1] == 0, A_CFM[:, i] * disc, A_CFM[:, i - 1])

    disc_vec = np.vander([disc], T * N + 1, increasing = True).T

    disc_CashFlows = CFM @ disc_vec

    price = sum(disc_CashFlows) * K / Omega
    se = np.sqrt(sum((disc_CashFlows * K - price) ** 2)) / Omega

    return (price, se)

def LSM_put_pol_5(T, N, r, S, sigma, Omega, K, sim):
    M = 5
    dt = 1 / N
    data = sim[:Omega, :] / K
    CFM = np.maximum(1 - data, 0)
    CFM[:, 0] = 0
    A_CFM = CFM.copy()

    cont_bound = []  # Start with the final boundary value (strike price)
    disc = np.exp(-r * dt)

    for i in range(T * N, 0, -1):
        ITM = CFM[:, i - 1] > 0
        if not ITM.any():  # If no valid points, append strike price
            continue

        Y = disc * A_CFM[ITM, i]
        X = data[ITM, i - 1]
        Z = np.vander(X, M, increasing=True)[:, 1:]

        model = LinearRegression()
        model.fit(Z, Y)

        Q = np.vander(data[:, i - 1], M, increasing=True)[:, 1:]

        E = model.predict(Q)

        CFM[:, i - 1] = np.where(CFM[:, i - 1] < E, 0, CFM[:, i - 1])
        CFM[CFM[:, i - 1] > 0, i:] = 0
        A_CFM[:, i - 1] = np.where(CFM[:, i - 1] == 0, A_CFM[:, i] * disc, A_CFM[:, i - 1])

    disc_vec = np.vander([disc], T * N + 1, increasing=True).T

    disc_CashFlows = CFM @ disc_vec

    price = sum(disc_CashFlows) * K / Omega
    se = np.sqrt(sum((disc_CashFlows * K - price) ** 2)) / Omega

    return (price, se)



function_names = ['Standard', 'no normalizer', '2 basis', '5 basis', 'Legendre - 3 basis', 'Legendre - 5 basis']

BM_0 = BM(1,50,0.06,36,0.2,5*10**5)

def f1(x):
    return LSM_put(1,50,0.06,36,0.2,x,40, BM_0)
def f2(x):
    return LSM_put_2(1,50,0.06,36,0.2,x,40, BM_0)
def f3(x):
    return LSM_put_5(1,50,0.06,36,0.2,x,40, BM_0)
def f4(x):
    return LSM_put_pol(1,50,0.06,36,0.2,x,40, BM_0)

x = np.linspace(2,5.5, 200)

x_log = 10**x

x_log_int = [int(i) for i in x_log]

y_lag_3 = np.array([f1(i) for i in x_log_int])[:,:,0]
y_lag_2 = np.array([f2(i) for i in x_log_int])[:,:,0]
y_lag_5 = np.array([f3(i) for i in x_log_int])[:,:,0]
y_pol_9 = np.array([f4(i) for i in x_log_int])[:,:,0]

#Price
plt.figure(facecolor="#fafafa") #colors outside the box
plt.grid(alpha=0.5, linestyle = '--', color = 'black', zorder = 1)
ax = plt.gca()
ax.set_facecolor("#fafafa")
ax.set_axisbelow(True)
plt.axhline(CRR_put(1,10**4,0.06,36,0.2,40), label = 'Benchmark Price', linestyle = ':', linewidth = 4, color = 'black')
plt.plot(x,y_lag_2[:,0], label = '2 basis', linewidth = 3, color = 'pink', linestyle = '-')
plt.plot(x,y_lag_3[:,0], label = '3 basis', linewidth = 3, color = '#d95f0e', linestyle = '-.')
plt.plot(x,y_lag_5[:,0], label = '5 basis', linewidth = 3, color = 'tan', linestyle = '-')
plt.plot(x,y_pol_9[:,0], label = '9 basis (polynomial)', linewidth = 3, color = 'grey', linestyle = '-.')
#plt.title('Choice of basis, and amount of stock paths - Price', fontsize=36)
plt.xticks(fontsize=20)  # Enlarge tick labels on x-axis
plt.yticks(fontsize=20)
plt.xlabel("log_10 transform of Omega", fontsize=26)  # Bigger x-axis label
plt.ylabel("Option price", fontsize=26)
plt.legend(fontsize=26)
plt.ylim((4.38,4.72))

#S.E.
plt.figure(facecolor="#fafafa") #colors outside the box
plt.grid(alpha=0.5, linestyle = '--', color = 'black', zorder = 1)
ax = plt.gca()
ax.set_facecolor("#fafafa")
ax.set_axisbelow(True)
plt.plot(x,y_lag_2[:,1], label = '2 basis', linewidth = 3, color = 'pink', linestyle = '--')
plt.plot(x,y_lag_3[:,1], label = '3 basis', linewidth = 3, color = '#d95f0e', linestyle = '-.')
plt.plot(x,y_lag_5[:,1], label = '5 basis', linewidth = 3, color = 'tan', linestyle = '--')
plt.plot(x,y_pol_9[:,1], label = '9 basis (polynomial)', linewidth = 3, color = 'grey', linestyle = '-.')
#plt.title('Choice of basis, and amount of stock paths - s.e.', fontsize=36)
plt.xticks(fontsize=20)  # Enlarge tick labels on x-axis
plt.yticks(fontsize=20)
plt.xlabel("log_10 transform of Omega", fontsize=26)  # Bigger x-axis label
plt.ylabel("Option price s.e.", fontsize=26)
plt.legend(fontsize=26)
plt.ylim((0,0.15))
plt.tight_layout()

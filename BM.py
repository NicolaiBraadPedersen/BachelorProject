import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import time
import seaborn as sns

def BM(T, N, r, S, sigma, Omega):
    dt = 1 / N
    Z = np.random.normal(0, np.sqrt(dt), size=(Omega, int(T * N)))
    x = np.zeros((Omega, int(T * N) + 1))
    x[:, 0] = S
    x[:, 1:] = S * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * dt + sigma * Z, axis=1))
    return x

# Kernel function
def Kernel(x):
    return 2 * np.sin(np.arcsin( 2 * x - 1 ) / 3)

def BM_ISD(T, N, r, S, sigma, Omega, alpha):
    dt = 1 / N

    # Generate initial stock prices
    U = np.random.uniform(low=0, high=1, size=Omega)
    X_0 = S + alpha * Kernel(U)  # Shape: (Omega,)

    # Generate Brownian motion increments
    Z = np.random.normal(0, np.sqrt(dt), size=(Omega, T * N))  # Shape: (Omega, T*N)

    # Compute cumulative sum for Brownian motion
    exponent = np.cumsum((r - 0.5 * sigma ** 2) * dt + sigma * Z, axis=1)  # Shape: (Omega, T*N)

    # Compute stock prices using vectorized operations
    x = np.zeros((Omega, T * N + 1))
    x[:, 0] = X_0  # Set initial prices
    x[:, 1:] = X_0[:, np.newaxis] * np.exp(exponent)  # Broadcast multiplication

    return x

def BM_ISD_2(T,N,r,S,sigma,Omega,alpha,alpha_opt):
    dt = 1 / N

    # Generate initial stock prices
    U = np.random.uniform(low=0, high=1, size=Omega)
    X_0 = S + alpha_opt * Kernel(U)  # Shape: (Omega,)

    # Generate Brownian motion increments
    Z = np.random.normal(0, np.sqrt(dt), size=(Omega, T * N))  # Shape: (Omega, T*N)

    # Compute cumulative sum for Brownian motion
    exponent = np.cumsum((r - 0.5 * sigma ** 2) * dt + sigma * Z, axis=1)  # Shape: (Omega, T*N)

    # Compute stock prices using vectorized operations
    x = np.zeros((Omega, T * N + 1))
    x[:, 0] = X_0  # Set initial prices
    x[:, 1:] = X_0[:, np.newaxis] * np.exp(exponent)  # Broadcast multiplication

    return x

def Lag_Pol(X):
    exp_neg_half_X = np.exp(-X / 2)
    L_0 = exp_neg_half_X
    L_1 = exp_neg_half_X * (1 - X)
    L_2 = exp_neg_half_X * (1 - 2 * X + X ** 2 / 2)
    return np.column_stack((L_0, L_1, L_2))


def LSM_put_bound(T, N, r, S, sigma, Omega, K):
    dt = 1 / N
    data = BM(T, N, r, S, sigma, Omega) / K
    CFM = np.maximum(1 - data, 0)
    CFM[:, 0] = 0
    A_CFM = CFM.copy()

    disc = np.exp(-r * dt)

    for i in range(T * N, 0, -1):
        ITM = CFM[:, i - 1] > 0
        if not ITM.any():  # If no valid points, append strike price
            continue

        Y = disc * A_CFM[ITM, i]
        X = data[ITM, i - 1]
        Z = Lag_Pol(X)

        model = LinearRegression()
        model.fit(Z, Y)
        a, b, c, d = model.intercept_, *model.coef_

        Q = data[:, i - 1]
        E = a + b * Lag_Pol(Q)[:, 0] + c * Lag_Pol(Q)[:, 1] + d * Lag_Pol(Q)[:, 2]

        CFM[:, i - 1] = np.where(CFM[:, i - 1] < E, 0, CFM[:, i - 1])
        CFM[CFM[:, i - 1] > 0, i:] = 0
        A_CFM[:, i - 1] = np.where(CFM[:, i - 1] == 0, A_CFM[:, i] * disc, A_CFM[:, i - 1])

    cont_bound = []

    for i in range(T*N):
        has_cashflow = CFM[:,i] > 0
        if np.any(CFM[:,i] > 0) == True:
            cont_bound_element = (K - np.min(CFM[:,i][has_cashflow]) * K)
        else:
            cont_bound_element = np.nan
        cont_bound.append(cont_bound_element)

    cont_bound.append(40)
    return cont_bound

def LSM_put_bound_ISD(T, N, r, S, sigma, Omega, K, alpha):
    dt = 1 / N
    data = BM_ISD(T, N, r, S, sigma, Omega, alpha) / K
    CFM = np.maximum(1 - data, 0)
    CFM[:, 0] = 0
    A_CFM = CFM.copy()

    disc = np.exp(-r * dt)

    for i in range(T * N, 0, -1):
        ITM = CFM[:, i - 1] > 0
        if not ITM.any():  # If no valid points, append strike price
            continue

        Y = disc * A_CFM[ITM, i]
        X = data[ITM, i - 1]
        Z = Lag_Pol(X)

        model = LinearRegression()
        model.fit(Z, Y)
        a, b, c, d = model.intercept_, *model.coef_

        Q = data[:, i - 1]
        E = a + b * Lag_Pol(Q)[:, 0] + c * Lag_Pol(Q)[:, 1] + d * Lag_Pol(Q)[:, 2]

        CFM[:, i - 1] = np.where(CFM[:, i - 1] < E, 0, CFM[:, i - 1])
        CFM[CFM[:, i - 1] > 0, i:] = 0
        A_CFM[:, i - 1] = np.where(CFM[:, i - 1] == 0, A_CFM[:, i] * disc, A_CFM[:, i - 1])

    cont_bound = []

    for i in range(T*N):
        has_cashflow = CFM[:,i] > 0
        if np.any(CFM[:,i] > 0) == True:
            cont_bound_element = (K - np.min(CFM[:,i][has_cashflow]) * K)
        else:
            cont_bound_element = np.nan
        cont_bound.append(cont_bound_element)

    cont_bound.append(40)
    return cont_bound

#Plotting the initial distribution of the stock prices
A = BM_ISD(1,50,0.06,40,0.2,10**5,5)[:,0]
plt.figure(facecolor="#fafafa")
plt.grid(alpha=0.5, linestyle = '--', color = 'black', zorder = 1)
ax = plt.gca()
ax.set_facecolor("#fafafa")
ax.set_axisbelow(True)
plt.hist(A, bins=40, density=True, edgecolor='black', color='#e6d7c3')
sns.kdeplot(A, color='tan', linewidth=2)
plt.xticks(fontsize=20)  # Enlarge tick labels on x-axis
plt.yticks(fontsize=20)
plt.ylabel('')
plt.xlim(34.5, 45.5)
plt.tight_layout()

#Plotting the Sample paths and optimal stopping boundary
cmap = ['#d95f0e','pink','grey','tan']

BM_0 = BM_ISD(1,2000,0.06,40,0.2,4,5)
data = np.array(BM_0)
x = np.linspace(0,1,len(data[1]))

y1 = LSM_put_bound(1,50,0.06,36,0.2,10**5,40, 5)
x1 = np.linspace(0,1,len(y1))

y = np.interp(np.linspace(0, len(y1)-1, len(data[1])), np.arange(len(y1)), y1)

plt.figure(facecolor="#fafafa") #colors outside the box
plt.grid(alpha=0.5, linestyle = '--', color = 'black', zorder = 1)
ax = plt.gca()
ax.set_facecolor("#fafafa")
ax.set_axisbelow(True)

for i in range(len(data)):
    clr = cmap[i]
    arr = data[i]
    indices = np.where(arr <= y)

    if indices[0].size > 0:
        index = indices[0][0]

        first_part_y = arr[:index + 1]
        first_part_x = x[:index + 1]
        second_part_y = arr[index + 1:]
        second_part_x = x[index + 1:]
        plt.plot(first_part_x, first_part_y, color=clr)
        plt.plot(second_part_x, second_part_y, color=clr, alpha=0.30)

    else:
        first_part_y = arr
        first_part_x = x
        plt.plot(first_part_x, first_part_y, color=clr)

plt.plot(x1,y1, color = 'black', linestyle ='-', linewidth = 4, label = 'Optimal stopping boundary')
plt.xlabel("Time", fontsize=26)  # Bigger x-axis label
plt.ylabel("Stock price", fontsize=26)
plt.legend(fontsize = 26)
#plt.title('Black-Scholes model and optimal stopping boundary', fontsize=36)
plt.xticks(fontsize=20)  # Enlarge tick labels on x-axis
plt.yticks(fontsize=20)

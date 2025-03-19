import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from itertools import product
from BM import BM
from Hedging import *
from LSM_put_bound import *

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

def Lag_Pol(X):
    L_0 = np.exp(-X / 2)
    L_1 = np.exp(-X / 2) * (1 - X)
    L_2 = np.exp(-X / 2) * (1 - 2 * X + X ** 2 / 2)
    return np.column_stack((L_0, L_1, L_2))

def LSM_put_data(T, N, r, Omega, K, Stocks):

    dt = 1 / N
    data = Stocks / K  # Normalize data
    CFM = np.maximum(1 - data, 0)
    CFM[:, 0] = 0
    A_CFM = CFM.copy()

    disc = np.exp(-r * dt)

    for i in range(int(T*N), 1, -1):
        ITM = CFM[:, i - 1] > 0
        if np.sum(ITM) == 0:
            continue
        Y = disc * A_CFM[ITM, i]
        X = data[ITM, i - 1]
        Z = Lag_Pol(X)

        model = LinearRegression()
        model.fit(Z, Y)
        a = model.intercept_
        b, c, d = model.coef_

        Q = data[:, i - 1]
        E = a + b * Lag_Pol(Q)[:, 0] + c * Lag_Pol(Q)[:, 1] + d * Lag_Pol(Q)[:, 2]

        CFM[:, i - 1] = np.where(CFM[:, i - 1] < E, 0, CFM[:, i - 1])
        CFM[CFM[:, i - 1] > 0, i:] = 0
        A_CFM[:, i - 1] = np.where(CFM[:, i - 1] == 0, A_CFM[:, i] * disc, A_CFM[:, i - 1])

    disc_vec = np.vander([disc], int(T * N) + 1, increasing = True).T

    disc_CashFlows = CFM @ disc_vec

    price = sum(disc_CashFlows) * K / Omega
    se = np.sqrt(sum((disc_CashFlows * K - price) ** 2)) / Omega

    #This step below, is to return price for every state, and hereby gain much

    return (price, se)

def LSM_put_findiff(T, N, r, S, sigma, Omega, K, epsilon):
    df = BM(T, N, r, S, sigma, Omega)

    Stocks_0 = df * (1 - epsilon/S)
    Stocks_epsilon = df * (1 + epsilon/S)

    price_0 = LSM_put_data(T, N, r, Omega, K, Stocks_0)[0]
    price_epsilon = LSM_put_data(T, N, r, Omega, K, Stocks_epsilon)[0]

    return (price_epsilon - price_0) / (2 * epsilon)

x = np.linspace(30,60,20)

#Get the benchmark from CRR greeks, this is done with 200 points between 10, 90
x_benchmark = np.linspace(30,60,201)
y_benchmark = [CRR_put_Greeks(1,10**4,0.06,s, 0.2, 40, 0)[1] for s in x_benchmark]

#Here we find the LSM delta, with a finite difference method, with varying epsilon

y_1 = [LSM_put_findiff(1, 50, 0.06, s, 0.2,10**5,40,0.1) for s in x]
y_2 = [LSM_put_findiff(1, 50, 0.06, s, 0.2,10**5,40,0.05) for s in x]
y_4 = [LSM_put_findiff(1, 50, 0.06, s, 0.2,10**5,40,0.005) for s in x]

plt.figure(facecolor="#fafafa") #colors outside the box
plt.grid(alpha=0.5, linestyle = '--', color = 'black', zorder = 1)
ax = plt.gca()
ax.set_facecolor("#fafafa")
ax.set_axisbelow(True)
plt.plot(x_benchmark, y_benchmark, label = 'benchmark', linewidth = 8, color = 'black')
plt.plot(x,y_1, label = 'epsilon = 0.1', linewidth = 6, color = '#d95f0e', linestyle = '--')
plt.plot(x,y_2, label = 'epsilon = 0.05', linewidth = 4, color = 'pink', linestyle = '-.')
plt.plot(x,y_4, label = 'epsilon = 0.005', linewidth = 4, color = 'tan', linestyle = ':')
#plt.title('LSM finite difference - Smoothness of Delta, with different epsilon', fontsize=36)
plt.xticks(fontsize=20)  # Enlarge tick labels on x-axis
plt.yticks(fontsize=20)
plt.xlabel("Stock price", fontsize=26)  # Bigger x-axis label
plt.ylabel("Delta", fontsize=26)
plt.legend(fontsize=26)
plt.tight_layout()

#Now we wish to find the hedging error of this method
Omega0 = 10**5
S0 = 40
N0 = 50
T0 = 1
r0 = 0.06
interest_rate = np.exp(r0*T0/N0)
sigma0 = 0.2
K0 = 40
n0 = 20 #number of interpolation points

price = CRR_put_Greeks(T0,10**4,r0,S0,sigma0,K0,0)[0]
#finding the continuation/stopping region for the stock, so we are able to find when the option is exercised
Cont_bound = LSM_put_bound(T0, N0, r0, S0, sigma0, 1 * 10**5, K0)


#For the greeks interpolation function: Note that we need to change the -10 to -30 for EU options
s0 = np.linspace(K0 - 10, K0 + 20, n0 + 1)

time_grid = np.linspace(0, T0 * N0 - 1, T0 * N0, dtype=int)
#Creating delta matrix from above information
f_s_t = np.array([[LSM_put_findiff(1-t/N0,N0,0.06,s,0.2,10**5,40,0.05) for s in s0] for t in time_grid])


#Simulating stock paths
Stock_paths = BM(T0, N0, r0, S0, sigma0, Omega0)

#Calculating delta for all stock paths at all timepoints
Delta = Delta_Interpol(f_s_t, s0, time_grid, T0, N0, K0, Omega0, Stock_paths)

#The stopping time for the option.
exercise_mask = (Stock_paths <= Cont_bound)
exercise_mask &= np.cumsum(exercise_mask, axis=1) == 1
#This extra mask, is true until we hit the stopping time, we use this when we hedge, since we adjust our portfolio until exercise
before_exercise_mask = np.cumsum(exercise_mask, axis=1) == 0

#CashFlowMatrix for the Option
option_CFM = np.zeros_like(Stock_paths)
option_CFM[exercise_mask] = (K0 - Stock_paths)[exercise_mask]

#We define the replicating portfolio value
replicating_pf = np.zeros_like(Stock_paths)
replicating_pf[:,0] = price


#We define the risk neutral position, that should help us finance the short position in the stock
riskneutral_position = np.zeros_like(Stock_paths)
riskneutral_position[:,0] = replicating_pf[:,0] - Delta[:,0] * Stock_paths[:,0]

for i in range(T0*N0):
    replicating_pf[:,i+1] = riskneutral_position[:,i] * interest_rate + Delta[:,i] * Stock_paths[:,i+1]
    riskneutral_position[:, i+1] = replicating_pf[:, i+1] - Delta[:,i+1] * Stock_paths[:,i+1]



#We now find the hedging errors at all exercise points
hedge_errors = np.zeros_like(Stock_paths)
hedge_errors[exercise_mask] = (replicating_pf - option_CFM)[exercise_mask]
#We also need to include all the errors for options, that weren't exercised
nonexercised_paths_mask = np.sum(exercise_mask, axis=1) == 0
hedge_errors[nonexercised_paths_mask,N0] = (replicating_pf - (option_CFM))[nonexercised_paths_mask,N0]

disc_vector = np.vander([interest_rate],N0+1).T

disc_hedge_errors = (hedge_errors @ disc_vector)[:,0]

plt.figure(facecolor="#fafafa") #colors outside the box
plt.grid(alpha=0.5, linestyle = '--', color = 'black', zorder = 1)
ax = plt.gca()
ax.set_facecolor("#fafafa")
ax.set_axisbelow(True)
plt.hist(disc_hedge_errors, bins=40, density=True, edgecolor='black', color='#e6d7c3')
#plt.title('LSM finite difference - Hedging error', fontsize=36)
sns.kdeplot(disc_hedge_errors, color='tan', linewidth=2)
plt.xticks(fontsize=20)  # Enlarge tick labels on x-axis
plt.yticks(fontsize=20)
plt.xlim(-2.5, 2.5)
plt.ylabel("")
plt.tight_layout()

print(
np.mean(disc_hedge_errors),
np.var(disc_hedge_errors),
np.min(disc_hedge_errors),
np.max(disc_hedge_errors)
)
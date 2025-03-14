import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as stats
from scipy.interpolate import RegularGridInterpolator
from sklearn.linear_model import LinearRegression
import seaborn as sns
from BM import *
from Interpolation import *

def CRR_put_Greeks(T, N, r, S, sigma, K):
    dt = 1/N
    u = np.exp( sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    R = np.exp(r*dt)
    p = (R - d) / (u - d)
    #print (u, d, R)

    X = np.array([max(0, K - (S * (d ** k * u ** (T * N-k)))) for k in range(T * N+1)])

    XX = X.copy()

    X = np.delete(X, -1)
    XX = np.delete(XX, 0)


    Y = np.array([max(0, K - (S * (d ** k * u ** (T * N-k)))) for k in range(T * N)])

    for i in range(T * N, 0, -1):

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

# Parameters
Omega0 = 10**5
S0 = 40
N0 = 50
T0 = 1
r0 = 0.06
interest_rate = np.exp(r0*T0/N0)
sigma0 = 0.2
K0 = 40
n0 = 50 #number of interpolation points

def BS_EU_put(T, r, S, sigma, K, t):

    d1 = 1 / (sigma * np.sqrt(T - t)) * ( np.log(S / K) + (r + sigma ** 2 / 2) * (T-t) )
    d2 = d1 - sigma * np.sqrt(T - t)

    price = S * (stats.norm(0,1).cdf(d1) - 1) - np.exp(-r * (T-t)) * K * (stats.norm(0,1).cdf(d2) - 1)

    delta = stats.norm(0,1).cdf(d1)-1

    Gamma = stats.norm(0,1).pdf(d1) / ( S * sigma * np.sqrt(T-t))

    return price, delta, Gamma

def Delta_EU(T, r, S, sigma, K, t):
    d1 = 1 / (sigma * np.sqrt(T - t)) * (np.log(S / K) + (r + sigma ** 2 / 2) * (T - t))
    return stats.norm(0,1).cdf(d1)-1

#For the greeks interpolation function: Note that we need to change the -10 to -30 for EU options
s0 = np.linspace(K0 - 20, K0 + 20, n0 + 1)
time_grid = np.linspace(0, T0 * N0 - 1, T0 * N0, dtype=int)

#Price of the option
price = CRR_put_Greeks(T0, N0, r0, S0, sigma0, K0)[0]

#Creating delta matrix from above information
f_s_t = np.array([[Delta_EU(T0,0.06,s,0.2,K0,t/N0) for s in s0] for t in time_grid]).T

#finding the continuation/stopping region for the stock, so we are able to find when the option is exercised
Cont_bound = LSM_put_bound(T0, N0, r0, S0, sigma0, 10**5, K0)

#Simulating stock paths
Stock_paths = BM(T0, N0, r0, S0, sigma0, Omega0)

#Calculating delta for all stock paths at all timepoints
Delta = Delta_Interpol_EU(f_s_t, s0, time_grid, T0, N0, K0, Omega0, Stock_paths)

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
replicating_pf[:,0] = np.full(Omega0, price)

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

hedge_errors_early = disc_hedge_errors[~nonexercised_paths_mask]
hedge_errors_end = disc_hedge_errors[nonexercised_paths_mask]

plt.figure(facecolor="#fafafa") #colors outside the box
plt.grid(alpha=0.5, linestyle = '--', color = 'black', zorder = 1)
ax = plt.gca()
ax.set_facecolor("#fafafa")
ax.set_axisbelow(True)
plt.hist(hedge_errors_early, bins=40, density=True, edgecolor='black', color='#e6d7c3', label='Exercised options')
sns.kdeplot(hedge_errors_early, color='tan', linewidth=2)
plt.hist(hedge_errors_end, bins=40, density=True, edgecolor='black', color='#BDBDBD', label='Non Exercised options')
sns.kdeplot(hedge_errors_end, color='grey', linewidth=2)
plt.legend(fontsize=26)
#plt.title('Black-Scholes (EU delta) - Hedging error (split data)', fontsize=36)
plt.xlim(-2.5, 2.5)
plt.xticks(fontsize=20)  # Enlarge tick labels on x-axis
plt.yticks(fontsize=20)
plt.ylabel('')

plt.figure(facecolor="#fafafa") #colors outside the box
plt.grid(alpha=0.5, linestyle = '--', color = 'black', zorder = 1)
ax = plt.gca()
ax.set_facecolor("#fafafa")
ax.set_axisbelow(True)
plt.hist(disc_hedge_errors, bins=40, density=True, edgecolor='black', color='#e6d7c3')
sns.kdeplot(disc_hedge_errors, color='tan', linewidth=2)
#plt.title('Black-Scholes (EU delta) - Hedging error', fontsize=36)
plt.xlim(-2.5, 2.5)
plt.xticks(fontsize=20)  # Enlarge tick labels on x-axis
plt.yticks(fontsize=20)
plt.ylabel('')

print(
np.mean(disc_hedge_errors),
np.var(disc_hedge_errors),
np.min(disc_hedge_errors),
np.max(disc_hedge_errors)
)
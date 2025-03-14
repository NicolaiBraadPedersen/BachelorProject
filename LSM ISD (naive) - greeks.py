import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import copy
import seaborn as sns
from BM import BM_ISD
#BM_ISD(T, N, r, S, sigma, Omega, alpha)
from LSM_put_bound import LSM_put_bound
#LSM_put_bound_ISD(T, N, r, S, sigma, Omega, K)

BM_alpha_05 = BM_ISD(1,50,0.06,40,0.2,10**5,0.5)
BM_alpha_25 = BM_ISD(1,50,0.06,40,0.2,10**5,25)

def LSM_ISD(T, N, r, S, K, Stock_paths, C_bound,t):

    #Recalculate the stockpaths, this is nessecary if S neq 40! Since the Stock_paths are generated from S = 40
    Stock_paths_copy = Stock_paths * (1 + (S-40)/40)

    #We want to only select the right number of timepoints!
    if t > N:
        return np.nan
    if t == 0:
        index = T * N + 1
    else:
        index = -t

    discount_rate = np.exp(-r/(N*T))
    # The stopping time for the option.
    exercise_mask = (Stock_paths_copy[:,:index] <= C_bound[t:])
    exercise_mask &= np.cumsum(exercise_mask, axis=1) == 1
    #Create cashflows for option in all paths
    CFM = np.zeros_like(Stock_paths_copy[:,:index])
    CFM[exercise_mask] = (K - Stock_paths_copy[:,:index])[exercise_mask]

    disc_vector = np.vander([discount_rate], N - t + 1, increasing=True).T

    Value_at_t0 = (CFM @ disc_vector) [:,0] #Written as Z in Stentoft

    Regressor = np.vander(Stock_paths_copy[:,0]-S, N = 9, increasing=True) #(Stock_paths[:,0]-S)), or in Stentoft X_n-x_0

    model = LinearRegression()
    model.fit(Regressor, Value_at_t0)

    price = model.intercept_
    delta = model.coef_[1]
    return price, delta, Value_at_t0

Cont_bound_alpha_25 = LSM_put_bound_ISD(1, 50, 0.06, 40, 0.2, 10**5, 40, 25)
Cont_bound_alpha_05 = LSM_put_bound_ISD(1, 50, 0.06, 40, 0.2, 10**5, 40, 0.5)

#To show the value at t=0, also figure A1 in Stentoft
Val_alpha_25 = LSM_ISD(1,50,0.06,40,40,BM_alpha_25,Cont_bound_alpha_25,0)[2]
Val_alpha_05 = LSM_ISD(1,50,0.06,40,40,BM_alpha_05,Cont_bound_alpha_05,0)[2]

plt.figure(facecolor="#fafafa") #colors outside the box
plt.grid(alpha=0.5, linestyle = '--', color = 'black', zorder = 1)
ax = plt.gca()
ax.set_facecolor("#fafafa")
ax.set_axisbelow(True)
plt.scatter(BM_alpha_25[:,0],Val_alpha_25,s = 1, color= '#d95f0e')
#plt.title('LSM ISD naive - Data in regression at t=0, alpha = 25', fontsize=36)
plt.xticks(fontsize=20)  # Enlarge tick labels on x-axis
plt.yticks(fontsize=20)
plt.xlabel("Starting stock price", fontsize=26)  # Bigger x-axis label
plt.ylabel("Value at t=0", fontsize=26)
plt.tight_layout()

plt.figure(facecolor="#fafafa") #colors outside the box
plt.grid(alpha=0.5, linestyle = '--', color = 'black', zorder = 1)
ax = plt.gca()
ax.set_facecolor("#fafafa")
ax.set_axisbelow(True)
plt.scatter(BM_alpha_05[:,0],Val_alpha_05, s = 1, color= '#d95f0e')
#plt.title('LSM ISD naive - Data in regression at t=0, alpha = 0.5', fontsize=36)
plt.xticks(fontsize=20)  # Enlarge tick labels on x-axis
plt.yticks(fontsize=20)
plt.xlabel("Starting stock price", fontsize=26)  # Bigger x-axis label
plt.ylabel("Value at t=0", fontsize=26)
plt.ylim((-0.2,10.2))
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

price = CRR_put_Greeks(T0,10**4,r0,S0,sigma0,K0)[0]
#finding the continuation/stopping region for the stock, so we are able to find when the option is exercised
Cont_bound_a05 = LSM_put_bound_ISD(1, 50 , 0.06, 40, 0.2, 10**5, 40,0.5)
Cont_bound_a25 = LSM_put_bound_ISD(1, 50 , 0.06, 40, 0.2, 10**5, 40,25)

#For the greeks interpolation function: Note that we need to change the -10 to -30 for EU options
s0 = np.linspace(K0 - 10, K0 + 20, n0 + 1)

time_grid = np.linspace(0, T0 * N0 - 1, T0 * N0, dtype=int)
#Creating delta matrix from above information
f_s_t_alpha_25 = np.array([[LSM_ISD(T0,N0,r0,s,40,BM_alpha_25,Cont_bound_a25,t)[1] for s in s0] for t in time_grid]).T
f_s_t_alpha_05 = np.array([[LSM_ISD(T0,N0,r0,s,40,BM_alpha_05,Cont_bound_a05,t)[1] for s in s0] for t in time_grid]).T

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

#run code above to get hedging errors for alpha = 0.5 and 25, and store these two seperatly!
disc_hedge_errors_25 = ...
disc_hedge_errors_05 = ...

plt.figure(facecolor="#fafafa") #colors outside the box
plt.grid(alpha=0.5, linestyle = '--', color = 'black', zorder = 1)
ax = plt.gca()
ax.set_facecolor("#fafafa")
ax.set_axisbelow(True)
plt.hist(disc_hedge_errors_25, bins=40, density=True, edgecolor='black', color='#e6d7c3')
#plt.title('LSM ISD (naive), alpha = 25 - Hedging error', fontsize=36)
sns.kdeplot(disc_hedge_errors_25, color='tan', linewidth=2)
plt.xticks(fontsize=20)  # Enlarge tick labels on x-axis
plt.yticks(fontsize=20)
plt.xlim(-2.5, 2.5)
plt.ylabel("")
plt.tight_layout()

print(
np.mean(disc_hedge_errors_25),
np.var(disc_hedge_errors_25),
np.min(disc_hedge_errors_25),
np.max(disc_hedge_errors_25)
)

plt.figure(facecolor="#fafafa") #colors outside the box
plt.grid(alpha=0.5, linestyle = '--', color = 'black', zorder = 1)
ax = plt.gca()
ax.set_facecolor("#fafafa")
ax.set_axisbelow(True)
plt.hist(disc_hedge_errors_05, bins=40, density=True, edgecolor='black', color='#e6d7c3')
#plt.title('LSM ISD (naive), alpha = 0.5 - Hedging error', fontsize=36)
sns.kdeplot(disc_hedge_errors_05, color='tan', linewidth=2)
plt.xticks(fontsize=20)  # Enlarge tick labels on x-axis
plt.yticks(fontsize=20)
plt.xlim(-2.5, 2.5)
plt.ylabel("")
plt.tight_layout()

print(
np.mean(disc_hedge_errors_05),
np.var(disc_hedge_errors_05),
np.min(disc_hedge_errors_05),
np.max(disc_hedge_errors_05)
)
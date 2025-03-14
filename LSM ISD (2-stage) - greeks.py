import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from BM import *
from Interpolation import *

BM_alpha_05 = BM_ISD(1,50,0.06,40,0.2,10**5,0.5)
BM_alpha_5 = BM_ISD(1,50,0.06,40,0.2,10**5,5)
BM_alpha_10 = BM_ISD(1,50,0.06,40,0.2,10**5,10)
BM_alpha_25 = BM_ISD(1,50,0.06,40,0.2,10**5,25)


Cont_bound_a05 = LSM_put_bound_ISD(1, 50 , 0.06, 40, 0.2, 10**5, 40,0.5)
Cont_bound_a5 = LSM_put_bound_ISD(1, 50 , 0.06, 40, 0.2, 10**5, 40,5)
Cont_bound_a10 = LSM_put_bound_ISD(1, 50 , 0.06, 40, 0.2, 10**5, 40,10)
Cont_bound_a25 = LSM_put_bound_ISD(1, 50 , 0.06, 40, 0.2, 10**5, 40,25)


def LSM_ISD_2_pre(T, N, r, S, sigma, K, Stock_paths, C_bound, alpha, alpha_opt):

    Stock_paths_trunc = np.zeros_like(Stock_paths)
    Truncate = (Stock_paths[:,0] < S - alpha_opt) | (Stock_paths[:,0] > S + alpha_opt)
    Stock_paths_trunc[Truncate] = BM_ISD_2(T, N, r, S, sigma, np.sum(Truncate), alpha, alpha_opt)
    Stock_paths_trunc[~Truncate] = Stock_paths[~Truncate]

    discount_rate = np.exp(-r/(N*T))
    # The stopping time for the option.
    exercise_mask = (Stock_paths_trunc[:,2:] <= C_bound[2:])
    exercise_mask &= np.cumsum(exercise_mask, axis=1) == 1
    #Create cashflows for option in all paths
    CFM = np.zeros_like(Stock_paths_trunc[:,2:])
    CFM[exercise_mask] = (K - Stock_paths_trunc[:,2:])[exercise_mask]

    disc_vector = (np.vander([discount_rate], N, increasing=True).T)[1:,:]

    x = np.vander(np.linspace(10,60,200), N=9, increasing=True)[:,1:]

    Value_at_t1 = (CFM @ disc_vector) [:,0] #Written as Z in Stentoft
    Value_at_t1_ITM = Value_at_t1[Value_at_t1>0]
    Regressor_at_t1_ITM = np.vander(Stock_paths_trunc[Value_at_t1>0,1], N=9, increasing=True)[:,1:]
    Regressor_at_t1 = np.vander(Stock_paths_trunc[:, 1], N=9, increasing=True)[:, 1:]

    model = LinearRegression()
    model.fit(Regressor_at_t1_ITM, Value_at_t1_ITM)
    y_ITM = model.predict(x)

    model = LinearRegression()
    model.fit(Regressor_at_t1, Value_at_t1)
    Predicted_value = model.predict(np.vander(Stock_paths_trunc[:,1], N=9, increasing=True)[:,1:])
    y_ALL = model.predict(x)

    Value_at_t0 = np.maximum(Predicted_value, np.maximum(K-Stock_paths_trunc[:,1],0)) * discount_rate #Written as Z in Stentoft

    return Value_at_t0, Value_at_t1, y_ITM, y_ALL

#To show the value at t=0, also figure A1 in Stentoft
Val_alpha_25 = LSM_ISD_2_pre(1,50,0.06,40,0.2,40,BM_alpha_25,Cont_bound_a25,25,5)[0]
Val_alpha_05 = LSM_ISD_2_pre(1,50,0.06,40,0.2,40,BM_alpha_05,Cont_bound_a05,25,5)[0]

plt.figure(facecolor="#fafafa") #colors outside the box
plt.grid(alpha=0.5, linestyle = '--', color = 'black', zorder = 1)
ax = plt.gca()
ax.set_facecolor("#fafafa")
ax.set_axisbelow(True)
plt.scatter(BM_alpha_25[:,0],Val_alpha_25,s = 1, color= '#d95f0e')
#plt.title('LSM ISD 2 stage - Data in regression at t=0, alpha = 25', fontsize=36)
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
#plt.title('LSM ISD 2 stage - Data in regression at t=0, alpha = 0.5', fontsize=36)
plt.xticks(fontsize=20)  # Enlarge tick labels on x-axis
plt.yticks(fontsize=20)
plt.xlabel("Starting stock price", fontsize=26)  # Bigger x-axis label
plt.ylabel("Value at t=0", fontsize=26)
plt.ylim((-0.2,10.2))
plt.tight_layout()

#Data available for preliminary regression
pre_Val_ITM = LSM_ISD_2_pre(1,50,0.06,40,0.2,40,BM_alpha_25,Cont_bound_a25,25,25)[1]
ITM = pre_Val_ITM>0
pre_Val_ITM = pre_Val_ITM[ITM]
pre_Val = LSM_ISD_2_pre(1,50,0.06,40,0.2,40,BM_alpha_25,Cont_bound_a25,25,25)[1]
y_pre_val_ITM = LSM_ISD_2_pre(1,50,0.06,40,0.2,40,BM_alpha_25,Cont_bound_a25,25,25)[2]
y_pre_val = LSM_ISD_2_pre(1,50,0.06,40,0.2,40,BM_alpha_25,Cont_bound_a25,25,25)[3]

plt.figure(facecolor="#fafafa") #colors outside the box
plt.grid(alpha=0.5, linestyle = '--', color = 'black', zorder = 1)
ax = plt.gca()
ax.set_facecolor("#fafafa")
ax.set_axisbelow(True)
plt.scatter(BM_alpha_25[:,1][ITM],pre_Val_ITM,s = 1, color= '#d95f0e')
plt.plot(np.linspace(10,60,200),y_pre_val_ITM, linewidth = 4, label = 'Regression function', color = 'black')
#plt.title('LSM ISD 2 stage - Data in regression at t=0, alpha = 25', fontsize=36)
plt.xticks(fontsize=20)  # Enlarge tick labels on x-axis
plt.yticks(fontsize=20)
plt.xlabel("Starting stock price", fontsize=26)  # Bigger x-axis label
plt.ylabel("Value at t=dt", fontsize=26)
plt.legend(fontsize = 26)
plt.ylim((-2.5,27.5))
plt.xlim((12.5,67.5))
plt.tight_layout()

plt.figure(facecolor="#fafafa") #colors outside the box
plt.grid(alpha=0.5, linestyle = '--', color = 'black', zorder = 1)
ax = plt.gca()
ax.set_facecolor("#fafafa")
ax.set_axisbelow(True)
plt.scatter(BM_alpha_25[:,1],pre_Val,s = 1, color= '#d95f0e')
plt.plot(np.linspace(10,60,200),y_pre_val, linewidth = 4, label = 'Regression function', color = 'black')
#plt.title('LSM ISD 2 stage - Data in regression at t=0, alpha = 25', fontsize=36)
plt.xticks(fontsize=20)  # Enlarge tick labels on x-axis
plt.yticks(fontsize=20)
plt.xlabel("Starting stock price", fontsize=26)  # Bigger x-axis label
plt.ylabel("Value at t=dt", fontsize=26)
plt.ylim((-2.5,27.5))
plt.xlim((12.5,67.5))
plt.tight_layout()

#Now we continue to the hedging part, where we redefine the LSM ISD since multiple lines are obsolete

def LSM_ISD_2(T, N, r, S, sigma, K, Stock_paths, C_bound,t,alpha,alpha_opt):

    if t == 0:
        index = T * N + 1
    else:
        index = -t

    Stock_paths_copy = Stock_paths * (1 + (S - 40) / 40)

    Stock_paths_trunc = np.zeros_like(Stock_paths_copy)
    Truncate = (Stock_paths_copy[:,0] < S - alpha_opt) | (Stock_paths_copy[:,0] > S + alpha_opt)
    Stock_paths_trunc[Truncate] = BM_ISD_2(T, N, r, S, sigma, np.sum(Truncate), alpha, alpha_opt)
    Stock_paths_trunc[~Truncate] = Stock_paths_copy[~Truncate]

    discount_rate = np.exp(-r/(N*T))
    # The stopping time for the option.
    exercise_mask = (Stock_paths_trunc[:,2:index] <= C_bound[2+t:])
    exercise_mask &= np.cumsum(exercise_mask, axis=1) == 1
    #Create cashflows for option in all paths
    CFM = np.zeros_like(Stock_paths_trunc[:,2:index])
    CFM[exercise_mask] = (K - Stock_paths_trunc[:,2:index])[exercise_mask]

    disc_vector = (np.vander([discount_rate], N-t, increasing=True).T)[1:,:]

    Value_at_t1 = (CFM @ disc_vector) [:,0] #Written as Z in Stentoft
    Regressor_at_t1 = np.vander(Stock_paths_trunc[:,1], N=9, increasing=True)[:,1:]

    model = LinearRegression()
    model.fit(Regressor_at_t1, Value_at_t1)

    Predicted_value = model.predict(Regressor_at_t1)

    Value_at_t0 = np.maximum(Predicted_value, np.maximum(K-Stock_paths_trunc[:,1],0)) * discount_rate #Written as Z in Stentoft

    Regressor = np.vander(Stock_paths_trunc[:,0]-S, N = 9, increasing=True)[:,1:] #(Stock_paths[:,0]-S)), or in Stentoft X_n-x_0

    model = LinearRegression()
    model.fit(Regressor, Value_at_t0)

    price = model.intercept_
    delta = model.coef_[0]
    return price, delta

#Research for optimal choice of alpha
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

#For the greeks interpolation function: Note that we need to change the -10 to -30 for EU options
s0 = np.linspace(K0 - 10, K0 + 20, n0 + 1)

time_grid = np.linspace(0, T0 * N0 - 1, T0 * N0, dtype=int)
#Creating delta matrix from above information

f_s_t_alpha_25_2 = np.array([[LSM_ISD_2(T0,N0,r0,s,sigma0,40,BM_alpha_25,Cont_bound_a25,t,25,25)[1] for s in s0] for t in time_grid]).T
f_s_t_alpha_10_2 = np.array([[LSM_ISD_2(T0,N0,r0,s,sigma0,40,BM_alpha_10,Cont_bound_a10,t,10,25)[1] for s in s0] for t in time_grid]).T
f_s_t_alpha_5_2  = np.array([[LSM_ISD_2(T0,N0,r0,s,sigma0,40,BM_alpha_5,Cont_bound_a5,t,5,25)[1] for s in s0] for t in time_grid]).T
f_s_t_alpha_05_2 = np.array([[LSM_ISD_2(T0,N0,r0,s,sigma0,40,BM_alpha_05,Cont_bound_a05,t,0.5,25)[1] for s in s0] for t in time_grid]).T

#Follow the code below, but changes in Cont_bound, f_s_t

f_s_t = #

Cont_bound = #

Stock_paths = BM(T0,N0,r0,S0, sigma0,Omega0)

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

#Save the different start values of f_s_t and BM and Cont_bound as
disc_hedge_errors_25_2 = ...
disc_hedge_errors_10_2 = ...
disc_hedge_errors_5_2  = ...
disc_hedge_errors_05_2 = ...

plt.figure(facecolor="#fafafa") #colors outside the box
plt.grid(alpha=0.5, linestyle = '--', color = 'black', zorder = 1)
ax = plt.gca()
ax.set_facecolor("#fafafa")
ax.set_axisbelow(True)
sns.kdeplot(disc_hedge_errors_25_2, color='#d95f0e', linewidth=4, label = 'alpha = 25')
sns.kdeplot(disc_hedge_errors_10_2, color='pink', linewidth=4, label = 'alpha = 10')
sns.kdeplot(disc_hedge_errors_5_2, color='grey', linewidth=4, label = 'alpha = 5', linestyle = '--')
sns.kdeplot(disc_hedge_errors_05_2, color='tan', linewidth=4, label = 'alpha = 0.5')
plt.xticks(fontsize=20)  # Enlarge tick labels on x-axis
plt.yticks(fontsize=20)
plt.legend(fontsize=26)
plt.xlim(-5, 5)
plt.ylabel("")
plt.tight_layout()

#Concluding that optimal alpha is 5 and now we can continue to hedge and so on,
#hence the difference in the f_s_t below is the optimal alpha

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

#For the greeks interpolation function: Note that we need to change the -10 to -30 for EU options
s0 = np.linspace(K0 - 10, K0 + 20, n0 + 1)

time_grid = np.linspace(0, T0 * N0 - 1, T0 * N0, dtype=int)
#Creating delta matrix from above information

f_s_t_alpha_25 = np.array([[LSM_ISD_2(T0,N0,r0,s,sigma0,40,BM_alpha_25,Cont_bound_a25,t,25,5)[1] for s in s0] for t in time_grid]).T
f_s_t_alpha_05 = np.array([[LSM_ISD_2(T0,N0,r0,s,sigma0,40,BM_alpha_05,Cont_bound_a05,t,0.5,5)[1] for s in s0] for t in time_grid]).T

#Choose the different cont_bounds respectivly
Cont_bound = LSM_put_bound(T0,N0,r0,S0,sigma0,Omega0,K0)
f_s_t = ...

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

#Run code above for the two different values of alpha

disc_hedge_errors_25_2 = ...
disc_hedge_errors_05_2 = ...

#Hedge error
plt.figure(facecolor="#fafafa") #colors outside the box
plt.grid(alpha=0.5, linestyle = '--', color = 'black', zorder = 1)
ax = plt.gca()
ax.set_facecolor("#fafafa")
ax.set_axisbelow(True)
plt.hist(disc_hedge_errors_25_2, bins=40, density=True, edgecolor='black', color='#e6d7c3')
#plt.title('LSM ISD 2 stage, alpha 25 - Hedging error', fontsize=36)
sns.kdeplot(disc_hedge_errors_25_2, color='tan', linewidth=2)
plt.xticks(fontsize=20)  # Enlarge tick labels on x-axis
plt.yticks(fontsize=20)
plt.xlim(-2.5, 2.5)
plt.ylabel("")
plt.tight_layout()

print(
np.mean(disc_hedge_errors_25_2),
np.var(disc_hedge_errors_25_2),
np.min(disc_hedge_errors_25_2),
np.max(disc_hedge_errors_25_2)
)

#Hedge error
plt.figure(facecolor="#fafafa") #colors outside the box
plt.grid(alpha=0.5, linestyle = '--', color = 'black', zorder = 1)
ax = plt.gca()
ax.set_facecolor("#fafafa")
ax.set_axisbelow(True)
plt.hist(disc_hedge_errors_05_2, bins=40, density=True, edgecolor='black', color='#e6d7c3')
#plt.title('LSM ISD 2 stage, alpha 0.5 - Hedging error', fontsize=36)
sns.kdeplot(disc_hedge_errors_05_2, color='tan', linewidth=2)
plt.xticks(fontsize=20)  # Enlarge tick labels on x-axis
plt.yticks(fontsize=20)
plt.xlim(-2.5, 2.5)
plt.ylabel("")
plt.tight_layout()

print(
np.mean(disc_hedge_errors_05_2),
np.var(disc_hedge_errors_05_2),
np.min(disc_hedge_errors_05_2),
np.max(disc_hedge_errors_05_2)
)

#We wish to compare if choosing the optimal stopping boundary better has an influence on the hedging error:
#We will use the hedge errors for alpha=25 as before, but lets run a new hedge ekspirement where we only change the cont_bound
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


#We
Cont_bound = CRR_put_bound(T0,10**5,r0,S0,sigma0,K0)[0::999]
f_s_t = f_s_t_alpha_25_2

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

disc_hedge_errors_25_2_cbound = (hedge_errors @ disc_vector)[:,0]

#here we change number of interpolation points to 50
n0 = 50
s0 = np.linspace(K0 - 10, K0 + 20, n0 + 1)

f_s_t = np.array([[LSM_ISD_2(T0,N0,r0,s,sigma0,40,BM_alpha_25,Cont_bound_a25,t,25,5)[1] for s in s0] for t in time_grid]).T

Cont_bound = LSM_put_bound_ISD(1, 50 , 0.06, 40, 0.2, 10**5, 40,25)

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

disc_hedge_errors_25_2_n50 = (hedge_errors @ disc_vector)[:,0]

plt.figure(facecolor="#fafafa") #colors outside the box
plt.grid(alpha=0.5, linestyle = '--', color = 'black', zorder = 1)
ax = plt.gca()
ax.set_facecolor("#fafafa")
ax.set_axisbelow(True)
sns.kdeplot(disc_hedge_errors_25_2, color='tan', linewidth=7, label = 'Standard')
sns.kdeplot(disc_hedge_errors_25_2_cbound, color='grey', linestyle = '--', linewidth=6, label = 'Benchmark boundary')
sns.kdeplot(disc_hedge_errors_25_2_n50, color='#d95f0e', linestyle = ':', linewidth=5, label = '50 interpolation points')
#plt.title('Changes in optimal stopping boundary', fontsize=36)
plt.xticks(fontsize=20)  # Enlarge tick labels on x-axis
plt.yticks(fontsize=20)
plt.legend(fontsize=26)
plt.xlim(-5, 5)
plt.ylabel("")
plt.tight_layout()
import numpy as np
from sklearn.linear_model import LinearRegression
import time
import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame
from itertools import product

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

    return(val[0], delta[0])

def BM(T, N, r, S, sigma, Omega):
    dt = 1 / N
    Z = np.random.normal(0, np.sqrt(dt), size=(Omega, int(T * N)))
    x = np.zeros((Omega, int(T * N) + 1))
    x[:, 0] = S
    x[:, 1:] = S * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * dt + sigma * Z, axis=1))
    return x

# Kernel function
def Kernel(x):
    return 2 * np.sin(np.arcsin(2 * x - 1) / 3)

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

def LSM_put_bound_ISD(T, N, r, S, sigma, Omega, K, alpha):
    M = 9
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
        Z = np.vander(X, M ,increasing=True)[:,1:]

        model = LinearRegression()
        model.fit(Z, Y)

        Q = np.vander(data[:, i - 1],M,increasing=True)[:,1:]

        E = model.predict(Q)

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

    cont_bound.append(K)
    return cont_bound

def LSM_ISD(T, N, r, S, K, Stock_paths, C_bound):

    discount_rate = np.exp(-r/(N*T))
    # The stopping time for the option.
    exercise_mask = (Stock_paths[:,] <= C_bound)
    exercise_mask &= np.cumsum(exercise_mask, axis=1) == 1
    #Create cashflows for option in all paths
    CFM = np.zeros_like(Stock_paths[:,])
    CFM[exercise_mask] = (K - Stock_paths[:,])[exercise_mask]

    disc_vector = np.vander([discount_rate], N + 1, increasing=True).T

    Value_at_t0 = (CFM @ disc_vector) [:,0] #Written as Z in Stentoft

    Regressor = np.vander(Stock_paths[:,0]-S, N = 9, increasing=True) #(Stock_paths[:,0]-S)), or in Stentoft X_n-x_0

    model = LinearRegression()
    model.fit(Regressor, Value_at_t0)

    price = model.intercept_
    delta = model.coef_[1]
    return price, delta

def LSM_ISD_2(T, N, r, S, sigma, K, Stock_paths, C_bound, alpha, alpha_opt):

    discount_rate = np.exp(-r/(N*T))
    # The stopping time for the option.
    Stock_paths_trunc = np.zeros_like(Stock_paths)
    Truncate = (Stock_paths[:,0] < S - alpha_opt) | (Stock_paths[:,0] > S + alpha_opt)
    Stock_paths_trunc[Truncate] = BM_ISD_2(T, N, r, S, sigma, np.sum(Truncate), alpha, alpha_opt)
    Stock_paths_trunc[~Truncate] = Stock_paths[~Truncate]


    exercise_mask = (Stock_paths_trunc[:,2:] <= C_bound[2:])
    exercise_mask &= np.cumsum(exercise_mask, axis=1) == 1
    #Create cashflows for option in all paths
    CFM = np.zeros_like(Stock_paths_trunc[:,2:])
    CFM[exercise_mask] = (K - Stock_paths_trunc[:,2:])[exercise_mask]

    disc_vector = (np.vander([discount_rate], N, increasing=True).T)[1:,:]

    Value_at_t1 = (CFM @ disc_vector) [:,0] #Written as Z in Stentoft
    Value_at_t1_ITM = Value_at_t1
    Regressor_at_t1 = np.vander(Stock_paths_trunc[:,1], N=9, increasing=True)[:,1:]

    model = LinearRegression()
    model.fit(Regressor_at_t1, Value_at_t1_ITM)

    Predicted_value = model.predict(np.vander(Stock_paths_trunc[:,1], N=9, increasing=True)[:,1:])

    Value_at_t0 = np.maximum(Predicted_value, np.maximum(K-Stock_paths_trunc[:,1],0)) * discount_rate #Written as Z in Stentoft

    Regressor = np.vander(Stock_paths_trunc[:,0]-S, N = 9, increasing=True)[:,1:] #(Stock_paths[:,0]-S)), or in Stentoft X_n-x_0

    model = LinearRegression()
    model.fit(Regressor, Value_at_t0)

    price = model.intercept_
    delta = model.coef_[0]
    return price, delta

S0 = 40
sigma0 = 0.2
T0 = 1
K_values = [36,40,44]
r0 = 0.06
N0 = 50
Omega0 = 100000
alpha_opt0 = 5
alpha_values = [0.5,5,25]

combinations = product(alpha_values, K_values)
results_ISD_naive = []
results_ISD_2 = []

for alpha0, K0 in combinations:
    Cont_bound_K0 = LSM_put_bound_ISD(T0, N0, r0, S0, sigma0, 10**5, K0,alpha0)
    CRR = CRR_put_Greeks(T0,10**4,r0,S0,sigma0,K0)
    CRR_p = CRR[0]
    CRR_d = CRR[1]
    LSM_ISD0 = [LSM_ISD(T0,N0,r0,S0,K0,BM_ISD(T0,N0,r0,S0,sigma0,Omega0,alpha0),Cont_bound_K0) for i in range(10**2)]
    LSM_ISD_p = np.array(LSM_ISD0)[:,0]
    LSM_ISD_p_mean = np.mean(LSM_ISD_p)
    LSM_ISD_p_se = np.std(LSM_ISD_p-CRR_p)
    LSM_ISD_p_CI = abs(LSM_ISD_p_mean-CRR_p)/(LSM_ISD_p_se) > 2.626
    LSM_ISD_d = np.array(LSM_ISD0)[:,1]
    LSM_ISD_d_mean = np.mean(LSM_ISD_d)
    LSM_ISD_d_se = np.std(LSM_ISD_d)
    LSM_ISD_d_CI = abs(LSM_ISD_d_mean-CRR_d)/(LSM_ISD_d_se) > 2.626

    results_ISD_naive.append((K0, "%.1f"%alpha0, "%.3f"%CRR_p, "%.3f"%LSM_ISD_p_mean, f'({"%.3f"%LSM_ISD_p_se})', LSM_ISD_p_CI, "%.3f"%CRR_d, "%.3f"%LSM_ISD_d_mean, f'({"%.3f"%LSM_ISD_d_se})', LSM_ISD_d_CI))

combinations = product(alpha_values, K_values)
for alpha0, K0 in combinations:
    Cont_bound_alpha = LSM_put_bound_ISD(T0, N0, r0, S0, sigma0, 10**5, K0,alpha0)
    CRR = CRR_put_Greeks(T0,10**4,r0,S0,sigma0,K0)
    CRR_p = CRR[0]
    CRR_d = CRR[1]
    LSM_ISD0 = [LSM_ISD_2(T0,N0,r0,S0,sigma0,K0,BM_ISD(T0,N0,r0,S0,sigma0,Omega0,alpha0),Cont_bound_alpha,alpha0,alpha_opt0) for i in range(10**2)]
    LSM_ISD_p = np.array(LSM_ISD0)[:,0]
    LSM_ISD_p_mean = np.mean(LSM_ISD_p)
    LSM_ISD_p_se = np.std(LSM_ISD_p)
    LSM_ISD_p_CI = abs(LSM_ISD_p_mean-CRR_p)/(LSM_ISD_p_se) > 2.626
    LSM_ISD_d = np.array(LSM_ISD0)[:,1]
    LSM_ISD_d_mean = np.mean(LSM_ISD_d)
    LSM_ISD_d_se = np.std(LSM_ISD_d)
    LSM_ISD_d_CI = abs(LSM_ISD_d_mean-CRR_d)/(LSM_ISD_d_se) > 2.626

    results_ISD_2.append((K0, "%.1f"%alpha0, "%.3f"%CRR_p, "%.3f"%LSM_ISD_p_mean, f'({"%.3f"%LSM_ISD_p_se})', LSM_ISD_p_CI, "%.3f"%CRR_d, "%.3f"%LSM_ISD_d_mean, f'({"%.3f"%LSM_ISD_d_se})', LSM_ISD_d_CI))

df_naive = pd.DataFrame(results_ISD_naive, columns=['K', 'alpha', 'BM', 'LSM ISD', 's.e.','significant', 'BM', 'LSM ISD', 's.e.', 'significant'])
latex_table_naive = df_naive.to_latex(index=False, column_format='|c|c|c|c|c|c|c|c|c|')

df_2 = pd.DataFrame(results_ISD_2, columns=['K', 'alpha', 'BM', 'LSM ISD', 's.e.','significant', 'BM', 'LSM ISD', 's.e.', 'significant'])
latex_table_2 = df_2.to_latex(index=False, column_format='|c|c|c|c|c|c|c|c|c|')

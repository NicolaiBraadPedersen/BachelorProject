import numpy as np
from scipy.interpolate import RegularGridInterpolator

def Delta_Interpol(dlt, s0, time_grid, T, N, K, Omega, S):
    #Interpolate, values of already found delta
    interp_f = RegularGridInterpolator((s0, time_grid), dlt, method='linear', bounds_error=True, fill_value=None)
    #defining empty/basic arrays
    neg_one_vec = np.zeros(Omega) - 1
    Delta_M = np.zeros_like(S)

    for i in range(0, T * N):
        t = time_grid[i]  # Current time
        #defining if the option is within a reasonable bound

        mask_in_range = (S[:, i] >= K - 10) & (S[:, i] <= K + 20) #change here for -30 if we have EU option
        mask_notin_range = (S[:, i] < K - 10) #change here for -30 if we have EU option

        # Default to -1 for out-of-range stock prices (too negative). For too large we let delta be 0
        if S[mask_notin_range, i].size > 0:
            Delta_M[mask_notin_range, i] = neg_one_vec[mask_notin_range]

        # Calculate Delta for in-range stock prices using the interpolation function
        S_in_range = S[mask_in_range, i]
        points = np.array([S_in_range, np.full_like(S_in_range, t)]).T
        Delta_M[mask_in_range, i] = interp_f(points)
    return Delta_M

def Delta_Interpol_EU(dlt, s0, time_grid, T, N, K, Omega, S):
    #Interpolate, values of already found delta
    interp_f = RegularGridInterpolator((s0, time_grid), dlt, method='linear', bounds_error=True, fill_value=None)
    #defining empty/basic arrays
    neg_one_vec = np.zeros(Omega) - 1
    Delta_M = np.zeros_like(S)

    for i in range(0, T * N):
        t = time_grid[i]  # Current time
        #defining if the option is within a reasonable bound

        mask_in_range = (S[:, i] >= K - 20) & (S[:, i] <= K + 20) #change here for -30 if we have EU option
        mask_notin_range = (S[:, i] < K - 20) #change here for -30 if we have EU option

        # Default to -1 for out-of-range stock prices (too negative). For too large we let delta be 0
        if S[mask_notin_range, i].size > 0:
            Delta_M[mask_notin_range, i] = neg_one_vec[mask_notin_range]

        # Calculate Delta for in-range stock prices using the interpolation function
        S_in_range = S[mask_in_range, i]
        points = np.array([S_in_range, np.full_like(S_in_range, t)]).T
        Delta_M[mask_in_range, i] = interp_f(points)
    return Delta_M

#Used in all files that also use interpolation
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
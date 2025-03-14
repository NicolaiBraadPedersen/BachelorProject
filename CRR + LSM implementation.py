import numpy as np
from array import array
import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.linear_model import LinearRegression
from itertools import product
from scipy import stats as stats

def BS_EU_put(T, r, S, sigma, K, t):

    d1 = 1 / (sigma * np.sqrt(T-t)) * ( np.log(S / K) + (r + sigma ** 2 / 2) * (T-t) )
    d2 = d1 - sigma * np.sqrt(T - t)

    price = S * (stats.norm(0,1).cdf(d1) - 1) - np.exp(-r * (T-t)) * K * (stats.norm(0,1).cdf(d2) - 1)

    return price

def BM(T, N, r, S, sigma, Omega):
    data = np.zeros((Omega, T * N + 1))
    dt = 1 / N
    for omega in range(Omega):
        # Simulate the Brownian Motion
        Z = np.random.normal(0, np.sqrt(dt), size=T * N)
        x = np.zeros(T * N + 1)
        x[0] = S
        for i in range(1, T * N + 1):
            x[i] = x[i - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * Z[i - 1])
        data[omega, :] = x  # Store the stock's price in the data matrix
    return data

def Lag_Pol(X):
    L_0 = np.exp(-X / 2)
    L_1 = np.exp(-X / 2) * (1 - X)
    L_2 = np.exp(-X / 2) * (1 - 2 * X + X ** 2 / 2)
    return np.column_stack((L_0, L_1, L_2))

def LSM_put(T, N, r, S, sigma, Omega, K):
    dt = 1 / N
    data = BM(T, N, r, S, sigma, Omega) / K  # Normalize data
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

        Q = data[:, i - 1]
        E = a + b * Lag_Pol(Q)[:, 0] + c * Lag_Pol(Q)[:, 1] + d * Lag_Pol(Q)[:, 2]

        CFM[:, i - 1] = np.where(CFM[:, i - 1] < E, 0, CFM[:, i - 1])
        CFM[CFM[:, i - 1] > 0, i:] = 0
        A_CFM[:, i - 1] = np.where(CFM[:, i - 1] == 0, A_CFM[:, i] * disc, A_CFM[:, i - 1])

    disc_vec = np.vander([disc], T * N + 1, increasing = True).T

    disc_CashFlows = CFM @ disc_vec

    price = sum(disc_CashFlows) * K / Omega
    se = np.sqrt(sum((disc_CashFlows * K - price) ** 2)) / Omega

    return (price, se)

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


S_values = [36, 38, 40, 42, 44]
sigma_values = [0.2, 0.4]
T_values = [1, 2]
K0 = 40
r0 = 0.06
N0 = 50
Omega0 = 100000

combinations = product(S_values, sigma_values, T_values)
results = []

for S0, sigma0, T0 in combinations:
    LSM = LSM_put(T0, N0, r0, S0, sigma0, Omega0, K0)
    CRR = CRR_put(T0, 10**4, r0, S0, sigma0, K0)
    difference_AMR = CRR - LSM[0]
    EU = BS_EU_put(T0, r0, S0, sigma0, K0, 0)
    difference_EU = LSM[0] - EU


    results.append((S0, sigma0, T0, "%.3f"%EU, "%.3f"%CRR, "%.3f"%LSM[0], "%.4f"%LSM[1], "%.3f"%difference_AMR, "%.3f"%difference_EU,))

df = pd.DataFrame(results, columns=['S', 'sigma', 'T', 'European','CRR','LSM', 's.e. of LSM', 'BM deviation', 'early exercise value'])
df['sigma'] = df['sigma'].apply(lambda x: f"{x:.1f}")
latex_table = df.to_latex(index=False, column_format='ccccccccc')


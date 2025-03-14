import numpy as np
from sklearn.linear_model import LinearRegression

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

        Q = Lag_Pol(data[:, i - 1])
        E = model.predict(Q)

        CFM[:, i - 1] = np.where(CFM[:, i - 1] < E, 0, CFM[:, i - 1])
        CFM[CFM[:, i - 1] > 0, i:] = 0
        A_CFM[:, i - 1] = np.where(CFM[:, i - 1] == 0, A_CFM[:, i] * disc, A_CFM[:, i - 1])

    disc_vec = np.vander([disc], T * N + 1, increasing = True).T

    disc_CashFlows = CFM @ disc_vec

    price = sum(disc_CashFlows) * K / Omega
    se = np.sqrt(sum((disc_CashFlows * K - price) ** 2)) / Omega

    return (price, se)

def LSM_put_bound(T, N, r, S, sigma, Omega, K):
    dt = 1 / N
    data = BM(T, N, r, S, sigma, Omega) / K
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
        Z = Lag_Pol(X)

        model = LinearRegression()
        model.fit(Z, Y)

        Q = Lag_Pol(data[:, i - 1])
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

    cont_bound.append(40)
    return cont_bound

cmap = ['#d95f0e','pink','grey','tan']

BM_0 = BM(1,2000,0.06,36,0.2,4)
data = np.array(BM_0)
x = np.linspace(0,1,len(data[1]))

y1 = LSM_put_bound(1,50,0.06,36,0.2,2*10**5,40)
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
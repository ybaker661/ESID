import numpy as np
import pandas as pd


T = 288
N = 110

dp_markov = pd.read_csv (r'/Users/ybaker661/Desktop/Research_S2022/ESID/ESID_data/dispatch_MarkovDA1618.csv')
ddat = dp_markov["Discharge"].to_numpy()
pdat = dp_markov["Charge"].to_numpy()
ddat = ddat.reshape((365,T))
pdat = pdat.reshape((365,T))
price_hist = pd.read_csv("./ESID_data/price.csv")
price_hist = np.array(price_hist.RTP[:])
price_hist = price_hist.reshape((365,T))

actsum = np.sum(abs(ddat)+abs(pdat), axis = 1)
inds = np.argpartition(actsum, 110)[-(110):]


# print(np.max(psum))

# dates = np.random.choice(365,N,replace=False)

df_price = price_hist[inds, :]
df_d = ddat[inds, :]
df_p = pdat[inds, :]
print(df_price.shape)
paras = pd.DataFrame([[10, 0, 0.5, 0.9]],columns=("c1", "c2", "P", "eta"))

np.savez("Results/data1/stochastic/data", paras = paras, price = df_price, d=df_d, p=df_p, days=inds)
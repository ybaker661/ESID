import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from utils import PolytopeProjectionc1, data_generator_val
import time

#training/data params
lr = 10e-2
iter = 200
N_train = 20
N_valid = 10
T=288
bias = 0
var = 0
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
vallist = [1, 3, 8, 9.25, 10, 14.5, 15, 20, 25, 30]
resdir = "Results/stochastic_c1/"
# DATA LOADING AND PREP

# df_price = np.load("df_price.npy")
df_dp = np.load("Results/data1/stochastic/data.npz")

df_price = df_dp["price"]
print(df_price)
P1 = df_dp["p"].max()
P2 = df_dp["d"].max()
E1 = 0
E2 = 1
eta = 0.8

# add noise to data
noise_d = np.random.normal(bias, var, df_dp["d"].shape)
noise_p = np.random.normal(bias, var, df_dp["p"].shape)
d = np.clip(df_dp["d"] + noise_d, 0, P2)
p = np.clip(df_dp["p"] + noise_p, 0, P1)

price_tensor = torch.from_numpy(df_price[0:N_train])
d_tensor = torch.from_numpy(d[0:N_train])
p_tensor = torch.from_numpy(p[0:N_train])
y_tensor = tuple([d_tensor, p_tensor])


for n, curr in enumerate(vallist):
    # INITIALIZING NETWORK
    torch.manual_seed(0)
    vals = curr
    layer = PolytopeProjectionc1( curr, P1, P2, eta, T)
    opt1 = optim.Adam(layer.parameters(), lr=lr)
    # INITIALIZING RESULTS DATAFRAME
    df = pd.DataFrame(columns=("loss", "c1"))

    
    # TRAINING
    tstart = time.time()
    for ite in range(iter):
        tite = time.time()
        dp_pred = layer(price_tensor)

        loss = nn.MSELoss()(y_tensor[0], dp_pred[0]) + nn.MSELoss()(y_tensor[1], dp_pred[1])
        opt1.zero_grad()
        loss.backward()
        opt1.step()
        with torch.no_grad():
            i = 0
            for param in layer.parameters():
                if i == 2:
                    param.clamp_(0, 100)
                if i == 3:
                    param.clamp_(-100, 0)
                if i == 4:
                    param.clamp_(0.8, 1)
                i = i + 1
        # if(ite%10 == 0):
        print(ite)
        print("Loss", loss.detach())
        print("layer.c1.gradient =", layer.c1.grad, "c1 value =", layer.c1.detach().numpy())
        print("P1 value =", P1)
        print("P2 value =", P2)
        print("E1 value =", E1)
        print("E2 value =", E2)
        print("eta value =", eta)
        print(loss.detach().numpy())
        df.loc[ite] = [
            loss.detach().numpy(),
            layer.c1.detach().numpy()[0],
        ]
        print("time for one iteration: " + str(time.time() - tite) + " s")

    #SAVING RESULTs
    print("SAVING RESULTS FOR SET: ", curr)
    df.to_csv(resdir + "set_" + str(n) + ".csv")

    price_valid = df_price[100:]
    d_valid = d[100:]
    p_valid = p[100:]
    y_valid = p_valid - d_valid

    d_pred, p_pred = data_generator_val(
        layer.c1.detach().numpy()[0],
        layer.c2.detach().numpy()[0],
        upperbound_p=0.5,
        lowerbound_p=0,
        upperbound_e=layer.E1.detach().numpy()[0],
        lowerbound_e= layer.E2.detach().numpy()[0],
        efficiency=layer.eta.detach().numpy()[0],
        price_hist=price_valid,
        N=N_valid,
        T=T,
    )
    y_pred = p_pred - d_pred
    mse = np.square(y_pred - y_valid).mean()
    print("mse for validation set: ", mse)
    np.savez("result", mse = mse, d_valid = d_valid,  p_valid = p_valid, d_pred = d_pred, p_pred = p_pred, learning = df)
    tfin = time.time()
    print("total time to run " + str(iter) + " iterations: " + str((tfin-tstart)/60) + " mins")
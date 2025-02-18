import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from utils import PolytopeProjection, data_generator_val
import time

#training/data params
lr = 10e-2
iter = 55
N_train = 20
N_valid = 10
T=288
bias = 0
var = 0
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
vallist = []

vallist.append([1,	1,	0.1,	-0.1,	0.1])
vallist.append([3,	3,	0.2,	-0.2,	0.2])
vallist.append([8,	8,	0.3,	-0.3,	0.3])
vallist.append([9.25,	9.25,	0.4,	-0.4,	0.4])
vallist.append([10,	10,	0.5,	-0.5,	0.5])
vallist.append([14.5,	14.5,	0.6,	-0.6,	0.6])
vallist.append([15,	15,	0.7,	-0.7,	0.7])
vallist.append([20,	20,	0.8,	-0.8,	0.8])
vallist.append([25,	25,	0.9,	-0.9,	0.9])
vallist.append([30,	30,	1,	-1,	1])
resdir = "Results/data1/stochastic/"
# DATA LOADING AND PREP

# df_price = np.load("df_price.npy")
df_dp = np.load("Results/data1/stochastic/data.npz")

df_price = df_dp["price"]
P1 = df_dp["p"].max()
P2 = df_dp["d"].max()

# add noise to data
noise_d = np.random.normal(bias, var, df_dp["d"].shape)
noise_p = np.random.normal(bias, var, df_dp["p"].shape)
d = np.clip(df_dp["d"] + noise_d, 0, P2)
print(d)
p = np.clip(df_dp["p"] + noise_p, 0, P1)

price_tensor = torch.from_numpy(df_price[0:N_train])
d_tensor = torch.from_numpy(d[0:N_train])
p_tensor = torch.from_numpy(p[0:N_train])
y_tensor = tuple([d_tensor, p_tensor])


for n, curr in enumerate(vallist):
    # INITIALIZING NETWORK
    torch.manual_seed(0)
    vals = curr
    layer = PolytopeProjection(P1, P2, T, vals = vals)
    opt1 = optim.Adam(layer.parameters(), lr=lr)
    # INITIALIZING RESULTS DATAFRAME
    df = pd.DataFrame(columns=("loss", "c1", "c2", "E1", "E2", "eta"))

    
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
        print("layer.c1.gradient =", layer.c1.grad, "c1 value =", layer.c1.detach().numpy(), "truth =", df_dp['paras'][0][0])
        print("layer.c2.gradient =", layer.c2.grad, "c2 value =", layer.c2.detach().numpy(), "truth =", df_dp['paras'][0][1])
        print("layer.E1.gradient =", layer.E1.grad, "E1 value =", layer.E1.detach().numpy(), "truth =", 0.25*df_dp['paras'][0][2])
        print("layer.E2.gradient =", layer.E2.grad, "E2 value =", layer.E2.detach().numpy(), "truth =", -0.25*df_dp['paras'][0][2])
        print(
            "layer.eta.gradient =",
            layer.eta.grad,
            "eta value =",
            layer.eta.detach().numpy(), "truth =", df_dp['paras'][0][3])
        df.loc[ite] = [
            loss.detach().numpy(),
            layer.c1.detach().numpy()[0],
            layer.c2.detach().numpy()[0],
            layer.E1.detach().numpy()[0],
            layer.E2.detach().numpy()[0],
            layer.eta.detach().numpy()[0],
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
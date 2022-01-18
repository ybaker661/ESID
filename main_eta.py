import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
from utils import PolytopeProjectionETA

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

df_price = np.load("df_price.npy")
df_dp = np.load("df_dp.npz")
eta = 0.9
P1 = df_dp["p"].max()
P2 = df_dp["d"].max()
E = np.zeros(df_price.shape)
for i in range(E.shape[0]):
    E[i] = (np.tril(np.ones((288, 288))) * df_dp["p"][i] * eta).sum(axis=1) - (
        np.tril(np.ones((288, 288))) * df_dp["d"][i] / eta
    ).sum(axis=1)
E1 = E.max()
E2 = E.min()

price_tensor = torch.from_numpy(df_price)
d_tensor = torch.from_numpy(df_dp["d"])
p_tensor = torch.from_numpy(df_dp["p"])
y_tensor = tuple([d_tensor, p_tensor])

torch.manual_seed(0)
layer = PolytopeProjectionETA(P1, P2, E1, E2, eta, T=288)
opt1 = optim.Adam(layer.parameters(), lr=5e-1)

for ite in range(2000):
    if ite == 0:
        opt1.param_groups[0]["lr"] = 1e-1

    dp_pred = layer(price_tensor)

    loss = nn.MSELoss()(y_tensor[0], dp_pred[0]) + nn.MSELoss()(y_tensor[1], dp_pred[1])
    opt1.zero_grad()
    loss.backward()
    opt1.step()

    # if(ite%10 == 0):
    print(ite)
    print("Loss", loss.detach())
    print("layer.c1.gradient =", layer.c1.grad, "c1 value =", layer.c1.detach().numpy())
    print("layer.c2.gradient =", layer.c2.grad, "c2 value =", layer.c2.detach().numpy())
    print("P1 value =", P1)
    print("P2 value =", P2)
    print("E1 value =", E1)
    print("E2 value =", E2)
    print("eta value =", eta)


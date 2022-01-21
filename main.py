import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
import pandas as pd
from utils import PolytopeProjection

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

df_price = np.load("df_price.npy")
df_dp = np.load("df_dp.npz")
P1 = df_dp["p"].max()
P2 = df_dp["d"].max()

price_tensor = torch.from_numpy(df_price)
d_tensor = torch.from_numpy(df_dp["d"])
p_tensor = torch.from_numpy(df_dp["p"])
y_tensor = tuple([d_tensor, p_tensor])

torch.manual_seed(0)
layer = PolytopeProjection(P1, P2, T=288)
opt1 = optim.Adam(layer.parameters(), lr=5e-1)

df = pd.DataFrame(columns=("loss", "c1", "c2", "E1", "E2", "eta"))

for ite in range(1000):
    if ite == 0:
        opt1.param_groups[0]["lr"] = 1e-1

    if ite == 200:
        opt1.param_groups[0]["lr"] = 1e-2

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
    print("layer.E1.gradient =", layer.E1.grad, "E1 value =", layer.E1.detach().numpy())
    print("layer.E2.gradient =", layer.E2.grad, "E2 value =", layer.E2.detach().numpy())
    print(
        "layer.eta.gradient =",
        layer.eta.grad,
        "eta value =",
        layer.eta.detach().numpy(),
    )
    df.loc[ite] = [
        loss.detach().numpy(),
        layer.c1.detach().numpy()[0],
        layer.c2.detach().numpy()[0],
        layer.E1.detach().numpy()[0],
        layer.E2.detach().numpy()[0],
        layer.eta.detach().numpy()[0],
    ]

df.to_csv("converge_test.csv")

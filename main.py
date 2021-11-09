import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
from utils import PolytopeProjection
use_cuda = torch.cuda.is_available() 
device   = torch.device("cuda" if use_cuda else "cpu")

df_price = np.load('df_price.npy')
df_dp = np.load('df_dp.npz')

price_tensor = torch.from_numpy(df_price)
d_tensor = torch.from_numpy(df_dp['d'])
p_tensor = torch.from_numpy(df_dp['p'])
y_tensor = tuple([d_tensor,p_tensor])

torch.manual_seed(0)
layer = PolytopeProjection(T = 288)
opt1 = optim.Adam(layer.parameters(), lr=5e-1)

## gradcheck
# price = torch.from_numpy(df_price)[0]
# torch_params = [torch.stack([price,price]),(10+0.01*torch.randn(2, 1, dtype=torch.double)).requires_grad_(),
#                 (100+0.01*torch.randn(2, 1, dtype=torch.double)).requires_grad_(),(0.5/12+0.01*torch.rand(2, 1, dtype=torch.double)).requires_grad_(),
#                 (0.01*torch.rand(2, 1, dtype=torch.double)).requires_grad_(),(1+0.01*torch.rand(2, 1, dtype=torch.double)).requires_grad_(),
#                 (0.01*torch.rand(2, 1, dtype=torch.double)).requires_grad_(),(0.5+0.01*torch.rand(2, 1, dtype=torch.double)).requires_grad_(),
#                 (0.9+0.01*torch.rand(2, 1, dtype=torch.double)).requires_grad_()]
# print(autograd.gradcheck(lambda *x: layer.self.layer(*x).sum(), tuple(torch_params), eps=1e-4, atol=1e-3, check_undefined_grad=False))


# print('Initial \alpha value', layer.alpha)
# print('Initial N value', layer.N)
# print('Initial M value', layer.M)

for ite in range(2000):
    if(ite == 0):
        opt1.param_groups[0]["lr"] = 1e-1

    dp_pred = layer(price_tensor)

    loss = nn.MSELoss()(y_tensor[0], dp_pred[0]) + nn.MSELoss()(y_tensor[1], dp_pred[1])
    opt1.zero_grad()
    loss.backward()
    opt1.step()
    
    # if(ite%10 == 0):
    print(ite)
    print('Loss', loss.detach())
    print('layer.c1.gradient =', layer.c1.grad, 'c1 value =', layer.c1.detach().numpy())
    print('layer.c2.gradient =', layer.c2.grad, 'c2 value =', layer.c2.detach().numpy())
    print('layer.P1.gradient =', layer.P1.grad, 'P1 value =', layer.P1.detach().numpy())
    print('layer.P2.gradient =', layer.P2.grad, 'P2 value =', layer.P2.detach().numpy())
    print('layer.E1.gradient =', layer.E1.grad, 'E1 value =', layer.E1.detach().numpy())
    print('layer.E2.gradient =', layer.E2.grad, 'E2 value =', layer.E2.detach().numpy())
    print('layer.e0.gradient =', layer.e0.grad, 'e0 value =', layer.e0.detach().numpy())
    print('layer.eta.gradient =', layer.eta.grad, 'eta value =', layer.eta.detach().numpy())
    

# print('alpha True:', alpha_value, 'Forecast:', layer.alpha.detach().numpy())
# print('N True:', bound, 'Forecast:', layer.N.detach().numpy())
# print('M True:', -bound, 'Forecast:', layer.M.detach().numpy())
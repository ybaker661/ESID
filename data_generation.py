import numpy as np
import pandas as pd
from utils import data_generator

## data dimension
N_train = 2
dim = 288
hidden_dim = 2
output_dim = 288

## initialize parameters
c1_value = np.random.uniform(10, 50)
c2_value = np.random.uniform(10, 50)
bound = np.random.uniform(0, 5)
eta = np.random.uniform(0.6, 1)
print('Generating data!', 'P1=', bound/12, 'E1=', bound*2, 'c1 =', c1_value, 'c2 =', c2_value, 'eta =', eta)

## load price data

price_hist = pd.read_csv('.\ESID_data\price.csv')

## generate dispatch data and save price
# df_price, df_dp = data_generator(c1_value=c1_value, c2_value=c2_value, 
#                                 upperbound_p=bound/12, lowerbound_p=0, 
#                                 upperbound_e=2*bound, lowerbound_e=0, 
#                                 initial_e=bound, efficiency=eta, 
#                                 price_hist = price_hist, N=N_train, T=dim)

df_price, df_d, df_p = data_generator(c1_value=10, c2_value=100, 
                                upperbound_p=0.5/12, lowerbound_p=0, 
                                upperbound_e=1, lowerbound_e=0, 
                                initial_e=0.5, efficiency=0.9, 
                                price_hist = price_hist, N=N_train, T=dim)
np.save('df_price', df_price)
np.savez('df_dp', d = df_d, p =df_p)
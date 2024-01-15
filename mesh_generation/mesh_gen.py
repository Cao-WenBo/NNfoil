# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from differ import diff2d,diff1d

def Cal_mx(x_cor): 
    x_cor = x_cor.reshape(-1)
    num = x_cor.shape[0]
    mx = np.empty([num, degree+1])
    for index in range(degree+1):
        f = np.math.factorial(degree) / (np.math.factorial(index) * np.math.factorial(degree-index))
        mx[:, index] = f * np.power(x_cor, index+n_1) * np.power(1-x_cor, degree-index+n_2)
    return mx

# input 
N = 200  # Number of grid points on the surface
M = 100 # 
R = 15  # far field radius

degree = 5; n_1 = 0.5; n_2 = 1;           #CST hyperparameters

# CST parameters
# S809
# b_up = [0.176975,0.248619,0.27815,0.363124,0.0952525,0.257714];
# b_down = [-0.128087,-0.239555,-0.426286,-0.238043,-0.0898671,0.0148932];
# # NACA0012
b_up = [0.171787,0.155338,0.161996,0.137638,0.145718,0.143815]
b_down = [-0.171787,-0.155338,-0.161996,-0.137638,-0.145718,-0.143815]
# # Circle
# b_up = [1,1,1,1,1,1]; b_down = [-1,-1,-1,-1,-1,-1]; n_2 = 0.5; 

#NACA2412
# b_up = [0.19275714, 0.2046805 , 0.21408554, 0.20480742, 0.18492655, 0.24378487]
# b_down = [-0.15055915, -0.11322028, -0.07613992, -0.12520993, -0.03790618, -0.11515926]

# %%
X, Y = np.zeros((M, N)), np.zeros((M, N))
#翼型表面
theta = np.linspace(0, 2*np.pi, N+1).reshape(-1,1); theta = theta[:-1]

x_wall = 0.5 * np.cos(theta)

mx = Cal_mx(x_wall.reshape(-1)+0.5)
y_wall = np.r_[mx[:int(N/2)]@np.array(b_up).reshape(-1,1), mx[int(N/2):]@np.array(b_down).reshape(-1,1)]

X0, Y0 = x_wall, y_wall
#远场
x_far, y_far = R * np.cos(theta), R * np.sin(theta)
X1, Y1 = x_far, y_far
#插值形成初始网格
X[0], Y[0], X[-1], Y[-1] = X0.flatten(), Y0.flatten(), X1.flatten(), Y1.flatten()
for i in range(1,M-1):
    X[i] =X[0] + i / (M-1) * (X[-1] - X[0])
    Y[i] =Y[0] + i / (M-1) * (Y[-1] - Y[0])

tem = X + 1 
epoch = 500000
nodes = [-1,0,1]
for k in range(epoch):
    alpha = diff2d(X,nodes,0) ** 2 +  diff2d(Y,nodes,0) ** 2
    beta = diff2d(X,nodes,1,1,2) * diff2d(X,nodes,0) + diff2d(Y,nodes,1,1,2) * diff2d(Y,nodes,0)
    gamma = diff2d(X,nodes,1,1,2) ** 2 +  diff2d(Y,nodes,1,1,2) ** 2
    
    dXdxi, dYdxi = diff2d(X,nodes,1,1,2), diff2d(Y,nodes,1,1,2)
    X = 0.5 * (alpha * diff2d(X,nodes,1,2,2) - 0.5 * beta * diff2d(dXdxi,nodes,0) + \
    gamma * diff2d(X,nodes,0,2) + (alpha + gamma ) * 2 * X) / (alpha + gamma)
    Y = 0.5 * (alpha * diff2d(Y,nodes,1,2,2) - 0.5 * beta * diff2d(dYdxi,nodes,0) + \
    gamma * diff2d(Y,nodes,0,2) + (alpha + gamma ) * 2 * Y) / (alpha + gamma)
    
    X[0], Y[0], X[-1], Y[-1] = X0.flatten(), Y0.flatten(), X1.flatten(), Y1.flatten()
    
    err = abs((tem - X)).max()
    if  err < 1e-6:
        break
    tem = X.copy()
    if (k % np.clip(int(epoch/100),1,1000) == 0) or (k == epoch - 1):
        print('iteration：',k,'resiudal：', err)


# %% plot
XX = np.hstack((X,X[:,[0]]))
YY = np.hstack((Y,Y[:,[0]]))
plt.plot(XX.T,YY.T,XX,YY)

# %% save mesh 
XY = np.r_[X,Y]
header = np.array([N, M])
file_name = 'NACA0012_200_100.x'
np.savetxt(file_name, header.reshape(1, -1), fmt='%d', delimiter='\t')

# 追加写入data到文件
with open(file_name, "ab") as f:  # 注意使用 "ab" 模式以追加方式打开文件
    np.savetxt(f, XY, fmt='%.16f')

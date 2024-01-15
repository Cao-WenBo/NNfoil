# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 22:16:16 2020

@author: CaoXiaoMao

This package for differential 
"""

import numpy as np


def coeff(node, order=1):
    '''
    本函数用于计算给定结点的差分系数
    node为结点相对于计算点的位置向量，如[-2,-1,0]
    order为所求导数的阶数(默认为1)
    '''
    node = np.array(node)
    m = node.size

    factor, pownode = np.ones(m), np.ones((m, m))
    b = np.zeros((m, 1))
    b[order] = 1

    for i in range(1, m):
        factor[i] = i * factor[i-1]
    for i in range(1,m):
        pownode[:,i] = node * pownode[:,i-1]
        
    A = pownode / factor
    A = A.T
    x = np.linalg.solve(A, b)
    return x

def diff1d(u, node, order=1, bondary=1):
    '''
    本函数用于计算向量u的差分(只能是向量，不能是第二个维度为1的numpy二维数组)
    u为待差分向量，node为差分结点，order为差分阶数(默认为1)，
    bondary为边界，默认为1，使用单侧差分计算边界;若为0，边界取0;若为2，采用周期性边界条件
    返回值为与u同大小的数组
    '''
    u = np.array(u)
    node = np.array(node)
        
    M=u.size        
    coeff_u = coeff(node, order)
    m, n = node[0], node[-1]
    len_node = len(node)
    du = u * 0.0
    
    for i in range(len_node):
        du[-m:M-n] = du[-m:M-n] + coeff_u[i] * u[-m+node[i]:M-n+node[i]]

    #边界点使用单侧差分
    if bondary == 1:
        node_left = node - m
        coeff_left = coeff(node_left, order)
        for i in range(len_node):
            du[:-m] = du[:-m] + coeff_left[i] * u[node_left[i]:-m+node_left[i]]
            
        node_right = node - n
        coeff_right = coeff(node_right, order)
        for i in range(len_node):
            du[M-n:] = du[M-n:] + coeff_right[i] * u[M-n+node_right[i]:M+node_right[i]]
    
    #周期性边界条件
    if bondary == 2:
        u = np.hstack((u[-1+m+1:],u,u[:n]))
        du = u * 0
        M=u.size  
        for i in range(len_node):
            du[-m:M-n] = du[-m:M-n] + coeff_u[i] * u[-m+node[i]:M-n+node[i]]
        du = du[-m:M-n]
        
    return du

def diff2d(u, node, dim, order=1, bondary=1):
    '''
    本函数用于计算numpy数组u沿某方向的差分
    u为待差分数组，node为差分模板，dim为差分方向.order为差分阶数(默认为1)，
    bondary为边界，默认为1，使用单侧差分计算边界;若为0，边界取0;若为2，采用周期性边界条件
    返回值为与u同大小的数组
    '''
    u = np.array(u)
    node = np.array(node)
    if dim == 1:
        u = u.T
        
    (M,N)=u.shape        
    coeff_u = coeff(node, order)
    m, n = node[0], node[-1]
    len_node = len(node)
    du = u * 0.0
    
    for i in range(len_node):
        du[-m:M-n] = du[-m:M-n] + coeff_u[i] * u[-m+node[i]:M-n+node[i]]

    #边界点使用单侧差分
    if bondary == 1:
        node_left = node - m
        coeff_left = coeff(node_left, order)
        for i in range(len_node):
            du[:-m] = du[:-m] + coeff_left[i] * u[node_left[i]:-m+node_left[i]]
            
        node_right = node - n
        coeff_right = coeff(node_right, order)
        for i in range(len_node):
            du[M-n:] = du[M-n:] + coeff_right[i] * u[M-n+node_right[i]:M+node_right[i]]
    
    #周期性边界条件
    if bondary == 2:
        u = np.vstack((u[-1+m+1:],u,u[:n]))
        du = u * 0
        (M,N)=u.shape  
        for i in range(len_node):
            du[-m:M-n] = du[-m:M-n] + coeff_u[i] * u[-m+node[i]:M-n+node[i]]
        du = du[-m:M-n]
    
    if dim == 1:
        du = du.T
    return du

def jacobian_trans(x, y, nodes):
    Xxi = diff2d(x, nodes, dim=0); Xeta = diff2d(x, nodes, dim=1)
    Yxi = diff2d(y, nodes, dim=0); Yeta = diff2d(y, nodes, dim=1)
    
    jac = np.stack([Xxi, Yxi, Xeta, Yeta], axis=2).reshape(x.shape[0],x.shape[1],2,2)
    jac_inv = np.linalg.inv(jac)
    return jac, jac_inv

def diff2d_jac_inv(jac_inv, u, nodes):
    u_xi = diff2d(u, nodes, dim=0); u_eta = diff2d(u, nodes, dim=1)
    u_xieta = np.stack([u_xi, u_eta], axis=2).reshape(u.shape[0],u.shape[1],2,1)
    u_xy = (jac_inv@u_xieta).squeeze()
    return u_xy

def diff2d_xy(x, y, u, nodes):
    jac, jac_inv = jacobian_trans(x, y, nodes)
    
    u_xi = diff2d(u, nodes, dim=0); u_eta = diff2d(u, nodes, dim=1)
    u_xieta = np.stack([u_xi, u_eta], axis=2).reshape(u.shape[0],u.shape[1],2,1)
    u_xy = (jac_inv@u_xieta).squeeze()
    return u_xy

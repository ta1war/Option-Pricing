#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 09:50:30 2022

@author: udaytalwar
"""

import scipy.stats as stats
import math as m 
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

#set DPI of all figures 

import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 500

#calculating option price using Black-Scholes 

def d(S, K, sigma, r, t):
    
    '''
    S = Current Price
    K = Strike Price
    sigma = Volatility
    r = annualized risk-free rate
    t = time to expiration
    
    returns d1, d2 for option price calculation using Black Scholes
    '''
    d1 = (m.log(S/K) + (r + (sigma**2/2))*t) * (1/(sigma*m.sqrt(t)))
    
    d2 = d1 - sigma*m.sqrt(t)
    
    return d1, d2

def option_price(S, K, sigma, r, t, flag, d1 = 0, d2 = 0):
    
    '''
    S = Current Price
    K = Strike Price
    sigma = Volatility
    r = annualized risk-free rate
    t = time to expiration
    flag = 'Call' or 'Put'
    
    returns option price according to Black Scholes
    '''
    
    if d1 == 0 and d2 == 0:
    
        d1, d2 = d(S, K, sigma, r, t)
        
        if flag == 'Call':
            
            price = stats.norm.cdf(d1)*S - stats.norm.cdf(d2)*K*m.exp(-r*t)
            
        elif flag == 'Put':
            
            price = stats.norm.cdf(-d2)*K*m.exp(-r*t) - stats.norm.cdf(-d1)*S
            
        return price 

    else: 
        
        if flag == 'Call':
            
            price = stats.norm.cdf(d1)*S - stats.norm.cdf(d2)*K*m.exp(-r*t)
            
        elif flag == 'Put':
            
            price = stats.norm.cdf(-d2)*K*m.exp(-r*t) - stats.norm.cdf(-d1)*S
            
        return price  
    
#calculating option price using Monte Carlo Approximation

def option_price_MC(S, K, sigma, r, t, flag, steps, paths, error = False):

    '''
    S = Current Price
    K = Strike Price
    sigma = Volatility
    r = annualized risk-free rate
    t = time to expiration
    flag = 'Call' or 'Put'
    steps = number of steps per path 
    paths = number of paths to simulate
    error = False, if true returns (price, approximation error)
    
    returns expected option price from simulation

    '''
        
    timestep = t/steps 
    
    nudt = (r - 0.5*sigma**2)*timestep
    
    volsdt = vol*m.sqrt(timestep)
    
    lnS = m.log(S)
    
    random_matr = np.random.normal(size=(steps,paths))

    delta_lnSt = nudt + volsdt*random_matr
    
    lnSt = lnS + np.cumsum(delta_lnSt, axis = 0)
    
    lnSt = np.concatenate( (np.full(shape = (1,paths), fill_value = lnS), lnSt))
    
    ST = np.exp(lnSt)
    
    if flag == 'Call':    
        price = np.maximum(0, ST - K)
    
    elif flag == 'Put':
        price = np.maximum(0, K - ST)
    
    initial_price = np.exp(-r*t)*np.sum(price[-1])/paths
    
    err = np.sqrt(np.sum((price[-1]-initial_price)**2)/(paths-1))
    
    SE = err/m.sqrt(paths)
    
    if error == True: 
        
        return initial_price, SE

    else:
        return initial_price

#calculating option price using Binomial Trees

def option_price_BT(S, K, sigma, r, t, flag, steps):

    '''
    S = Current Price
    K = Strike Price
    sigma = Volatility
    r = annualized risk-free rate
    t = time to expiration
    flag = 'Call' or 'Put'
    steps = number of steps per path 
    
    returns option price using Binomial Trees
    '''

    
    timestep = t/steps #equivalent to dt, a discrete time step to iterate through
    
    
    #the formulas for A, d, u and p are described in Wilmott's book 
    
    A = (1.0/2.0)*(m.exp(-r*timestep)+m.exp((r+sigma**2)*timestep))
    
    d = A - m.sqrt(A**2 - 1)
    
    u = A + m.sqrt(A**2 - 1)
    
    p = (m.exp(r*timestep)-d)/(u-d)

    #discount function to discount the final value back to our initial value ie current option price
    discount = m.exp(-r*timestep)

    #price at node (i,j) at timestep i is = start price * d^(i-j)*u^(i) where j is the number of the node at the 
    # given time step, where we count from the bottom of the tree up to the top

    price = S * d ** (np.arange(steps,-1,-1)) * u **(np.arange(0, steps+1, 1))
    
        #price - K for Call, K - price for put
    if flag == 'Call':
    
        price = np.maximum(price - K, np.zeros(steps+1))
    
    elif flag == 'Put':
        
        price = np.maximum(K - price, np.zeros(steps+1))
    
        #apply the discount function to iteratively arrive at initial value
        
    for j in np.arange(steps, 0, -1):
        
        # formula below also given in Wilmott's book 
        
        price = discount * (p * price[1:j+1] + (1-p) * price[0:j])
    
    return price[0]
    

t = 30.0/365.0
S0 = 100
K = 100
flag = 'Put'
vol = (0.4/30.0)*np.sqrt(365)
r = 0.015

steps = 20
paths = 10000
steps_bt = 500

print('Price according to B-S = ',str(round(option_price(S0, K, vol, r, t, flag), 3)))

print('Price according to Monte Carlo = ',str(round(option_price_MC(S0, K, vol, r, t, flag, steps, paths), 3)))

print('Price according to Binomial Tree= ',str(round(option_price_BT(S0, K, vol, r, t, flag, steps_bt), 3)))


cumu_price = 0

mc_price = []

for i in range(1,501):
    
    price = option_price_MC(S0, K, vol, r, t, flag, steps, paths)
    
    cumu_price += price
    
    mc_price.append(cumu_price/i)

plt.plot(mc_price, label = 'Monte-Carlo iterations', linewidth = 1, color = 'k')
plt.axhline(y = option_price(S0, K, vol, r, t, flag), color = 'r', linewidth = 0.5, linestyle = '--', \
            label = 'Black-Scholes price')
plt.xlabel('Iteration number')
plt.ylabel(r'$\frac{\sum_{i = 1}^n MC~Price_{i}}{n}$')
plt.title('Monte-Carlo approximation improves through successive iterations')
plt.legend()

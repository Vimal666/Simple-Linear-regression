# -*- coding: utf-8 -*-
"""
Created on Thu May 14 20:11:05 2020

@author: intel
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

delivery=pd.read_csv("D://DATA SCIENCE//ASSIGNMENT//assignment4//delivery.csv")
delivery.columns
plt.hist(delivery.Deliverytime)
plt.boxplot(delivery.Deliverytime)
plt.hist(delivery.Sortingtime)
plt.boxplot(delivery.Sortingtime)
plt.plot(delivery.Deliverytime,delivery.Sortingtime,"ro");plt.xlabel("Sortingtime");plt.ylabel("Deliverytime")
delivery.corr()
import statsmodels.formula .api as smf
model=smf.ols("Deliverytime~Sortingtime",data=delivery).fit()
type(model)
model.params
model.summary()
model.conf_int(0.05)
pred=model.predict(delivery)
pred

# -*- coding: utf-8 -*-
"""
Created on Thu May 14 19:19:25 2020

@author: intel
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# reading a csv file using pandas library
caloriesconsumed=pd.read_csv("D://DATA SCIENCE//ASSIGNMENT//assignment4//caloriesconsumed.csv")
caloriesconsumed.columns
caloriesconsumed

plt.hist(caloriesconsumed.Weight)
plt.boxplot(caloriesconsumed.Weight)
plt.plot(caloriesconsumed.Calories,caloriesconsumed.Weight,"ro");plt.xlabel("Calories");plt.ylabel("Weight")
plt.hist(caloriesconsumed.Calories)
plt.boxplot(caloriesconsumed.Calories)
caloriesconsumed.corr()
import statsmodels.formula.api as smf
model=smf.ols("Weight~Calories",data=caloriesconsumed).fit()
type(model)
model.params
model.summary()
model.conf_int(0.05)
pred=model.predict(caloriesconsumed)
pred

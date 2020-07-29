# -*- coding: utf-8 -*-
"""
Created on Thu May 14 22:09:08 2020

@author: intel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
empdata=pd.read_csv("D://DATA SCIENCE//ASSIGNMENT//assignment4//empdata.csv")
empdata.columns
plt.hist(empdata.Churnoutrate)
plt.boxplot(empdata.Churnoutrate)
plt.hist(empdata.Salaryhike)
plt.boxplot(empdata.Salaryhike)
plt.plot(empdata.Churnoutrate,empdata.Salaryhike,"bo");plt.xlabel("salaryhike");plt.ylabel("Churnoutrate")
empdata.corr()
import statsmodels.formula.api as smf
model=smf.ols("Churnoutrate~Salaryhike",data=empdata).fit()
model.params
model.summary()
model.conf_int(0.05)
pred=model.predict(empdata)
pred
pred.corr(empdata.Churnoutrate)
plt.scatter(x=empdata['Salaryhike'],y=empdata['Churnoutrate'],color='blue');plt.plot(empdata['Salaryhike'],pred,color="red");plt.xlabel('Salaryhike');plt.ylabel('Churnoutrate')

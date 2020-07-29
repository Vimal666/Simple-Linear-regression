# -*- coding: utf-8 -*-
"""
Created on Thu May 14 22:39:10 2020

@author: intel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
Salarydata=pd.read_csv("D://DATA SCIENCE//ASSIGNMENT//assignment4//Salarydata.csv")
Salarydata.columns
plt.hist(Salarydata.Salary)
plt.boxplot(Salarydata.Salary)
plt.hist(Salarydata.YearsExperience)
plt.plot(Salarydata.Salary,Salarydata.YearsExperience,"ro");plt.xlabel("YearsExperience");plt.ylabel("Salary")
Salarydata.corr()
import statsmodels.formula.api as smf
model=smf.ols("Salary~YearsExperience",data=Salarydata).fit()
model.params
model.summary()
model.conf_int(0.05)
pred=model.predict(Salarydata)
pred
pred.corr(Salarydata.Salary)
plt.scatter(x=Salarydata['YearsExperience'],y=Salarydata['Salary'],color='green');plt.plot(Salarydata['YearsExperience'],pred,color='black');plt.xlabel('YearsExperience');plt.ylabel('Salary')
model2=smf.ols("Salary~np.log(YearsExperience)",data=Salarydata).fit()
model2.params
model2.summary()
model2.conf_int(0.01)
pred2=model2.predict(Salarydata)
pred2
pred2.corr(Salarydata.Salary)
plt.scatter(x=Salarydata['YearsExperience'],y=Salarydata['Salary'],color='green');plt.plot(Salarydata['YearsExperience'],pred2,color='blue');plt.xlabel('YearsExperience');plt.ylabel('Salary')

import statsmodels.formula.api as smf
model3=smf.ols("np.log(Salary)~YearsExperience",data=Salarydata).fit()
model3.params
model3.summary()
model3.conf_int(0.01)
pred3=model3.predict(Salarydata)
pred3
pred3.corr(Salarydata.Salary)
plt.scatter(x=Salarydata['YearsExperience'],y=Salarydata['Salary'],color='green');plt.plot(Salarydata['YearsExperience'],np.exp(pred3),color='blue');plt.xlabel(YearsExperience);plt.plot(Salary)

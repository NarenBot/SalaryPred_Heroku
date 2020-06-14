# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 11:37:24 2020

@author: Naren
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error

salary_df = pd.read_excel(r"C:\Users\Naren\Anaconda3\Scripts(Spyder)\Salary Prediction\Salary.xlsx")

def cat_int(word):
    word_dict = {"zero":0, "one":1, "two":2, "three":3, "four":4, "five":5, "six":6, "seven":7, "eight":8,
                 "nine":9, "ten":10, "eleven":11, "twelve":12, 0:0}
    return word_dict[word]
    
salary_df["Experience"] = salary_df["Experience"].apply(lambda x : cat_int(x))

y = salary_df['Salary']
data = salary_df.drop('Salary', axis=1)

#plt.scatter(data.Experience,y)
#sns.heatmap(salary_df.corr(), annot=True)
xtrain, xtest, ytrain, ytest = train_test_split(data, y, test_size=0.15, random_state=42)

reg_model = LinearRegression()
reg_model.fit(xtrain,ytrain)

#ypred = reg_model.predict(xtest)
#acc = mean_squared_error(ytest,ypred)


#PICKLE:
pickle_out = open(r"C:\Users\Naren\Anaconda3\Scripts(Spyder)\Salary Prediction\Model.pkl","wb")
pickle.dump(reg_model, pickle_out)
pickle_out.close()



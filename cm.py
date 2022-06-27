import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
df=pd.read_csv("bna.csv")

factors=df[["variance","skewness","curtisis","entropy",]]
heart_attack=df["target"]
factors_train,factors_test_split(factors,heart_attack,test_size=0.25,random_state=0),
heart_attack_train,heart_attack_test=train_test

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
sc_x=StandardScaler()

factors_train=sc_x.fit_transform(factors_train)
factors_test=sc_x.transform(factors_test)
classifier2=LogisticRegression(random_state=0)
classifier2.fit(factors_train,heart_attack_train)

heart_attack_prediction_1=classifier2.predict(factors_test)
predicted_values_1 = [] 
for i in heart_attack_prediction_1: 
  if i == 0: 
    predicted_values_1.append("authorized") 
  else: 
    predicted_values_1.append("forged") 
    
actual_values_1 = [] 
for i in heart_attack_test.ravel(): 
  if i == 0: 
    actual_values_1.append("authorized") 
  else: 
    actual_values_1.append("forged")

cm=confusion_matrix(actual_values_1,predicted_values_1)
ax=plt.subplot()
sns.heatmap(cm,annot=True,ax=ax)
ax.set_xlabel("forged")
ax.set_ylabel("Authorized")
ax.set_title("Confusion Matrix")
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)

acuracy=(23+33)/(23+33+10+10)
print(acuracy)
#importing required modules

import matplotlib.pyplot as mplp
import pandas as pn
import sklearn as sk
import pandas_ods_reader as por

#reading the file

dataset = por.read_ods('sucides.ods',1,headers=False)
x=dataset.iloc[1:,:1].values
y=dataset.iloc[1:,-1:].values

#printing the number of sucides in a year

print(y)

#splitting the data into traing set and testing set

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

#predicting the values using LinearRegression

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
print(x_test)
y_pred

#plotting the traing set

mplp.scatter(x_train,y_train,color='red')
mplp.plot(x_train,reg.predict(x_train),color='blue')
mplp.title('Suicides in Telangana(Training set)')
mplp.xlabel('Year')
mplp.ylabel('No of suicides committed')
mplp.show()

#plotting the tseting set

mplp.scatter(x_test,y_test,color='red')
mplp.plot(x_train,reg.predict(x_train),color='blue')
mplp.title('Suicides in Telangana(Test set)')
mplp.xlabel('Year')
mplp.ylabel('No of suicides committed')
mplp.show()

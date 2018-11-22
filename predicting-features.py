'''
predict companies profits based on 4 features
'''

import pandas as pd
import numpy as np

#process the data as X & Y
dataset=pd.read_csv('50_Startups.csv')
X=dataset.iloc[:,:4].values
Y=dataset.iloc[:,4].values

print(X)


#because newyork... needs number to make
#so use LabelEncoder(1-d), OneHotEncoder(2-d) 对字符串数据进行二值化
#方法一 先用 LabelEncoder() 转换成连续的数值型变量，再用 OneHotEncoder() 二值化
#* 方法二 直接用 LabelBinarizer() 进行二值化

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder=LabelEncoder()
X[:,3]=labelencoder.fit_transform(X[:,3])
print(X[:,3])

#categorical_features：可能取值为all、indices数组或mask

#若为all时，代表所有的特征都被视为分类特征

#若为indices数组时，表示分类特征的indices值

#若mask时，表示特征长度数组
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()


#Avoiding Dummy Variable Trap use m-1 var
X = X[: , 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred=regressor.predict(X_test)

print('预测准确率：',regressor.score(X_test, Y_test))
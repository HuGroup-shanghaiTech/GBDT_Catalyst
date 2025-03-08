from ast import Delete
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error

train_data = pd.read_csv(r"./Train_Data.csv")
x_columns = []
for x in train_data.columns:
    if x not in ['id', 'label']:
        x_columns.append(x)
x_train = train_data[x_columns]
y_train = train_data['label']

test_data = pd.read_csv(r"./Test_Data.csv")
x_columns = []
for x in test_data.columns:
    if x not in ['id', 'label']:
        x_columns.append(x)
x_test = test_data[x_columns]
y_test = test_data['label']

params = {'n_estimators' : 400, 'max_depth' : 3, 'min_samples_split' : 2, 'learning_rate' : 0.05, 'loss' : 'ls'}

# 模型训练，使用GBDT算法
gbr = GradientBoostingRegressor(n_estimators=400, max_depth=3, min_samples_split=2, learning_rate=0.05)
gbr.fit(x_train, y_train)
joblib.dump(gbr, 'train_model.m')   # 保存模型

mse = mean_squared_error(y_test, gbr.predict(x_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

test_score = np.zeros((params['n_estimators'],),dtype=np.float64)
for i, y_pred in enumerate(gbr.staged_predict(x_test)):
    test_score[i] = gbr.loss_(y_test, y_pred)
print(y_test, y_pred)

gbr = joblib.load('train_model.m')
test_data = pd.read_csv(r"./Test_Data.csv")
testx_columns = []
for xx in test_data.columns:
    if xx not in ['id', 'label']:
        testx_columns.append(xx)
test_x = test_data[testx_columns]
test_y = gbr.predict(test_x)
test_y = np.reshape(test_y, (-1))

df = pd.DataFrame()
df['id'] = test_data['id']
df['label'] = test_y
df.to_csv("./data_pre.csv", header=None, index=None)

feature_importance = gbr.feature_importances_

for i in range(len(feature_importance)):
    print(train_data.columns[i+1], feature_importance[i])

# SalePrice - 房间价值，所要预测的目标值
# MSSubClass: 公寓类别
# MSZoning: 公寓地带划分类别
# LotFrontage: 公寓旁的街道上拥有的资产
# LotArea: 房间尺寸，单位为平方英尺
# Street: 通行道路类别
# Alley: 通行小巷类别
# LotShape: 形状类别
# LandContour: 平整类别
# Utilities: 实用性
# LotConfig: 批量配置
# LandSlope: 倾斜度
# Neighborhood: Ames市内的实际地址
# Condition1: 是否靠近主要道路或铁路
# Condition2: 是否靠近主要道路或铁路（如果存在第二个）
# BldgType: 房间类型
# HouseStyle: 房间风格
# OverallQual: 房间的整体质量
# OverallCond: 房间整体状态排名
# YearBuilt: 房间建造日期
# YearRemodAdd: 房间装修改造日期
# RoofStyle: 屋顶类型
# RoofMatl: 屋顶材料
# Exterior1st: 房间外的覆盖物
# Exterior2nd: 房间外的覆盖物（如果不只一种材料的话）
# MasVnrType: 房间内的砌体类型
# MasVnrArea: 每平方英尺的砌体面积
# ExterQual: 房间外部材料质量
# ExterCond: 当前房间外部的材料状况
# Foundation: 地基属性
# BsmtQual: 地基的高度
# BsmtCond: 地基的一般状态
# BsmtExposure: 道路或花园的地下室墙壁等级
# BsmtFinType1: 地下室1的质量
# BsmtFinSF1: 地下室1的面积
# BsmtFinType2: 地下室2的质量（如果存在的话）
# BsmtFinSF2: 地下室2的面积
# BsmtUnfSF: 没装修好的地下室面积
# TotalBsmtSF: 整体地下室的面积
# Heating: 供暖属性
# HeatingQC: 供暖的质量和状态
# CentralAir: 中央空调
# Electrical: 电力系统
# 1stFlrSF: 房间第一层的面积
# 2ndFlrSF: 房间第二层的面积
# LowQualFinSF: 所有楼层中质量低的面积
# GrLivArea: 地面以上的生活区域面积
# BsmtFullBath: 设备齐全的浴室
# BsmtHalfBath: 设备半齐全的浴室
# FullBath: 地面以上的所有浴室
# HalfBath: 地面以上的所有半成品浴室
# Bedroom: 基础等级以上的卧室数量
# Kitchen: 厨房数量
# KitchenQual: 厨房质量
# TotRmsAbvGrd: 地面以上不包含浴室的全部房间数量
# Functional: 家庭功能排名
# Fireplaces: 壁炉数量
# FireplaceQu: 壁炉质量
# GarageType: 车库位置
# GarageYrBlt: 每年建成的车库数量
# GarageFinish: 车库的内部装饰
# GarageCars: 车库内能停车的面积大小
# GarageArea: 车库整体大小
# GarageQual: 车库质量
# GarageCond: 车库状况
# PavedDrive: 铺好的车道
# WoodDeckSF: 木甲板的面积
# OpenPorchSF: 开放式门廊区面积
# EnclosedPorch: 关闭的门廊区面积
# 3SsnPorch: 三季门廊区面积
# ScreenPorch: 屏风门廊区面积
# PoolArea: 游泳池面积
# PoolQC: 游泳池质量
# Fence: 栅栏质量
# MiscFeature: 未在其他类别中包含的杂项功能
# MiscVal: 杂项功能的价值
# MoSold: 售出月份
# YrSold: 售出年份
# SaleType: 销售类别
# SaleCondition: 销售状态

import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
import math

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import gridspec
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn import neighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from IPython import display
from scipy import stats


# 展示特征数据
train_data = pd.read_csv('train.csv')
# print(train_data)

# 分析数据
# train_data['SalePrice'].describe()
# 查看梯度直方图
# sns.distplot(train_data["OverallQual"])
# 分析数据特征
# data_corr = pd.concat([train_data['SalePrice'], train_data['TotalBsmtSF']], axis=1)
# data_corr.plot.scatter(x='TotalBsmtSF', y='SalePrice', ylim=(0, 800000))
# 分析类别特征
# data_corr = pd.concat([train_data['SalePrice'], train_data['OverallQual']], axis=1)
# fig = sns.boxplot(x='OverallQual', y='SalePrice', data=data_corr)
# fig.axis(ymin=0, ymax=800000)
# 相关矩阵
# corrmat = train_data.corr()
# sns.heatmap(corrmat, vmax=.8, square=True)
# 特征与目标的相关矩阵
# k = 10
# corrmat = train_data.corr()
# cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
# cm = np.corrcoef(train_data[cols].values.T)
# sns.set(font_scale=1.25)
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
# 每个特征与目标的关系图
# sns.set()
# cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
# sns.pairplot(train_data[cols], size=2.5)
# 查询特征是否缺少部分信息
# total = train_data.isnull().sum().sort_values(ascending=False)
# percent = (train_data.isnull().sum() / train_data.isnull().count()).sort_values(ascending=False)
# missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# print(missing_data.head(20))
# 删除缺信息的特征
# train_data = train_data.drop((missing_data[missing_data['Total'] > 1]).index, 1)
# train_data = train_data.drop(train_data.loc[train_data['Electrical'].isnull()].index)
# print(train_data.isnull().sum().max())
# print(train_data['Electrical'].describe())
# 观察目标数据是否常规
# sns.distplot(train_data['GarageArea'], fit=stats.norm)
# fig = plt.figure()
# res = stats.probplot(train_data['GarageArea'], plot=plt)
# 将目标数据标准化
# train_data['SalePrice'] = np.log(train_data['SalePrice'])
# sns.distplot(train_data['SalePrice'], fit=stats.norm)
# fig = plt.figure()
# res = stats.probplot(train_data['SalePrice'], plot=plt)
# plt.show()



# OverallQual, GrLivArea, GarageCars, GarageArea, TotalBsmtSF, 1stFlrSF, FullBath, TotRmsAbvGrd, YearBuilt
# 房间整体质量， 生活面积， 车库内能停车的区域大小， 车库大小， 地下室面积， 房间第一层面积， 浴室数量， 总房间数， 建造年份
# 特征工程：
#   房间整体质量 * （0.6 * 生活面积 + 0.2 * 地下室面积 + 0.2 * 房间第一层面积）
#   保留 车库内能停车的区域大小， 去掉车库大小
#   浴室数量 / 总房间数
#   当前年份（2018） - 建造年份

## 提取特征
# cols = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'GarageCars', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
# fea1 = np.array(train_data[cols[0]].values * (0.6 * train_data[cols[1]].values + 0.2 * train_data[cols[2]].values + 0.2 * train_data[cols[3]].values))
# fea2 = np.array(train_data[cols[4]].values)
# fea3 = np.array(train_data[cols[5]].values / train_data[cols[6]].values)
# fea4 = np.array(2018 - train_data[cols[7]].values)
# x = np.c_[fea1, fea2, fea3, fea4]
# y = train_data['SalePrice'].values
# feature = train_data[cols]
# feature = train_data['OverallQual']
# label = train_data['SalePrice']
train_data['SalePrice'] = np.log(train_data['SalePrice'])
train_data['GrLivArea'] = np.log(train_data['GrLivArea'])
# train_data.sort_values(by='GrLivArea', ascending=False)[:2]
train_data = train_data.drop(train_data[train_data['Id'] == 1299].index)
train_data = train_data.drop(train_data[train_data['Id'] == 524].index)
train_data.dropna(axis=0, how='any')
train_data = pd.get_dummies(train_data)
cols = ['OverallQual', 'YearBuilt', 'GarageCars', 'FullBath', 'TotRmsAbvGrd', 'GrLivArea', 'TotalBsmtSF']
feature = train_data[cols]
label = train_data['SalePrice']

## 归一化并裁剪数据为训练和测试
# X_train,X_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.33, random_state=66)
# X_train,X_test, y_train, y_test = train_test_split(feature, label, test_size=0.33, random_state=1)

## 采用三种回归方法进行测试，找到拟合度最高的方法
# clfs = {
#         'LinearRegression': LinearRegression(), # 线性回归
#         'DecisionTreeRegressor': DecisionTreeRegressor(), # 决策树回归
#         'Svm': svm.SVR(), # 支持向量机回归
#         'KNeighborsRegressor': neighbors.KNeighborsRegressor(), # KNN回归
#         'RandomForestRegressor': RandomForestRegressor(n_estimators=400),  # 随机森林回归，决策树选择400个
#         'AdaBoostRegressor': AdaBoostRegressor(n_estimators=500), # AdaBoost回归，决策树选择500个
#         'GBRTRegressor': GradientBoostingRegressor(n_estimators=500), # GBRT回归，决策树选择800个
#         'BaggingRegressor': BaggingRegressor(), # Bagging回归
#         'ExtraTreeRegressor': ExtraTreeRegressor(), # 极端决策树回归
#         'BayesianRidge': BayesianRidge(), # 贝叶斯回归
#        }
# for clf in clfs:
#     try:
#         clfs[clf].fit(X_train, y_train.ravel())
#         pred_sc = clfs[clf].score(X_test, y_test)
#         print(clf + " cost: " + str(pred_sc))
#     except Exception as e:
#         print(clf + " Error:")
#         print(str(e))

## 训练数据，并保存模型
x = feature
y = label
clf = LinearRegression()
clf.fit(x, y)
# pred = clf.predict(x)
# data = pd.DataFrame()
# data['pred'] = pd.Series(pred)
# data['label'] = pd.Series(y)
# display.display(data.describe())
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
rfr = clf

## 梯度下降法训练回归模型


## 读取测试数据，并提取特征
test_data = pd.read_csv('test.csv')

## 提取测试数据特征
# # cols = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'GarageCars', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
# GarageCars = test_data['GarageCars'].fillna(1.766118) # 清洗数据
# TotalBsmtSF = test_data['TotalBsmtSF'].fillna(1046.117970) # 清洗数据
# fea1 = np.array(test_data[cols[0]].values * (0.6 * test_data[cols[1]].values + TotalBsmtSF + 0.2 * test_data[cols[3]].values))
# fea2 = np.array(GarageCars)
# fea3 = np.array(test_data[cols[5]].values / test_data[cols[6]].values)
# fea4 = np.array(2018 - test_data[cols[7]].values)
# data_test_x = np.c_[fea1, fea2, fea3, fea4]
# # test_data[cols].isnull().sum()
test_data['GrLivArea'] = np.log(test_data['GrLivArea'])
test_data = pd.get_dummies(test_data)
cols = ['OverallQual', 'YearBuilt', 'GarageCars', 'FullBath', 'TotRmsAbvGrd', 'GrLivArea', 'TotalBsmtSF']
data_test_x = test_data[cols]
data_test_x = data_test_x.fillna(value=0)

# all_data = pd.concat((train_data.loc[:,'MSSubClass':'SaleCondition'],
#                       test_data.loc[:,'MSSubClass':'SaleCondition']))
# # log transform the target:
# train_data["SalePrice"] = np.log1p(train_data["SalePrice"])
#
# # log transform skewed numeric features:
# numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
#
# skewed_feats = train_data[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
# skewed_feats = skewed_feats[skewed_feats > 0.75]
# skewed_feats = skewed_feats.index
#
# all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
# all_data = pd.get_dummies(all_data)
# #filling NA's with the mean of the column:
# all_data = all_data.fillna(all_data.mean())

## 预测测试数据并输入
fea_x = data_test_x
y_pred = rfr.predict(fea_x)
y_pred = np.exp(y_pred)
# print(y_te_pred)

## 保存预测结果
prediction = pd.DataFrame(y_pred, columns=['SalePrice'])
result = pd.concat([test_data['Id'], prediction], axis=1)
result.to_csv('Predictions.csv', index=False)
#KAGGLE STORE SALES TIME SERIES FORECASTING - THIRD ATTEMPT


# PACKAGES and settings ####
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import darts

np.set_printoptions(suppress=True, precision=8)
pd.options.display.float_format = '{:.8f}'.format




# LOAD MODIFIED DATA VERSION 2 ####
df_train = pd.read_csv("./ModifiedData/train_modified2.csv", encoding="utf-8")
df_test = pd.read_csv("./ModifiedData/test_modified2.csv", encoding="utf-8")


#set datetime index
df_train = df_train.set_index(pd.to_datetime(df_train.date))
df_train = df_train.drop("date", axis=1)
df_test = df_test.set_index(pd.to_datetime(df_test.date))
df_test = df_test.drop("date", axis=1)


#add category_store_no column for darts hierarchy
df_train["category_store_no"] = df_train["category"].astype(str) + "-" + df_train["store_no"].astype(str)
df_test["category_store_no"] = df_test["category"].astype(str) + "-" + df_test["store_no"].astype(str)




#create wide dataframes with dates as rows, sales numbers for each hierarchy node as columns


#total
total = pd.DataFrame(
  data=df_train.groupby("date").sales.sum(),
  index=df_train.groupby("date").sales.sum().index)

#category
category = pd.DataFrame(
  data=df_train.groupby(["date", "category"]).sales.sum(),
  index=df_train.groupby(["date", "category"]).sales.sum().index)
category = category.reset_index(level=1)
category = category.pivot(columns="category", values="sales")

#store
store_no = pd.DataFrame(
  data=df_train.groupby(["date", "store_no"]).sales.sum(),
  index=df_train.groupby(["date", "store_no"]).sales.sum().index)
store_no = store_no.reset_index(level=1)
store_no = store_no.pivot(columns="store_no", values="sales")

#category_store_no
category_store_no = pd.DataFrame(
  data=df_train.groupby(["date", "category_store_no"]).sales.sum(),
  index=df_train.groupby(["date", "category_store_no"]).sales.sum().index)
category_store_no = category_store_no.reset_index(level=1)
category_store_no = category_store_no.pivot(columns="category_store_no", values="sales")


#merge all wide dataframes
from functools import reduce
wide_frames = [total, category, store_no, category_store_no]

ts_train = reduce(lambda left, right: pd.merge(
  left, right, how="left", on="date"), wide_frames)
  
del total, category, store_no, wide_frames, category_store_no




#create target time series, map hierarchy to it


#create target covariates time series, separately for each hierarchy node
  #should it be just like the target series, with its own hierarchy?
    #oil will be the same for each node
    #onpromotion will be different for each node


#preprocessing steps:
  #log transform sales
  #difference oil, transactions, onpromotion
  #minmax scale oil, transactions, onpromotion

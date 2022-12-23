# TIME SERIES REGRESSION PART 1 - KAGGLE STORE SALES COMPETITION

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# Set printing options
np.set_printoptions(suppress=True, precision=4)
pd.options.display.float_format = '{:.4f}'.format


# Load original datasets
df_train = pd.read_csv("./OriginalData/train.csv", encoding="utf-8")
df_test = pd.read_csv("./OriginalData/test.csv", encoding="utf-8")
df_stores = pd.read_csv("./OriginalData/stores.csv", encoding="utf-8")
df_oil = pd.read_csv("./OriginalData/oil.csv", encoding="utf-8")
df_holidays = pd.read_csv("./OriginalData/holidays_events.csv", encoding="utf-8")
df_trans = pd.read_csv("./OriginalData/transactions.csv", encoding="utf-8")


# Rename some columns
df_holidays = df_holidays.rename(columns = {"type":"holiday_type"})
df_oil = df_oil.rename(columns = {"dcoilwtico":"oil"})
df_stores = df_stores.rename(columns = {
  "type":"store_type", "cluster":"store_cluster"})


# Combine df_train and df_test for data handling operations
df = pd.concat([df_train, df_test])


# Add columns from oil, stores and transactions datasets into main data
df = df.merge(df_trans, on = ["date", "store_nbr"], how = "left")
df = df.merge(df_oil, on = "date", how = "left")
df = df.merge(df_stores, on = "store_nbr", how = "left")


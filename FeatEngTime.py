# TIME SERIES REGRESSION PART 1 - KAGGLE STORE SALES COMPETITION


# TIME EFFECTS FEATURE ENGINEERING SCRIPT

import pandas as pd 
import numpy as np


# Set printing options
np.set_printoptions(suppress=True, precision=4)
pd.options.display.float_format = '{:.4f}'.format


# Load data
df_train = pd.read_csv(
  "./ModifiedData/Final/train_modified.csv", encoding="utf-8")
df_test = pd.read_csv(
  "./ModifiedData/Final/test_modified.csv", encoding="utf-8")

  
# Combine train and test data for feature engineering
df = pd.concat([df_train, df_test])
df = df.set_index(pd.to_datetime(df.date))
df = df.drop("date", axis=1)


# New year's day features
df["ny1"] = ((df.index.day == 1) & (df.index.month == 1)).astype(int)
df.loc[df["ny1"] == 1, ["local_holiday", "regional_holiday", "national_holiday"]] = 0
df["ny2"] = ((df.index.day == 2) & (df.index.month == 1)).astype(int)
df.loc[df["ny2"] == 1, ["local_holiday", "regional_holiday", "national_holiday"]] = 0


# Christmas-December features 
df["ny_eve31"] = ((df.index.day == 31) & (df.index.month == 12)).astype(int)
df["ny_eve30"] = ((df.index.day == 30) & (df.index.month == 12)).astype(int)
df.loc[(df["ny_eve31"] == 1) | (df["ny_eve30"] == 1), ["local_holiday", "regional_holiday", "national_holiday"]] = 0

df["xmas_before"] = 0
df.loc[
  (df.index.day.isin(range(13,24))) & (df.index.month == 12), "xmas_before"] = df.loc[
  (df.index.day.isin(range(13,24))) & (df.index.month == 12)].index.day - 12

df["xmas_after"] = 0
df.loc[
  (df.index.day.isin(range(24,28))) & (df.index.month == 12), "xmas_after"] = abs(df.loc[
  (df.index.day.isin(range(24,28))) & (df.index.month == 12)].index.day - 27)

df.loc[(df["xmas_before"] != 0) | (df["xmas_after"] != 0), ["local_holiday", "regional_holiday", "national_holiday"]] = 0


# Earthquake feature
# 18 > 17 > 19 > 20 > 21 > 22
df["quake_after"] = 0
df.loc[df.index == "2016-04-18", "quake_after"] = 6
df.loc[df.index == "2016-04-17", "quake_after"] = 5
df.loc[df.index == "2016-04-19", "quake_after"] = 4
df.loc[df.index == "2016-04-20", "quake_after"] = 3
df.loc[df.index == "2016-04-21", "quake_after"] = 2
df.loc[df.index == "2016-04-22", "quake_after"] = 1


# Split events, delete events column
df["dia_madre"] = ((df["event"] == 1) & (df.index.month == 5) & (df.index.day.isin([8,10,11,12,14]))).astype(int)
df["futbol"] = ((df["event"] == 1) & (df.index.isin(pd.date_range(start = "2014-06-12", end = "2014-07-13")))).astype(int)
df["black_friday"] = ((df["event"] == 1) & (df.index.isin(["2014-11-28", "2015-11-27", "2016-11-25"]))).astype(int)
df["cyber_monday"] = ((df["event"] == 1) & (df.index.isin(["2014-12-01", "2015-11-30", "2016-11-28"]))).astype(int)
df = df.drop("event", axis=1)


# Days of week dummies (monday intercept)
df["tuesday"] = (df.index.dayofweek == 1).astype(int)
df["wednesday"] = (df.index.dayofweek == 2).astype(int)
df["thursday"] = (df.index.dayofweek == 3).astype(int)
df["friday"] = (df.index.dayofweek == 4).astype(int)
df["saturday"] = (df.index.dayofweek == 5).astype(int)
df["sunday"] = (df.index.dayofweek == 6).astype(int)


# Holiday-event leads
df["local_lead1"] = df["local_holiday"].shift(-1).fillna(0)
df["regional_lead1"] = df["regional_holiday"].shift(-1).fillna(0)
df["national_lead1"] = df["national_holiday"].shift(-1).fillna(0)
df["diamadre_lead1"] = df["dia_madre"].shift(-1).fillna(0)
pd.isnull(df).sum()


# Split train-test, reset indexes, write to csv
df_train = df.iloc[range(0, len(df_train)), :]
df_test = df.iloc[range(len(df_train), len(df)), :]

df_train = df_train.reset_index()
df_test = df_test.reset_index()

df_train.to_csv(
  "./ModifiedData/Final/train_modified_timefeats.csv", index=False, encoding="utf-8")
df_test.to_csv(
  "./ModifiedData/Final/test_modified_timefeats.csv", index=False, encoding="utf-8")

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


# New year's day feature
df["ny1"] = ((df.index.day == 1) & (df.index.month == 1)).astype(int)
df.loc[df["ny1"] == 1, ["local_holiday", "regional_holiday", "national_holiday"]] = 0


# Christmas-December features 
df["ny_eve31"] = ((df.index.day == 31) & (df.index.month == 12)).astype(int)
df["ny_eve30"] = ((df.index.day == 30) & (df.index.month == 12)).astype(int)
df.loc[(df["ny_eve31"] == 1) | (df["ny_eve30"] == 1), ["local_holiday", "regional_holiday", "national_holiday"]] = 0

df["xmas_before"] = 0
df.loc[
  (df.index.day.isin(range(13,24))) & (df.index.month == 12), "xmas_before"] = 
  df.loc[
  (df.index.day.isin(range(13,24))) & (df.index.month == 12)].index.day - 12

df["xmas_after"] = 0
df.loc[
  (df.index.day.isin(range(24,28))) & (df.index.month == 12), "xmas_after"] = abs(df.loc[
  (df.index.day.isin(range(24,28))) & (df.index.month == 12)].index.day - 27)

df.loc[(df["xmas_before"] != 0) | (df["xmas_after"] != 0), ["local_holiday", "regional_holiday", "national_holiday"]] = 0


# Earthquake feature
df["quake_after"] = 0
df.loc[
  (df.index.day.isin(range(16,31)) & (df.index.month == 4) & (df.index.year == 2016)) |
  (df.index.day.isin(range(1,16)) & (df.index.month == 5) & (df.index.year == 2016)), "quake_after"] =
  df.loc[
  (df.index.day.isin(range(16,31)) & (df.index.month == 4) & (df.index.year == 2016)) |
  (df.index.day.isin(range(1,16)) & (df.index.month == 5) & (df.index.year == 2016))].index.day



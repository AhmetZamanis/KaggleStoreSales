# TIME SERIES REGRESSION PART 1 - KAGGLE STORE SALES COMPETITION


# DATA HANDLING SCRIPT

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


# Split holidays data into local, regional, national and events
events = df_holidays[df_holidays["holiday_type"] == "Event"]
df_holidays = df_holidays.drop(labels=(events.index), axis=0)
local = df_holidays.loc[df_holidays["locale"] == "Local"]
regional = df_holidays.loc[df_holidays["locale"] == "Regional"]
national = df_holidays.loc[df_holidays["locale"] == "National"]


# Inspect local holidays sharing same date & locale. Drop the transfer row
local[local.duplicated(["date", "locale_name"], keep = False)]
local = local.drop(265, axis = 0)


# Inspect regional holidays sharing same date & locale. None exist
regional[regional.duplicated(["date", "locale_name"], keep = False)]


# Inspect national holidays sharing same date & locale. Drop bridge days
national[national.duplicated(["date"], keep = False)]
national = national.drop([35, 39, 156], axis = 0)


# Inspect events sharing same date. Drop the earthquake row
events[events.duplicated(["date"], keep = False)]
events = events.drop(244, axis = 0)


# Add local_holiday binary column to local holidays data, to be merged into main 
# data.
local["local_holiday"] = (
  local.holiday_type.isin(["Transfer", "Additional", "Bridge"]) |
  ((local.holiday_type == "Holiday") & (local.transferred == False))
).astype(int)


# Add regional_holiday binary column to regional holidays data
regional["regional_holiday"] = (
  regional.holiday_type.isin(["Transfer", "Additional", "Bridge"]) |
  ((regional.holiday_type == "Holiday") & (regional.transferred == False))
).astype(int)


# Add national_holiday binary column to national holidays data
national["national_holiday"] = (
  national.holiday_type.isin(["Transfer", "Additional", "Bridge"]) |
  ((national.holiday_type == "Holiday") & (national.transferred == False))
).astype(int)


# Add event column to events
events["event"] = 1


# Merge local holidays binary column to main data, on date and city
local_merge = local.drop(
  labels = [
    "holiday_type", "locale", "description", "transferred"], axis = 1).rename(
      columns = {"locale_name":"city"})
df = df.merge(local_merge, how="left", on=["date", "city"])
df["local_holiday"] = df["local_holiday"].fillna(0).astype(int)


# Merge regional holidays binary column to main data
regional_merge = regional.drop(
  labels = [
    "holiday_type", "locale", "description", "transferred"], axis = 1).rename(
      columns = {"locale_name":"state"})
df = df.merge(regional_merge, how="left", on=["date", "state"])
df["regional_holiday"] = df["regional_holiday"].fillna(0).astype(int)


# Merge national holidays binary column to main data, on date
national_merge = national.drop(
  labels = [
    "holiday_type", "locale", "locale_name", "description", 
    "transferred"], axis = 1)
df = df.merge(national_merge, how="left", on="date")
df["national_holiday"] = df["national_holiday"].fillna(0).astype(int)


# Merge events binary column to main data
events_merge = events.drop(
  labels = [
    "holiday_type", "locale", "locale_name", "description", 
    "transferred"], axis = 1)
df = df.merge(events_merge, how="left", on="date")
df["event"] = df["event"].fillna(0).astype(int)


# Set datetime index
df = df.set_index(pd.to_datetime(df.date))
df = df.drop("date", axis=1)


# CPI adjust sales and oil, with CPI 2010 = 100, and CPI 2017 = CPI 2016
cpis = {
  "2010":100, "2013":112.8, "2014":116.8, "2015":121.5, "2016":123.6, 
  "2017":123.6
  }
for year in [2013, 2014, 2015, 2016, 2017]:
  df["sales"].loc[df.index.year==year] = df["sales"].loc[
    df.index.year==year] / cpis[str(year)] * cpis["2010"]
  df["oil"].loc[df.index.year==year] = df["oil"].loc[
    df.index.year==year] / cpis[str(year)] * cpis["2010"]


# Split train and test, drop sales and transactions from test
df_train = df.iloc[range(0, len(df_train)), :]
df_test = df.iloc[range(len(df_train), len(df)), :]
df_test = df_test.drop(["sales", "transactions"], axis = 1)


# Check missing values in train and test. For train, NAs in oil and transactions.
# For test, NAs in oil.
pd.isnull(df_train).sum()
pd.isnull(df_test).sum()


# Time interpolate missing values in oil (train-test separately). Some are left
# in train, these are all from the first day in the data. Backfill them with the
# next day's oil price.
df_train["oil"] = df_train["oil"].interpolate("time")
df_test["oil"] = df_test["oil"].interpolate("time")
df_train["oil"] = df_train["oil"].fillna(method="bfill")


# Time interpolate missing values in transactions (train only). Some are left,
# all from the first day in the data, 01-01-2013. Fill them in with transactions
# from 01-01-2014.
df_train["transactions"] = df_train["transactions"].interpolate("time")
df_train["transactions"] = df_train["transactions"].fillna(
  df_train["transactions"].loc[
    (df_train.index.day == 1) & (df_train.index.month == 1) & 
    (df_train.index.year == 2014)].median()
    )


# Export modified train and test data
df_train.to_csv(
  "./ModifiedData/Final/train_modified.csv", index=True, encoding="utf-8")
df_test.to_csv(
  "./ModifiedData/Final/test_modified.csv", index=True, encoding="utf-8")




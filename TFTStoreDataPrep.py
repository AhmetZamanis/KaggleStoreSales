# Data prep for TFT store model, because num_workers doesn't work in jupyter


# Import libraries
import pandas as pd 
import numpy as np
from darts.dataprocessing.transformers import Scaler

# Load original datasets
df_train = pd.read_csv("./OriginalData/train.csv", encoding="utf-8")
df_test = pd.read_csv("./OriginalData/test.csv", encoding="utf-8")
df_stores = pd.read_csv("./OriginalData/stores.csv", encoding="utf-8")
df_oil = pd.read_csv("./OriginalData/oil.csv", encoding="utf-8")
df_holidays = pd.read_csv("./OriginalData/holidays_events.csv", encoding="utf-8")
df_trans = pd.read_csv("./OriginalData/transactions.csv", encoding="utf-8")

# Combine df_train and df_test
df = pd.concat([df_train, df_test])

# Rename columns
df = df.rename(columns = {"family":"category"})
df_holidays = df_holidays.rename(columns = {"type":"holiday_type"})
df_oil = df_oil.rename(columns = {"dcoilwtico":"oil"})
df_stores = df_stores.rename(columns = {
  "type":"store_type", "cluster":"store_cluster"})

# Add columns from oil, stores and transactions datasets into main data
df = df.merge(df_stores, on = "store_nbr", how = "left")
df = df.merge(df_trans, on = ["date", "store_nbr"], how = "left")
df = df.merge(df_oil, on = "date", how = "left")


# Split holidays data into local, regional, national and events
events = df_holidays[df_holidays["holiday_type"] == "Event"]
df_holidays = df_holidays.drop(labels=(events.index), axis=0)
local = df_holidays.loc[df_holidays["locale"] == "Local"]
regional = df_holidays.loc[df_holidays["locale"] == "Regional"]
national = df_holidays.loc[df_holidays["locale"] == "National"]

# Drop duplicate rows in holidays-events
local = local.drop(265, axis = 0)
national = national.drop([35, 39, 156], axis = 0)
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


# CPI adjust sales and oil, with CPI 2010 = 100
cpis = {
  "2010": 100, "2013": 112.8, "2014": 116.8, "2015": 121.5, "2016": 123.6, 
  "2017": 124.1
  }
  
for year in [2013, 2014, 2015, 2016, 2017]:
  df["sales"].loc[df.index.year==year] = df["sales"].loc[
    df.index.year==year] / cpis[str(year)] * cpis["2010"]
  df["oil"].loc[df.index.year==year] = df["oil"].loc[
    df.index.year==year] / cpis[str(year)] * cpis["2010"]
del year

# Interpolate missing values in oil
df["oil"] = df["oil"].interpolate("time", limit_direction = "both")


# New year's day features
df["ny1"] = ((df.index.day == 1) & (df.index.month == 1)).astype(int)

# Set holiday dummies to 0 if NY dummies are 1
df.loc[df["ny1"] == 1, ["local_holiday", "regional_holiday", "national_holiday"]] = 0
df["ny2"] = ((df.index.day == 2) & (df.index.month == 1)).astype(int)
df.loc[df["ny2"] == 1, ["local_holiday", "regional_holiday", "national_holiday"]] = 0

# NY's eve features
df["ny_eve31"] = ((df.index.day == 31) & (df.index.month == 12)).astype(int)

df["ny_eve30"] = ((df.index.day == 30) & (df.index.month == 12)).astype(int)

df.loc[(df["ny_eve31"] == 1) | (df["ny_eve30"] == 1), ["local_holiday", "regional_holiday", "national_holiday"]] = 0

# Proximity to Christmas sales peak
df["xmas_before"] = 0

df.loc[
  (df.index.day.isin(range(13,24))) & (df.index.month == 12), "xmas_before"] = df.loc[
  (df.index.day.isin(range(13,24))) & (df.index.month == 12)].index.day - 12

df["xmas_after"] = 0
df.loc[
  (df.index.day.isin(range(24,28))) & (df.index.month == 12), "xmas_after"] = abs(df.loc[
  (df.index.day.isin(range(24,28))) & (df.index.month == 12)].index.day - 27)

df.loc[(df["xmas_before"] != 0) | (df["xmas_after"] != 0), ["local_holiday", "regional_holiday", "national_holiday"]] = 0

# Strength of earthquake effect on sales
# April 18 > 17 > 19 > 20 > 21 > 22
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

# Days of week dummies
df["tuesday"] = (df.index.dayofweek == 1).astype(int)
df["wednesday"] = (df.index.dayofweek == 2).astype(int)
df["thursday"] = (df.index.dayofweek == 3).astype(int)
df["friday"] = (df.index.dayofweek == 4).astype(int)
df["saturday"] = (df.index.dayofweek == 5).astype(int)
df["sunday"] = (df.index.dayofweek == 6).astype(int)

# Add category X store_nbr column for Darts hierarchy
df["category_store_nbr"] = df["category"].astype(str) + "-" + df["store_nbr"].astype(str)

# Train-test split
df_train = df.loc[:"2017-08-15"]
df_test = df.loc["2017-08-16":]

# Replace transactions NAs in train with 0
df_train["transactions"] = df_train["transactions"].fillna(0)
  
# Recombine train and test
df = pd.concat([df_train, df_test])

# Create wide dataframes with dates as rows, sales numbers for each hierarchy node as columns

# Total
total = pd.DataFrame(
  data=df_train.groupby("date").sales.sum(),
  index=df_train.groupby("date").sales.sum().index)

# Store
store_nbr = pd.DataFrame(
  data=df_train.groupby(["date", "store_nbr"]).sales.sum(),
  index=df_train.groupby(["date", "store_nbr"]).sales.sum().index)
store_nbr = store_nbr.reset_index(level=1)
store_nbr = store_nbr.pivot(columns="store_nbr", values="sales")

# Category x store
category_store_nbr = pd.DataFrame(
  data=df_train.groupby(["date", "category_store_nbr"]).sales.sum(),
  index=df_train.groupby(["date", "category_store_nbr"]).sales.sum().index)
category_store_nbr = category_store_nbr.reset_index(level=1)
category_store_nbr = category_store_nbr.pivot(columns="category_store_nbr", values="sales")

# Merge all wide dataframes
from functools import reduce
wide_frames = [total, store_nbr, category_store_nbr]
df_sales = reduce(lambda left, right: pd.merge(
  left, right, how="left", on="date"), wide_frames)
df_sales = df_sales.rename(columns = {"sales":"TOTAL"})
del total, store_nbr, wide_frames, category_store_nbr

# Print wide sales dataframe
print(df_sales.iloc[0:5, [0, 1, 2, 84, 148]])
print("Rows x columns: " + str(df_sales.shape))

from darts import TimeSeries
from itertools import product

# Create multivariate time series with sales components
ts_sales = TimeSeries.from_dataframe(df_sales, freq="D")

# Create lists of hierarchy nodes
categories = df_train.category.unique().tolist()
stores = df_train.store_nbr.unique().astype(str).tolist()
categories_stores = df_train.category_store_nbr.unique().tolist()

# Initialize empty dict
hierarchy_target = dict()

# Map store sales to total sales
for store in stores:
  hierarchy_target[store] = ["TOTAL"]

# Map category X store combinations to respective stores
for category, store in product(categories, stores):
  hierarchy_target["{}-{}".format(category, store)] = [store]

# Map hierarchy to ts_train
ts_sales = ts_sales.with_hierarchy(hierarchy_target)
print(ts_sales)

del category, store


# Fill gaps
from darts.dataprocessing.transformers import MissingValuesFiller
na_filler = MissingValuesFiller()
ts_sales = na_filler.transform(ts_sales)


# Aggregate time features by mean
total_covars1 = df.drop(
  columns=['id', 'store_nbr', 'category', 'sales', 'onpromotion', 'transactions', 'oil', 'city', 'state', 'store_type', 'store_cluster'], axis=1).groupby("date").mean(numeric_only=True)
  
# Add piecewise linear trend dummies
total_covars1["trend"] = range(1, 1701) # Linear trend dummy 1
total_covars1["trend_knot"] = 0
total_covars1.iloc[728:,-1] = range(0, 972) # Linear trend dummy 2

# Add Fourier features for monthly seasonality
from statsmodels.tsa.deterministic import DeterministicProcess
dp = DeterministicProcess(
  index = total_covars1.index,
  constant = False,
  order = 0, # No trend feature
  seasonal = False, # No seasonal dummy features
  period = 28, # 28-period seasonality (28 days, 1 month)
  fourier = 5, # 5 Fourier pairs
  drop = True # Drop perfectly collinear terms
)
total_covars1 = total_covars1.merge(dp.in_sample(), how="left", on="date")

# Create Darts time series with time features
ts_totalcovars1 = TimeSeries.from_dataframe(total_covars1, freq="D")

# Fill gaps in covars
ts_totalcovars1 = na_filler.transform(ts_totalcovars1)

# Retrieve covars with filled gaps
total_covars1 = ts_totalcovars1.pd_dataframe()


# Retrieve copy of total_covars1, drop Fourier terms, trend knot (leaving daily predictors common to all categories).
common_covars = total_covars1[total_covars1.columns[0:21].values.tolist()]

# Add differenced oil price and its MA to common covariates. 
common_covars["oil"] = df.groupby("date").oil.mean()


# Difference daily covariate series
from sktime.transformations.series.difference import Differencer
diff = Differencer(lags = 1)
common_covars["oil"] = diff.fit_transform(common_covars["oil"]).interpolate("time", limit_direction = "both")

common_covars["oil_ma28"] = common_covars["oil"].rolling(window = 28, center = False).mean()

common_covars["oil_ma28"] = common_covars["oil_ma28"].interpolate(
  method = "spline", order = 2, limit_direction = "both")
  
# Print common covariates
print(common_covars.columns)



from darts.utils.timeseries_generation import datetime_attribute_timeseries

# Initialize list of store covariates
store_covars = []

for store in [int(store) for store in stores]:
  
  # Retrieve common covariates
  covars = common_covars.copy()
  
  # Retrieve differenced sales EMA
  covars["sales_ema7"] = diff.fit_transform(
    df[df["store_nbr"] == store].groupby("date").sales.sum()
    ).interpolate(
  "linear", limit_direction = "backward"
  ).rolling(
    window = 7, min_periods = 1, center = False, win_type = "exponential").mean()
    
  # Retrieve differenced onpromotion, its MA
  covars["onpromotion"] = diff.fit_transform(
    df[df["store_nbr"] == store].groupby("date").onpromotion.sum()
    ).interpolate(
      "time", limit_direction = "both"
      )
  covars["onp_ma28"] = covars["onpromotion"].rolling(
    window = 28, center = False
    ).mean().interpolate(
  method = "spline", order = 2, limit_direction = "both"
  ) 
  
  # Retrieve differenced transactions, its MA
  covars["transactions"] = diff.fit_transform(
    df[df["store_nbr"] == store].groupby("date").transactions.sum().interpolate(
      "time", limit_direction = "both"
      )
    )
  covars["trns_ma7"] = covars["transactions"].rolling(
    window = 7, center = False
    ).mean().interpolate(
  "linear", limit_direction = "backward"
  )
  
  # Create darts TS, fill gaps
  covars = na_filler.transform(
    TimeSeries.from_dataframe(covars, freq = "D")
    )
  
  # Cyclical encode day of month using datetime_attribute_timeseries
  covars = covars.stack(
    datetime_attribute_timeseries(
      time_index = covars,
      attribute = "day",
      cyclic = True
      )
    )
    
   # Cyclical encode month using datetime_attribute_timeseries
  covars = covars.stack(
    datetime_attribute_timeseries(
      time_index = covars,
      attribute = "month",
      cyclic = True
      )
    )
    
  # Append TS to list
  store_covars.append(covars)
  
# Cleanup
del covars



# Create dataframe where column=static covariate and index=store nbr
store_static = df[["store_nbr", "city", "state", "store_type", "store_cluster"]].reset_index().drop("date", axis=1).drop_duplicates().set_index("store_nbr")
store_static["store_cluster"] = store_static["store_cluster"].astype(str)

# Encode static covariates
store_static = pd.get_dummies(store_static, sparse = True, drop_first = True)



# Create min-max scaler
scaler_minmax = Scaler()

# Train-validation split and scaling for covariates
x_store = []
for series in store_covars:
  
  # Split train-val series
  cov_train, cov_innerval, cov_outerval = series[:-76], series[-76:-31], series[-31:]
  
  # Scale train-val series
  cov_train = scaler_minmax.fit_transform(cov_train)
  cov_innerval = scaler_minmax.transform(cov_innerval)
  cov_outerval = scaler_minmax.transform(cov_outerval)
  
  # Rejoin series
  cov_train = (cov_train.append(cov_innerval)).append(cov_outerval)
  
  # Cast series to 32-bits for performance gains
  cov_train = cov_train.astype(np.float32)
  
  # Append series to list
  x_store.append(cov_train)
  
# Cleanup
del cov_train, cov_innerval, cov_outerval



# List of store sales
store_sales = [ts_sales[store] for store in stores]

# Train-validation split for store sales
y_train_store, y_val_store = [], []
for series in store_sales:
  
  # Add static covariates to series
  series = series.with_static_covariates(
    store_static[store_static.index == int(series.components[0])]
  )
  
  # Split train-val series
  y_train, y_val = series[:-15], series[-15:]
  
  # Cast series to 32-bits for performance gains
  y_train = y_train.astype(np.float32)
  y_val = y_val.astype(np.float32)
  
  # Append series
  y_train_store.append(y_train)
  y_val_store.append(y_val)
  
# Cleanup
del y_train, y_val

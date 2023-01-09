# TIME SERIES REGRESSION PART 1 - KAGGLE STORE SALES COMPETITION


# TIME EFFECTS EDA SCRIPT

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Set printing options
np.set_printoptions(suppress=True, precision=4)
pd.options.display.float_format = '{:.4f}'.format


# Set plotting options
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams["figure.autolayout"] = True


# Load train data
df_train = pd.read_csv(
  "./ModifiedData/Final/train_modified.csv", encoding="utf-8")
  

# Set datetime index
df_train = df_train.set_index(pd.to_datetime(df_train.date))
df_train = df_train.drop("date", axis=1)


# Aggregate total sales per day, across all stores and categories
total_sales = df_train.groupby("date").sales.sum()


# Create darts time series from total sales series
from darts import TimeSeries
ts_total = TimeSeries.from_series(total_sales, freq="D")


# Plot entire time series
total_sales.plot()
plt.show()
plt.close("all")


# Monthly sales in each year
sns.lineplot(
  x = total_sales.index.month, 
  y=total_sales, 
  hue=total_sales.index.year, data=total_sales, legend="brief")
plt.xlabel("month")
plt.show()
plt.close()


# Weekly sales in each year
sns.lineplot(
  x = total_sales.index.week, 
  y=total_sales, 
  hue=total_sales.index.year, data=total_sales, legend="brief")
plt.xlabel("week")
plt.show()
plt.close()


# Daily sales in each year
sns.lineplot(
  x = total_sales.index.dayofyear, 
  y=total_sales, 
  hue=total_sales.index.year, data=total_sales, legend="brief")
plt.xlabel("day of year")
plt.show()
plt.close()


# Plot annual seasonality, months aggregated
sales_monthly = total_sales.groupby(total_sales.index.month).sum()
sales_monthly.index = pd.RangeIndex(1, 13)
ts_monthly = TimeSeries.from_series(sales_monthly)
ts_monthly.plot()
plt.show()
plt.close("all")


# Plot annual seasonality, weeks aggregated
sales_weekly = total_sales.groupby(total_sales.index.week).sum()
sales_weekly.index = pd.RangeIndex(1, 54)
ts_weekly = TimeSeries.from_series(sales_weekly)
ts_weekly.plot()
plt.show()
plt.close("all")


# Plot annual seasonality, day of year aggregated
sales_daily = total_sales.groupby(total_sales.index.dayofyear).sum()
sales_daily.index = pd.RangeIndex(1, 367)
ts_daily = TimeSeries.from_series(sales_daily)
ts_daily.plot()
plt.show()
plt.close("all")


# Plot monthly seasonality, days of month aggregated
sales_dayofmonth = total_sales.groupby(total_sales.index.day).sum()
sales_dayofmonth.index = pd.RangeIndex(1, 32)
ts_dayofmonth = TimeSeries.from_series(sales_dayofmonth)
ts_dayofmonth.plot()
plt.show()
plt.close("all")


# Plot weekly seasonality, days of week aggregated
sales_dayofweek = total_sales.groupby(total_sales.index.dayofweek).sum()
sales_dayofweek.index = pd.RangeIndex(1, 8)
ts_dayofweek = TimeSeries.from_series(sales_dayofweek)
ts_dayofweek.plot()
plt.show()
plt.close("all")


# EXCLUDE 2017 FROM SEASONALITY PLOTS!!!

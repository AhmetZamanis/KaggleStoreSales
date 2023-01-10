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
plt.ylabel("daily sales, millions")
plt.show()
plt.close("all")


# Average sales per quarter of year
sns.lineplot(
  x = total_sales.index.quarter.astype(str), 
  y = total_sales, 
  hue = total_sales.index.year.astype(str), data=total_sales, legend = "brief")
plt.legend(title = "year", bbox_to_anchor=(1.05, 1.0), fontsize="small", loc='upper left')
plt.xlabel("quarter")
plt.show()
plt.close("all")


# Average sales per month of year
sns.lineplot(
  x = total_sales.index.month.astype(str), 
  y = total_sales, 
  hue = total_sales.index.year.astype(str), data=total_sales, legend = "brief")
plt.legend(title = "year", bbox_to_anchor=(1.05, 1.0), fontsize="small", loc='upper left')
plt.xlabel("month")
plt.show()
plt.close("all")


# Average sales per week of year
sns.lineplot(
  x = total_sales.index.week, 
  y=total_sales, 
  hue=total_sales.index.year.astype(str), data=total_sales, legend = "brief")
plt.legend(title = "year", bbox_to_anchor=(1.05, 1.0), fontsize="small", loc='upper left')
plt.xlabel("week")
plt.show()
plt.close("all")


# Average sales per day of year
sns.lineplot(
  x = total_sales.index.dayofyear, 
  y=total_sales, 
  hue=total_sales.index.year.astype(str), data=total_sales, legend="brief")
plt.legend(title = "year", bbox_to_anchor=(1.05, 1.0), fontsize="small", loc='upper left')
plt.xlabel("day of year")
plt.show()
plt.close("all")


# Average sales per day of month
sns.lineplot(
  x = total_sales.index.day, 
  y = total_sales, 
  hue = total_sales.index.year.astype(str), data=total_sales, legend="brief")
plt.legend(title = "year", bbox_to_anchor=(1.05, 1.0), fontsize="small", loc='upper left')
plt.xlabel("day of month")
plt.show()
plt.close("all")


# Average sales per day of week
sns.lineplot(
  x = total_sales.index.dayofweek, 
  y = total_sales, 
  hue = total_sales.index.year.astype(str), data=total_sales, legend="brief")
plt.legend(title = "year", bbox_to_anchor=(1.05, 1.0), fontsize="small", loc='upper left')
plt.xlabel("day of week")
plt.show()
plt.close("all")







# MAKE FIGURES: FIG1 - ANNUAL SEASONALITY, FIG 2 - MONTHLY AND WEEKLY SEASONALITY
# LAG, ACF, PACF PLOTS
# STL DECOMPOSITION, PLOT
# ADD CALENDAR FEATURES (TO FULL DATA), EXPORT (TRAIN AND TEST SEPARATE)

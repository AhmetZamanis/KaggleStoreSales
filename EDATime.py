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
sns.set_theme(context="paper")


# Load data
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
plt.ylabel("Daily sales, millions")
plt.show()
plt.savefig("./Plots/TimeEDA/FullSeries.png", dpi=300)
plt.close("all")


# FIG1: Annual seasonality, period averages
fig1, axes1 = plt.subplots(2,2, sharey=True)
fig1.suptitle('Average daily sales in given time periods,\n millions')

# Average sales per quarter of year
sns.lineplot(
  ax = axes1[0,0],
  x = total_sales.index.quarter.astype(str), 
  y = (total_sales / 1000000), 
  hue = total_sales.index.year.astype(str), data=total_sales, legend=False)
axes1[0,0].set_xlabel("quarter", fontsize=8)
axes1[0,0].set_ylabel("sales", fontsize=8)
axes1[0,0].tick_params(axis='both', which='major', labelsize=6)

# Average sales per month of year
sns.lineplot(
  ax = axes1[0,1],
  x = total_sales.index.month.astype(str), 
  y = (total_sales / 1000000), 
  hue = total_sales.index.year.astype(str), data=total_sales)
axes1[0,1].set_xlabel("month", fontsize=8)
axes1[0,1].set_ylabel("sales",fontsize=8)
axes1[0,1].legend(title = "year", bbox_to_anchor=(1.05, 1.0), fontsize="small", loc='best')
axes1[0,1].tick_params(axis='both', which='major', labelsize=6)

# Average sales per week of year
sns.lineplot(
  ax = axes1[1,0],
  x = total_sales.index.week, 
  y = (total_sales / 1000000), 
  hue = total_sales.index.year.astype(str), data=total_sales, legend=False)
axes1[1,0].set_xlabel("week of year", fontsize=8)
axes1[1,0].set_ylabel("sales",fontsize=8)
axes1[1,0].tick_params(axis='both', which='major', labelsize=6)
axes1[1,0].xaxis.set_ticks(np.arange(0, 52, 10))

# Average sales per day of year
sns.lineplot(
  ax = axes1[1,1],
  x = total_sales.index.dayofyear, 
  y = (total_sales / 1000000), 
  hue = total_sales.index.year.astype(str), data=total_sales, legend=False)
axes1[1,1].set_xlabel("day of year", fontsize=8)
axes1[1,1].set_ylabel("sales",fontsize=8)
axes1[1,1].tick_params(axis='both', which='major', labelsize=6)
axes1[1,1].xaxis.set_ticks(np.arange(0, 365, 100))

# Show fig1
plt.show()
fig1.savefig("./Plots/TimeEDA/AnnualSeasonality.png", dpi=300)
plt.close("all")


# FIG1.1: Annual seasonality, averaged over years
fig11, axes11 = plt.subplots(2,2, sharey=True)
fig11.suptitle('Average daily sales in given time periods,\n across all years, millions')

# Average sales per quarter of year
sns.lineplot(
  ax = axes11[0,0],
  x = total_sales.index.quarter.astype(str), 
  y = (total_sales / 1000000), 
  data=total_sales, legend=False)
axes11[0,0].set_xlabel("quarter", fontsize=8)
axes11[0,0].set_ylabel("sales", fontsize=8)
axes11[0,0].tick_params(axis='both', which='major', labelsize=6)

# Average sales per month of year
sns.lineplot(
  ax = axes11[0,1],
  x = total_sales.index.month.astype(str), 
  y = (total_sales / 1000000), 
  data=total_sales)
axes11[0,1].set_xlabel("month", fontsize=8)
axes11[0,1].set_ylabel("sales",fontsize=8)
axes11[0,1].tick_params(axis='both', which='major', labelsize=6)

# Average sales per week of year
sns.lineplot(
  ax = axes11[1,0],
  x = total_sales.index.week, 
  y = (total_sales / 1000000), 
  data=total_sales, legend=False)
axes11[1,0].set_xlabel("week of year", fontsize=8)
axes11[1,0].set_ylabel("sales",fontsize=8)
axes11[1,0].tick_params(axis='both', which='major', labelsize=6)
axes11[1,0].xaxis.set_ticks(np.arange(0, 52, 10))

# Average sales per day of year
sns.lineplot(
  ax = axes11[1,1],
  x = total_sales.index.dayofyear, 
  y = (total_sales / 1000000), 
  data=total_sales, legend=False)
axes11[1,1].set_xlabel("day of year", fontsize=8)
axes11[1,1].set_ylabel("sales",fontsize=8)
axes11[1,1].tick_params(axis='both', which='major', labelsize=6)
axes11[1,1].xaxis.set_ticks(np.arange(0, 365, 100))

# Show fig1.1
plt.show()
fig11.savefig("./Plots/TimeEDA/AnnualSeasonalityTotals.png", dpi=300)
plt.close("all")


# FIG2: Monthly and weekly seasonality
fig2, axes2 = plt.subplots(2)
fig2.suptitle('Monthly and weekly seasonality in sales,\n average daily sales in millions')

# Average sales per day of month
sns.lineplot(
  ax = axes2[0],
  x = total_sales.index.day, 
  y = (total_sales / 1000000), 
  hue = total_sales.index.year.astype(str), data=total_sales, legend=False)
axes2[0].set_xlabel("day of month", fontsize=8)
axes2[0].set_ylabel("sales", fontsize=8)
axes2[0].xaxis.set_ticks(np.arange(1, 32, 6))
axes2[0].xaxis.set_ticks(np.arange(1, 32, 1), minor=True)
axes2[0].yaxis.set_ticks(np.arange(0, 1.25, 0.25))
axes2[0].grid(which='minor', alpha=0.5)
axes2[0].grid(which='major', alpha=1)

# Average sales per day of week
sns.lineplot(
  ax = axes2[1],
  x = (total_sales.index.dayofweek+1).astype(str), 
  y = (total_sales / 1000000), 
  hue = total_sales.index.year.astype(str), data=total_sales)
axes2[1].legend(title = "year", bbox_to_anchor=(1.05, 1.0), fontsize="small", loc='best')
axes2[1].set_xlabel("day of week", fontsize=8)
axes2[1].set_ylabel("sales", fontsize=8)
axes2[1].yaxis.set_ticks(np.arange(0, 1.25, 0.25))

# Show fig2
plt.show()
fig2.savefig("./Plots/TimeEDA/MonthlyWeeklySeasonality.png", dpi=300)
plt.close("all")


# FIG2.1: Monthly and weekly seasonality, colored by month
fig21, axes21 = plt.subplots(2)
fig21.suptitle('Monthly and weekly seasonality in sales,\n average daily sales in millions')

# Average sales per day of month, colored by month
sns.lineplot(
  ax = axes21[0],
  x = total_sales.index.day, 
  y = (total_sales / 1000000), 
  hue = total_sales.index.month.astype(str), data=total_sales, errorbar=None)
axes21[0].legend(title = "month", bbox_to_anchor=(1.05, 1.0), fontsize="x-small", loc='best')
axes21[0].set_xlabel("day of month", fontsize=8)
axes21[0].set_ylabel("sales", fontsize=8)
axes21[0].xaxis.set_ticks(np.arange(1, 32, 6))
axes21[0].xaxis.set_ticks(np.arange(1, 32, 1), minor=True)
axes21[0].yaxis.set_ticks(np.arange(0, 1.25, 0.25))
axes21[0].grid(which='minor', alpha=0.5)
axes21[0].grid(which='major', alpha=1)

# Average sales per day of week, colored by month
sns.lineplot(
  ax = axes21[1],
  x = (total_sales.index.dayofweek+1).astype(str), 
  y = (total_sales / 1000000), 
  hue = total_sales.index.month.astype(str), data=total_sales, errorbar=None, legend=None)
axes21[1].set_xlabel("day of week", fontsize=8)
axes21[1].set_ylabel("sales", fontsize=8)
axes21[1].yaxis.set_ticks(np.arange(0, 1.25, 0.25))

# Show fig2.1
plt.show()
fig21.savefig("./Plots/TimeEDA/MonthlyWeeklySeasonalityByMonth.png", dpi=300)
plt.close("all")


# FIG2.2: Monthly and weekly seasonality, average across years
fig22, axes22 = plt.subplots(2)
fig22.suptitle('Monthly and weekly seasonality in sales,\n averaged across years, in millions')

# Average sales per day of month
sns.lineplot(
  ax = axes22[0],
  x = total_sales.index.day, 
  y = (total_sales / 1000000), 
  data=total_sales)
axes22[0].set_xlabel("day of month", fontsize=8)
axes22[0].set_ylabel("sales", fontsize=8)
axes22[0].xaxis.set_ticks(np.arange(1, 32, 6))
axes22[0].xaxis.set_ticks(np.arange(1, 32, 1), minor=True)
axes22[0].yaxis.set_ticks(np.arange(0, 1.25, 0.25))
axes22[0].grid(which='minor', alpha=0.5)
axes22[0].grid(which='major', alpha=1)

# Average sales per day of week
sns.lineplot(
  ax = axes22[1],
  x = (total_sales.index.dayofweek+1).astype(str), 
  y = (total_sales / 1000000), 
  data=total_sales)
axes22[1].set_xlabel("day of week", fontsize=8)
axes22[1].set_ylabel("sales", fontsize=8)
axes22[1].yaxis.set_ticks(np.arange(0, 1.25, 0.25))

# Show fig22
plt.show()
fig22.savefig("./Plots/TimeEDA/MonthlyWeeklySeasonalityTotals.png", dpi=300)
plt.close("all")


# FIG3: ACF and PACF plots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig3, axes3 = plt.subplots(2)
fig3.suptitle('Autocorrelation and partial autocorrelation,\n daily sales, up to 54 days')
plot_acf(total_sales, lags=range(0,55), ax=axes3[0])
plot_pacf(total_sales, lags=range(0,55), ax=axes3[1], method="ywm")

# Show fig3
plt.show()
fig3.savefig("./Plots/TimeEDA/AcfPacf.png", dpi=300)
plt.close("all")


# STL decomposition

# Weekly STL
from statsmodels.tsa.seasonal import STL
stl = STL(np.log(total_sales), period=52, robust=True).fit()
stl_trend = np.exp(stl.trend)
stl_seasonal = np.exp(stl.seasonal)
stl_resid = np.exp(stl.resid)

# FIG4: STL decomposition, weekly
fig4, axes4 = plt.subplots(4, sharex=True)
fig4.suptitle("Multiplicative STL decomposition,\n weekly seasonality period")

# Original series
sns.lineplot(
  ax = axes4[0],
  x = total_sales.index,
  y = total_sales,
  data = total_sales
)

# Trend
sns.lineplot(
  ax = axes4[1],
  x = stl_trend.index,
  y = stl_trend,
  data = stl_trend
)

# Seasonality
sns.lineplot(
  ax = axes4[2],
  x = stl_seasonal.index,
  y = stl_seasonal,
  data = stl_seasonal
)

# Residual
sns.lineplot(
  ax = axes4[3],
  x = stl_resid.index,
  y = stl_resid,
  data = stl_resid
)

# Show fig4
plt.show()
fig4.savefig("./Plots/TimeEDA/STLWeekly.png", dpi=300)
plt.close("all")


# Monthly STL on weekly STL residuals
stl2 = STL(stl_resid, period=12, robust=True).fit()

# FIG5: STL decomposition, monthly on weekly resids
fig5, axes5 = plt.subplots(4, sharex=True)
fig5.suptitle("Additive STL decomposition,\n monthly seasonality period,\n after removing weekly seasonality")

# Original series
sns.lineplot(
  ax = axes5[0],
  x = stl_resid.index,
  y = stl_resid,
  data = stl_resid
)

# Trend
sns.lineplot(
  ax = axes5[1],
  x = stl2.trend.index,
  y = stl2.trend,
  data = stl2.trend
)

# Seasonality
sns.lineplot(
  ax = axes5[2],
  x = stl2.seasonal.index,
  y = stl2.seasonal,
  data = stl2.seasonal
)

# Residual
sns.lineplot(
  ax = axes5[3],
  x = stl2.resid.index,
  y = stl2.resid,
  data = stl2.resid
)

# Show fig5
plt.show()
fig5.savefig("./Plots/TimeEDA/STLMonthlyAfterWeekly.png", dpi=300)
plt.close("all")


# FIG6: Zoom in on earthquake: 16 April 2016
april_sales = total_sales.loc[total_sales.index.month == 4]
may_sales = total_sales.loc[total_sales.index.month == 5]
sns.lineplot(
  x = april_sales.index.day,
  y = april_sales,
  hue = april_sales.index.year.astype(str),
  data = april_sales
)
plt.title("Effect of 16-04-16 earthquake on April sales")
plt.legend(title = "year", bbox_to_anchor=(1.05, 1.0), fontsize="small", loc='best')
plt.xlabel("days of month")
plt.xticks(np.arange(1, 32, 6))
plt.xticks(np.arange(1, 32, 1), minor=True)
plt.grid(which='minor', alpha=0.5)
plt.grid(which='major', alpha=1)
plt.show()
plt.savefig("./Plots/TimeEDA/QuakeApril.png", dpi=300)
plt.close("all")

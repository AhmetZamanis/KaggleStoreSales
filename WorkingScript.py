#KAGGLE STORE SALES FORECASTING COMPETITION

import pandas as pd
import numpy as np
from functools import reduce
from itertools import product
import matplotlib.pyplot as plt
from sktime.transformations.series.difference import Differencer
from statsmodels.tsa.deterministic import DeterministicProcess
import darts
from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers import(
  Scaler,
  Mapper
)

np.set_printoptions(suppress=True, precision=8)
pd.options.display.float_format = '{:.8f}'.format

# import sktime
# from sktime.utils.plotting import plot_series
# from sktime.utils.plotting import plot_lags
# from sktime.transformations.hierarchical.aggregate import Aggregator
# from sktime.forecasting.arima import AutoARIMA
# from sktime.forecasting.arima import ARIMA
# 
# 
# from statsmodels.graphics.tsaplots import plot_pacf
# from statsmodels.graphics.tsaplots import plot_acf
# from statsmodels.tsa.stattools import adfuller
# from statsmodels.tsa.stattools import ccf
# 
# from scipy.signal import periodogram
# from scipy.stats import pearsonr
# from scipy.stats import spearmanr
# 
# from sklearn.metrics import adjusted_mutual_info_score
# from sklearn.metrics import mean_squared_error as mse
# from sklearn.metrics import make_scorer
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import LinearRegression
# 
# 
# 
# from darts.models.forecasting.arima import ARIMA
# 
# from sktime.transformations.series.detrend import STLTransformer
# from statsmodels.tsa.seasonal import MSTL
# from statsmodels.tsa.seasonal import DecomposeResult







#LOAD DATASETS


#training and testing time series
df_train = pd.read_csv("./OriginalData/train.csv", encoding="utf-8")
df_train.shape
#(3000888, 6)

df_test = pd.read_csv("./OriginalData/test.csv", encoding="utf-8")
df_test.shape
#(28512, 5)

#stores data
df_stores = pd.read_csv("./OriginalData/stores.csv", encoding="utf-8")

#oil price data
df_oil = pd.read_csv("./OriginalData/oil.csv", encoding="utf-8")

#holidays data
df_holidays = pd.read_csv("./OriginalData/holidays_events.csv", encoding="utf-8")
df_holidays.date.nunique()
#there are duplicate dates in holidays

#transactions data
df_trans = pd.read_csv("./OriginalData/transactions.csv", encoding="utf-8")
df_trans.date.nunique()
#no duplicate dates


#data dictionary

#df_train, df_test
df_train.head()
df_test.head()
  #date: date in year, month, day
  #sales: sales that day
  #onpromotion: n. of items in the product family on promotion, at given date
  #family: category of product sales
  #store_nbr: store number (categorical)


#df_oil
df_oil.head()
  #oil prices by date


#df_stores
df_stores.head()
  #store_nbr: store number (categorical)
  #city: 22 cities
  #state: 16 states
  #type: type of store (5 types)
  #cluster: 17 clusters
  

#df_holidays
df_holidays.head()
  #type: 'Holiday', 'Transfer', 'Additional', 'Bridge', 'Work Day', 'Event'
    #Holiday: de jure holiday, de facto holiday if Transferred=False
    #Transfer: de jure no holiday, de facto holiday
    #Additional: de jure holiday, de facto holiday (additional days for a 1+ day holiday)
    #Bridge: de jure no holiday, de facto holiday (awarded to extend holidays)
    #Work Day: de jure holiday, de facto no holiday (detracted to make up for bridges)
  #locale: Local, Regional, National
  #locale_name: 23 different cities/regions, or Ecuador
  #description: 103 different holidays
  
  
#df_trans
df_trans.head()
#n. of transactions for each store, each date. all categories
  #for training data only
  #not all store-date combos have a transactions entry. replace NAs with zeroes
  #merge on date and store_no



#rename holiday_types column to avoid clashes at merge
df_holidays.rename(columns={"type":"holiday_type"}, inplace=True)

#rename oil column
df_oil.rename(columns={"dcoilwtico":"oil"}, inplace=True)


#inspect unique date and duplicate date holidays
df_holidays_unique = df_holidays.drop_duplicates(subset="date", keep=False)
df_holidays_duplicate = df_holidays[df_holidays.duplicated(["date"], keep=False)]


#split special events
df_events = df_holidays[df_holidays.holiday_type=="Event"]
df_holidays = df_holidays.drop(labels=(df_events.index), axis=0)


#split holidays into local, regional, national
df_local = df_holidays.loc[df_holidays.locale=="Local"]
df_regional = df_holidays.loc[df_holidays.locale=="Regional"]
df_national = df_holidays.loc[df_holidays.locale=="National"]


#check duplicate date & locale_name in each holiday dataframe
df_local_duplicated = df_local[df_local.duplicated(["date", "locale_name"], keep=False)]
#2 duplicates. remove the transfer row, row 265
df_local.loc[265]
df_local.drop(265, axis=0, inplace=True)
#no date&locale duplicates in local holidays remain 
del df_local_duplicated


df_regional_duplicated = df_regional[df_regional.duplicated(["date", "locale_name"], keep=False)]
#no date & locale duplicates in regional holidays
del df_regional_duplicated


df_national_duplicated = df_national[df_national.duplicated(["date"], keep=False)]
#6 duplicates, 3 pairs of bridge-additional in the same day.
#bridges are less common, so drop them
bridge_indexes = [35, 39, 156]
df_national.drop(labels=bridge_indexes, axis=0, inplace=True)
#no date&locale duplicates in national holidays remain
del df_national_duplicated, bridge_indexes


#check & handle event date duplicates (all events are nationwide)
df_events_duplicated = df_events[df_events.duplicated(["date"], keep=False)]
#1 pair of duplicates: earthquake and mother's day in 2016-05-08
#drop earthquake because it'll be factored in later, row 244
df_events.loc[244]
df_events.drop(244, axis=0, inplace=True)
#no date duplicates in events remain
del df_events_duplicated



#add calendar_holiday and actual_holiday columns
  #calendar_holiday=1 if: Holiday, Additional, Work Day
  #actual_holiday=1 if: (Holiday & Transferred=FALSE), Transfer, Additional, Bridge
# df_holidays["calendar_holiday"] = df_holidays.holiday_type.isin(["Holiday", "Additional", "Work Day"])
# df_holidays["actual_holiday"] = (
#   df_holidays.holiday_type.isin(["Transfer", "Additional", "Bridge"]) | 
# ((df_holidays.holiday_type=="Holiday") & (df_holidays.transferred==False))
# )

#add local_holiday column to df_local
df_local["local_holiday"] = (
  df_local.holiday_type.isin(["Transfer", "Additional", "Bridge"]) |
  ((df_local.holiday_type=="Holiday") & (df_local.transferred==False))
)

#add regional_holiday colulmn to df_regional
df_regional["regional_holiday"] = (
  df_regional.holiday_type.isin(["Transfer", "Additional", "Bridge"]) |
  ((df_regional.holiday_type=="Holiday") & (df_regional.transferred==False))
)

#add national_holiday column to df_national
df_national["national_holiday"] = (
  df_national.holiday_type.isin(["Transfer", "Additional", "Bridge"]) |
  ((df_national.holiday_type=="Holiday") & (df_national.transferred==False))
)

#add event column to df_events
df_events["event"] = True





#MERGE / JOIN DATASETS


#merge stores columns into train and test, by store_nbr
df_test = df_test.merge(df_stores, how="left", left_on="store_nbr", right_on="store_nbr")
df_train = df_train.merge(df_stores, how="left", left_on="store_nbr", right_on="store_nbr")


#set indexes to date
df_train.set_index(pd.PeriodIndex(df_train.date, freq="D"), inplace=True)
df_train.drop("date", axis=1, inplace=True)

df_test.set_index(pd.PeriodIndex(df_test.date, freq="D"), inplace=True)
df_test.drop("date", axis=1, inplace=True)

df_oil.set_index(pd.PeriodIndex(df_oil.date, freq="D"), inplace=True)
df_oil.drop("date", axis=1, inplace=True)

df_local.set_index(pd.PeriodIndex(df_local.date, freq="D"), inplace=True)
df_local.drop("date", axis=1, inplace=True)

df_regional.set_index(pd.PeriodIndex(df_regional.date, freq="D"), inplace=True)
df_regional.drop("date", axis=1, inplace=True)

df_national.set_index(pd.PeriodIndex(df_national.date, freq="D"), inplace=True)
df_national.drop("date", axis=1, inplace=True)

df_events.set_index(pd.PeriodIndex(df_events.date, freq="D"), inplace=True)
df_events.drop("date", axis=1, inplace=True)


#join oil price into train and test, by date index
df_train = df_train.join(df_oil, how="left", on="date")
df_test = df_test.join(df_oil, how="left", on="date")




#join holidays columns into train and test


#local
df_local_merge = df_local.drop(
  labels=["holiday_type", "locale", "description", "transferred"], axis=1)
df_local_merge.rename(columns={"locale_name":"city"}, inplace=True)

  
df_test = df_test.merge(df_local_merge, how="left", on=["date", "city"])
#works. remember to replace NAs with FALSE

df_train2 = df_train.merge(df_local_merge, how="left", on=["date", "city"])
#no duplicates generated, non NA rows added
df_train = df_train2




#regional
df_regional_merge = df_regional.drop(
  labels=[
    "holiday_type", "locale", "description", "transferred"], axis=1
)
df_regional_merge.rename(columns={"locale_name":"state"}, inplace=True)


df_test = df_test.merge(df_regional_merge, how="left", on=["date", "state"])

df_train2 = df_train.merge(df_regional_merge, how="left", on=["date", "state"])
#no duplicates generated, non NA rows added
df_train = df_train2




#national
df_national_merge = df_national.drop(
  labels=[
    "holiday_type", "locale", "locale_name", "description", "transferred"], axis=1
)


df_test = df_test.merge(df_national_merge, how="left", on="date")

df_train2 = df_train.merge(df_national_merge, how="left", on="date")
#no duplicates generated, non NA rows added
df_train = df_train2




#events
df_events_merge = df_events.drop(
  labels=[
    "holiday_type", "locale", "locale_name", "description", "transferred"], axis=1
)

df_test = df_test.merge(df_events_merge, how="left", on="date")

df_train2 = df_train.merge(df_events_merge, how="left", on="date")
#no duplicates generated, non NA rows added
df_train = df_train2


#set NA holiday values to False
holiday_cols = ['local_holiday','regional_holiday', 'national_holiday', 'event']

df_test[holiday_cols] = df_test[holiday_cols].fillna(value=False)
#worked for test data

df_train[holiday_cols] = df_train[holiday_cols].fillna(value=False)
#worked for train data



#check if there are any days with more than 1 value for the holiday and event columns
(((df_train.groupby("date").local_holiday.mean())==0) | ((df_train.groupby("date").local_holiday.mean())==1)).mean() 
#yes, but that's normal. can't use as grouping col in hierarchical data

(((df_train.groupby("date").regional_holiday.mean())==0) | ((df_train.groupby("date").regional_holiday.mean())==1)).mean() 
#yes, but that's normal. can't use as grouping col in hierarchical data

(((df_train.groupby("date").national_holiday.mean())==0) | ((df_train.groupby("date").national_holiday.mean())==1)).mean()
#no

(((df_train.groupby("date").event.mean())==0) | ((df_train.groupby("date").event.mean())==1)).mean()
#no
#the issue likely happens when you change the holiday-event columns again,
#after flagging special features like new years and christmas



#flag first day of year with binary feature, set national holidays columns to false at this date
df_train["new_years_day"] = (df_train.index.dayofyear==1)
df_test["new_years_day"] = (df_test.index.dayofyear==1)
df_train.national_holiday.loc[df_train.new_years_day==True] = False
#doesn't break one value per one date requirement


#flag christmas, 21-26, set national holidays columns to false at christmas
df_train["christmas"] = (df_train.index.month==12) & (df_train.index.day.isin(
  [21,22,23,24,25,26])) 
df_test["christmas"] = (df_test.index.month==12) & (df_test.index.day.isin(
  [21,22,23,24,25,26]))
df_train.national_holiday.loc[df_train.christmas==True] = False
#doesn't break one value per one date requirement


#flag paydays: 15th and last day of each month
df_train["payday"] = ((df_train.index.day==15) | (df_train.index.to_timestamp().is_month_end))
df_test["payday"] = ((df_test.index.day==15) | (df_test.index.to_timestamp().is_month_end))


#flag earthquakes: 2016-04-16 to 2016-05-16. set event to False for earthquake dates
earthquake_dates = pd.period_range(start="2016-04-16", end="2016-05-16", freq="D")
df_train["earthquake"] = (df_train.index.isin(earthquake_dates))
df_test["earthquake"] = False
df_train.loc[df_train.earthquake==True]
df_train.event.loc[df_train.earthquake==True] = False
#doesn't break one value per one date requirement


#check if the one value per one date requirement is violated because of flagged cols
(((df_train.groupby("date").local_holiday.mean())==0) | ((df_train.groupby("date").local_holiday.mean())==1)).mean() 
#yes, but that's normal. can't use as grouping col in hierarchical data

(((df_train.groupby("date").regional_holiday.mean())==0) | ((df_train.groupby("date").regional_holiday.mean())==1)).mean() 
#yes, but that's normal. can't use as grouping col in hierarchical data

(((df_train.groupby("date").national_holiday.mean())==0) | ((df_train.groupby("date").national_holiday.mean())==1)).mean()
#no

(((df_train.groupby("date").event.mean())==0) | ((df_train.groupby("date").event.mean())==1)).mean()
#no

(((df_train.groupby("date").new_years_day.mean())==0) | ((df_train.groupby("date").new_years_day.mean())==1)).mean()
#no

(((df_train.groupby("date").christmas.mean())==0) | ((df_train.groupby("date").christmas.mean())==1)).mean()
#no

(((df_train.groupby("date").payday.mean())==0) | ((df_train.groupby("date").payday.mean())==1)).mean()
#no

(((df_train.groupby("date").earthquake.mean())==0) | ((df_train.groupby("date").earthquake.mean())==1)).mean()
#no


#do some renaming and reordering
df_test = df_test.rename(columns={
  "store_nbr":"store_no",
  "family":"category",
  "type":"store_type",
  "cluster":"store_cluster"
}
)

df_train = df_train.rename(columns={
  "store_nbr":"store_no",
  "family":"category",
  "type":"store_type",
  "cluster":"store_cluster"
}
)

df_test = df_test[['id', 'category', 'onpromotion', 'city', 'state', 'store_no',
       'store_type', 'store_cluster', 'oil', 'local_holiday', 'regional_holiday', 
       'national_holiday', 'event',
       'new_years_day', 'payday', 'earthquake', "christmas"]]


df_train = df_train[['id', "sales", 'category', 'onpromotion', 'city', 'state', 'store_no',
       'store_type', 'store_cluster', 'oil', 'local_holiday', 'regional_holiday', 
       'national_holiday', 'event',
       'new_years_day', 'payday', 'earthquake', "christmas"]]
       



#handle missing values (before lags-indicators)


#check NAs
pd.isnull(df_train).sum()
#928422 in oil

pd.isnull(df_test).sum()
#7128 in oil



#fill in oil NAs with time interpolation
df_train["oil"] = df_train["oil"].interpolate("time")
df_test["oil"] = df_test["oil"].interpolate("time")
#got rid of all test NAs. 1782 remain for train


#check remaining train oil NAs 
df_train[pd.isnull(df_train["oil"])].oil
#all belong to the first day. fill them with the next day oil price


#fill first day oil NAs
df_train.oil = df_train.oil.fillna(method="bfill")


#check if it worked
pd.isnull(df_train).sum()
pd.isnull(df_test).sum()
#all done


#check if the oil price is the same 



# #plot actual oil prices vs. filled oil prices to see quality of interpolation
# 
# 
# #get daily oil prices in train and test data
# oil_train = df_train.oil.groupby("date").mean()
# oil_test = df_test.oil.groupby("date").mean()
# oil_filled = pd.concat([oil_train, oil_test])
# 
# 
# #plot these against the real oil prices
# ax = df_oil.oil.plot(ax=ax, linewidth=3, label="actual", color="red")
# ax = oil_filled.plot(linewidth=1, label="filled", color="blue")
# plt.show()
# #pretty good interpolation
# plt.close()       


#merge transactions data to df_train
df_trans.set_index(pd.PeriodIndex(df_trans.date, freq="D"), inplace=True)
df_trans.drop("date", axis=1, inplace=True)      
df_trans.rename(columns={"store_nbr":"store_no"}, inplace=True)

df_train = df_train.merge(df_trans, how="left", on=["date", "store_no"])
#no rows added
#no wrong columns added
df_train = df_train[['id', "sales", "transactions", 'category', 'onpromotion', 'city', 'state', 'store_no',
       'store_type', 'store_cluster', 'oil', 'local_holiday', 'regional_holiday', 'national_holiday', 'event',
       'new_years_day', 'payday', 'earthquake', "christmas"]]

pd.isnull(df_train.transactions).sum()
#245784 NAs 

df_train.transactions.fillna(0, inplace=True)
#none left




#final checks


#shapes
df_train.shape
#(3000888, 21)
df_test.shape
#(28512, 17)


#columns
df_train.columns
df_test.columns
#all there


del df_events_merge, df_local_merge, df_national_merge, df_regional_merge, df_train2


#save modified train and test data
df_train.to_csv("./ModifiedData/train_modified.csv", index=True, encoding="utf-8")
df_test.to_csv("./ModifiedData/test_modified.csv", index=True, encoding="utf-8")
#remember to reset dates to index when loading back











#TIME SERIES ANALYSIS


#load back modified data
df_train = pd.read_csv("./ModifiedData/train_modified.csv", encoding="utf-8")
df_test = pd.read_csv("./ModifiedData/test_modified.csv", encoding="utf-8")


#convert dates to indexes again
df_train.set_index(pd.PeriodIndex(df_train.date, freq="D"), inplace=True)
df_train.drop("date", axis=1, inplace=True)

df_test.set_index(pd.PeriodIndex(df_test.date, freq="D"), inplace=True)
df_test.drop("date", axis=1, inplace=True)




#TIME COMPONENTS OF AGGREGATE SALES SERIES


#aggregated sales data
sales_agg = df_train.groupby("date").sales.sum()
sales_agg = pd.DataFrame(data={
  "sales": sales_agg.values,
  "time": np.arange((len(sales_agg)))
}, index=sales_agg.index)
sales_agg.time = sales_agg.time + 1 


#inflation adjustment, base=2010
cpi = [100, 112.8, 116.8, 121.5, 123.6]
sales_agg.sales.loc[sales_agg.index.year==2013] = sales_agg.sales.loc[sales_agg.index.year==2013] / cpi[1]*cpi[0]
sales_agg.sales.loc[sales_agg.index.year==2014] = sales_agg.sales.loc[sales_agg.index.year==2014] / cpi[2]*cpi[0]
sales_agg.sales.loc[sales_agg.index.year==2015] = sales_agg.sales.loc[sales_agg.index.year==2015] / cpi[3]*cpi[0]
sales_agg.sales.loc[sales_agg.index.year==2016] = sales_agg.sales.loc[sales_agg.index.year==2016] / cpi[4]*cpi[0]
sales_agg.sales.loc[sales_agg.index.year==2017] = sales_agg.sales.loc[sales_agg.index.year==2017] / cpi[4]*cpi[0]



#time components plots
sns.set_theme(style="darkgrid")


#average daily sales across time, across all categories
ax = sns.lineplot(x="time", y="sales", label="sales", data=sales_agg)
ax.set_title("agg sales over time")
plt.show()
plt.close()
#generally increasing sales across time, slightly exponential
  #less stronger trend with adjusted sales
#dips at time 1, 365, 1458, 729, 1093
sales_agg.index[sales_agg.time.isin([1,365,729,1093,1458])]
#these are all jan 1, already flagged with a binary feature


#avg sales each month of the year, across all categories
ax = sns.lineplot(x=sales_agg.index.month, y="sales", hue=sales_agg.index.year, data=sales_agg, legend="brief")
ax.set_title("month of year")
plt.show()
plt.close()
#sales peak at december
#they are generally stable before december, 
  #except for 2015 which had a sharp increase in may and june
  #and 2014 where sales fluctuated considerably month to month
#overall, sales grow each year



#avg sales each week of the year, across all categories
ax = sns.lineplot(x=sales_agg.index.week, y="sales", hue=sales_agg.index.year, data=sales_agg)
ax.set_title("week of year")
plt.show()
plt.close()
#similar weekly seasonality to month.
#peak in 2016 weeks 16-18, due to earthquake, accounted for with binary feature
sales_agg.sales.loc[sales_agg.index.week==32]


#avg sales each day of the year, across all categories
ax = sns.lineplot(x=sales_agg.index.dayofyear, y="sales", hue=sales_agg.index.year, data=sales_agg)
ax.set_title("day of year")
plt.show()
#dip at new years day, stable fluctuation across year, peak towards end of year
plt.close()





#avg sales each day of the month, across all categories
ax = sns.lineplot(x=sales_agg.index.day, y="sales", hue=sales_agg.index.month, data=sales_agg)
ax.set_title("day of month")
plt.show()
plt.close()
#peak sales at start of month, days 1-3
#then drops until the 15th, slight recovery after 15th
#then drups until roughly 26-27, starts to recover towards the end of month
#different for december: the biggest peak happens at days 21-24 - christmas.
sales_agg.loc[(sales_agg.index.month==12) & (sales_agg.index.day.isin([20,21,22,23,24,30])), :]


#avg sales each day of the week, across all categories
ax = sns.lineplot(x=sales_agg.index.dayofweek, y="sales", hue=sales_agg.index.month, data=sales_agg)
ax.set_title("day of week")
plt.show()
#dips until tuesday, ramps up afterwards and peaks at sunday
plt.close()




#acf plots for avg sales, across time
ax = plot_acf(sales_agg.sales, lags=365)
#lags tend to lose significance around 250 days
plt.show()
plt.close()


ax = plot_acf(sales_agg.sales, lags=90)
#the within-week seasonality is clear and strong. the trend is clear but not very strong
plt.show()
plt.close()


ax = plot_acf(sales_agg.sales, lags=29)
#the within-week seasonality is clear and strong.
plt.show()
plt.close()


ax = plot_acf(sales_agg.sales, lags=7)
#sales at t are most correlated with sales at t - 7 (same day of previous week)
#a U shaped decline and increase in autocorrelation from t to t - 7
plt.show()
plt.close()





#summary of time components for aggregate sales:
  #trend: sales generally increase over time, slightly quadratic
  #seasonality:
    #generally stable across months with fluctuations, until the peak at december (months of year)
      #same story for weeks of year
      #same story for days of year, but a lot of fluctuation
      #new year's drops accounted for by binary feature
    #peak at start of each month, then declines, recovery towards end (days of month)
      #only exception is december, where the peak happens in christmas 21-24, flagged with binary feature
    #dip at wednesday, peak at sunday (days of week)



#time features that can be added for aggregate sales:
  #trend: quadratic time dummy
  #seasonality:
    #months of year (categorical) 
    #weeks of year (fourier). 
      #days of year likely too unstable.
    #days of month (fourier)
    #days of week (categorical)
    #plot a periodogram and decide?




#plot periodogram
def plot_periodogram(ts, detrend='linear', ax=None):
    from scipy.signal import periodogram
    fs = pd.Timedelta("1Y") / pd.Timedelta("1D")
    freqencies, spectrum = periodogram(
        ts,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    if ax is None:
        _, ax = plt.subplots()
    ax.step(freqencies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Ann (1)",
            "Semiann (2)",
            "Qtr (4)",
            "Bimon (6)",
            "Mon (12)",
            "Biwk (26)",
            "Wk (52)",
            "Semiwk (104)",
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax

plot_periodogram(sales_agg.sales)
plt.show()
plt.close()
#try between quarterly and monthly fourier features? 4-12? 6 as the sweet spot?









#TIME SERIES DECOMPOSITION, AGGREGATE SALES


#detrend & deseasonalize with linear reg

#use multiplicative decomposition (or log transform sales and use additive)
y_decomp =  np.log(sales_agg.sales + 0.00001)


#adjust for time and calendar features with linear reg (minus local and regional holidays, because it's aggregate data)

#get calendar dummies
calendar_cols = ['national_calendar_holiday', 'national_actual_holiday', 'event',
       'new_years_day', 'payday', 'earthquake', 'christmas']
x_decomp = df_train[calendar_cols].groupby("date").mean()


#add days of week dummies
days_week = pd.Series((x_decomp.index.dayofweek + 1), index=x_decomp.index)
x_decomp = x_decomp.merge(pd.get_dummies(
  days_week, prefix="weekday", drop_first=True), how="left", on="date"
)


#add trend and seasonality features with deterministic process
dp = DeterministicProcess(
  index=x_decomp.index,
  period=365,
  constant=True,
  order=1,
  fourier=4,
  drop=True
)
x_decomp = x_decomp.merge(dp.in_sample(), how="left", on="date")



# #add months of year dummies
# months_year = pd.Series((x_cal.index.month), index=x_cal.index)
# x_cal = x_cal.merge(pd.get_dummies(
#   months_year, prefix="month", drop_first=True), how="left", on="date"
# )


#create rolling time series split, 5 folds, 15 steps ahead
cv5 = TimeSeriesSplit(n_splits=5, gap=15)
#performs splits with first 283 * k observations:
  #k=1, train=283, test=1401
  #k=5, train=1415, test=269 etc.


#create linear regressor
decomp_lm = LinearRegression(fit_intercept=False)


#perform time series CV 
decomp_cv_res = cross_val_score(estimator=decomp_lm, X=x_decomp, y=y_decomp, scoring=make_scorer(mse), cv=cv5)
np.sqrt(decomp_cv_res)
#7.30514435, 0.29121716, 0.14066632, 0.19445181, 0.14731206  rmsle, for each respective split
np.sqrt(decomp_cv_res[1:4]).mean()
#0.2088 average rmsle excluding the first fold prediction
#mse with log transformed preds = msle with exp back-transformed preds (tested)


#fit decomposer on entire series
decomp_lm.fit(x_decomp, y_decomp)


#decompose the aggregate log sales
pred_decomp = pd.Series(decomp_lm.predict(x_decomp), index=x_decomp.index)
res_decomp = y_decomp - pred_decomp #innovation residuals (multiplicative decomposed)


#reverse log transformations
y_decomp_exp = np.exp(y_decomp)
pred_decomp_exp = np.exp(pred_decomp)




#DECOMPOSITION PREDICTIONS PLOTS


#plot actual vs predicted 2017 agg. sales
ax=sns.lineplot(x=x_decomp.trend, y=y_decomp_exp, label="sales")
sns.lineplot(x=x_decomp.trend, y=pred_decomp_exp.values, label="preds", ax=ax)
ax.set_title("2017 sales vs preds of decomposition model")
plt.show()
plt.close()
#the trend and annual seasonality is decently captured,
#but many drops and some peaks are missed. will likely be helped by lag features


ax = sns.lineplot(x=x_decomp.index.day, y=y_decomp_exp, label="sales")
sns.lineplot(x=x_decomp.index.day, y=pred_decomp_exp.values, label="preds", ax=ax)
ax.set_title("decomposition model preds by days of month")
plt.show()
plt.close()
#appears to have capture days of month fluctuations, 
#except the peaks at the start of the months,
#and the decline through a month


ax = sns.lineplot(x=x_decomp.index.dayofweek, y=y_decomp_exp, label="sales")
sns.lineplot(x=x_decomp.index.dayofweek, y=pred_decomp_exp.values, label="preds", ax=ax)
ax.set_title("decpmposition model preds by day of week")
plt.show()
plt.close()
#days of week fluctuations captured well




#INNOVATION RESIDUALS PLOTS


ax = sns.lineplot(x=x_decomp.trend, y=res_decomp.values)
ax.set_title("inno residuals of decomposition model")
plt.show()
plt.close()
#the increases in 2014 are beyond the model. other than that it looks mostly stationary


#check mean, autocorrelations of the decomposed innovation residuals
"{:.8f}".format(res_decomp.mean())
#innovation residuals have 0 mean. forecasts are not biased.


ax = plot_acf(res_decomp, lags=50)
plt.show()
plt.close()
#highest ACF at lag 1 (not a spike)
#then a slow sinusoidal decline


ax = plot_pacf(res_decomp, lags=50)
plt.show()
plt.close()
#highest PACF at lag 1 (spike)
#then a sharp decline at lag 2, no more spikes


#consider using an ARIMA(1,d,0)



#test stationarity with augmented dickey fuller test

#pre-decomposition aggregated sales
adfuller(y_decomp, autolag="AIC")[0]
#test stat -2.52
adfuller(y_decomp, autolag="AIC")[1].round(6)
#p value 0.111, time series is non-stationary


#decomposed residuals
adfuller_res = adfuller(res_decomp, autolag="AIC")
adfuller_res[0]
#test stat: -4.85
"{:.6f}".format(adfuller_res[1])
#p value 0.000043, time series is very stationary






#EVALUATE LAG FEATURES ON MULTIPLICATIVE DECOMPOSED RESIDUALS


#SALES LAGS


#pacf
plot_pacf(res_decomp, method="ywm", lags=60)
plt.show()
plt.close()
#significant lags:
  #1, 2, 3, 5, 11, 13, 14, 18, 28, 29, 30, 33, 34, 42, 49, 60
#most significant lags from each range:
  #1, 2, 3, 11, 28, 34, 42, 49, 60
  #1 is far more significant than anything else


#scatterplots


#1-6
fig, ax = plot_lags(res_decomp, lags=[i for i in range(1,7)])
plt.show()
plt.close()
#use lag 1 only
#use lag 1, 2, 3?


#most significant lags from 10>
fig, ax = plot_lags(res_decomp, lags=[11, 18, 28, 34, 42, 60])
plt.show()
plt.close()
#could use lag 1 only, others don't seem that different




#COVARIATE LAGS (ONPROMOTION, OIL, TRANSACTIONS)




#create data frame with covariates
sales_covar = pd.DataFrame(res_decomp, columns=["sales"], index=x_decomp.index)
sales_covar["onpromotion"] = df_train.groupby("date").onpromotion.sum()
sales_covar["oil"] = df_train.groupby("date").oil.mean()
sales_covar["trans"] = df_train.groupby("date").transactions.sum()
sales_covar.drop("sales", axis=1, inplace=True)


#check if covariates are stationary

#onpromotion
adfuller(sales_covar["onpromotion"], autolag="AIC")[0]
#test stat -1.1
"{:.6f}".format(adfuller(sales_covar["onpromotion"], autolag="AIC")[1])
#p value 0.72, very non-stationary


#oil
adfuller(sales_covar["oil"], autolag="AIC")[0]
#test stat -0.87
"{:.6f}".format(adfuller(sales_covar["oil"], autolag="AIC")[1])
#p value 0.8, very non-stationary


#transactions
adfuller(sales_covar["trans"], autolag="AIC")[0]
#test stat -6.28
"{:.6f}".format(adfuller(sales_covar["trans"], autolag="AIC")[1])
#p value 0, stationary


#difference the covariates
differencer = Differencer(lags=1)
sales_covar = differencer.fit_transform(sales_covar)
#all covariates now stationary


#center and scale the covariates and decomposed sales
scaler_std = StandardScaler()
sales_covar["sales"] = res_decomp.values
sales_covar_scaled = scaler_std.fit_transform(sales_covar.values)
sales_covar = pd.DataFrame(sales_covar_scaled, columns=sales_covar.columns, index=x_decomp.index)
sales_covar["time"] = x_decomp["trend"]



#ONPROMOTION


#lineplot of scaled agg sales and onpromotion, by days of month
ax = sns.lineplot(data=sales_covar, x=sales_covar.index.day, y="sales", label="sales")
sns.lineplot(ax=ax, data=sales_covar, x=sales_covar.index.day, y="onpromotion", label="onpromotion")
plt.show()
plt.close()
#onpromotion barely moves compared to sales


#lag 0
pearsonr(sales_covar.sales, sales_covar.onpromotion)
#zero pearson
spearmanr(sales_covar.sales, sales_covar.onpromotion)
#zero spearman
"{:.6f}".format(adjusted_mutual_info_score(sales_covar.onpromotion.values, sales_covar.sales.values))
#zero MI


#cross correlations for 365 lags
sales_prom_ccf = ccf(sales_covar.sales, sales_covar.onpromotion, adjusted=False)[0:365]
ax=sns.lineplot(x=range(0, len(sales_prom_ccf)), y=sales_prom_ccf)
ax.set_title("cross correlation of sales and onpromotion")
ax.set_xlabel("lag")
ax.set_ylabel("corr")
plt.show()
plt.close()
#none
#sum of onpromotion is likely not predictive for aggregate sales




#OIL


#lineplot of scaled agg sales and onpromotion, by weeks of year
ax = sns.lineplot(data=sales_covar, x=sales_covar.index.week, y="sales", label="sales")
sns.lineplot(ax=ax, data=sales_covar, x=sales_covar.index.week, y="oil", label="oil")
plt.show()
plt.close()
#no clear relation


#lag 0
pearsonr(sales_covar.sales, sales_covar.oil)
#zero pearson, but very high p value
spearmanr(sales_covar.sales, sales_covar.oil)
#zero spearman, very significant p
"{:.6f}".format(adjusted_mutual_info_score(sales_covar.sales, sales_covar.oil))
#zero MI


#cross correlations for 365 lags
sales_oil_ccf = ccf(sales_covar.sales, sales_covar.oil, adjusted=False)[0:730]
ax=sns.lineplot(x=range(0, len(sales_oil_ccf)), y=sales_oil_ccf)
ax.set_title("cross correlation of sales and oil")
ax.set_xlabel("lag")
ax.set_ylabel("corr")
plt.show()
plt.close()
#zero


#consider moving averages of oil as predictor of aggregate sales
oil_56=sales_covar.oil.rolling(56).mean()
oil_28=sales_covar.oil.rolling(28).mean()
oil_21=sales_covar.oil.rolling(21).mean()
oil_14=sales_covar.oil.rolling(14).mean()
oil_7=sales_covar.oil.rolling(7).mean()
oil_ma = pd.DataFrame(
  data={"oil_28":oil_28.values,
  "oil_14":oil_14.values,
  "oil_7":oil_7.values},
  index=sales_covar.index
)


#21-day ma
ax = sns.regplot(x=oil_21, y=sales_covar.sales, label="MA21")
ax.set_title("moving average 28")
plt.show()
plt.close()
#somewhat declining relationship. seems to be the sweet spot


pearsonr(sales_covar.sales, oil_56.fillna(method="bfill"))
#-0.06 pearson, very small p val
spearmanr(sales_covar.sales, oil_56.fillna(method="bfill"))
#-0.08 spearman, very small pval

pearsonr(sales_covar.sales, oil_28.fillna(method="bfill"))
#-0.18 pearson, very small p val
spearmanr(sales_covar.sales, oil_28.fillna(method="bfill"))
#-0.17 spearman, very small pval

pearsonr(sales_covar.sales, oil_21.fillna(method="bfill"))
#-0.17 pearson, very small p val
spearmanr(sales_covar.sales, oil_21.fillna(method="bfill"))
#-0.16 spearman, very small pval

pearsonr(sales_covar.sales, oil_14.fillna(method="bfill"))
#-0.14 pearson, very small p val
spearmanr(sales_covar.sales, oil_14.fillna(method="bfill"))
#-0.13 spearman, very small pval

pearsonr(sales_covar.sales, oil_7.fillna(method="bfill"))
#-0.1 pearson, very small p val
spearmanr(sales_covar.sales, oil_7.fillna(method="bfill"))
#-0.1 spearman, very small pval




#TRANSACTIONS (CONSIDER LAG 15+)


#lineplot of scaled agg sales and trans, by weeks of year
ax = sns.lineplot(data=sales_covar, x=sales_covar.index.day, y="sales", label="sales")
sns.lineplot(ax=ax, data=sales_covar, x=sales_covar.index.day, y="trans", label="trans")
plt.show()
plt.close()
#seems to vaguely follow the monthly pattern of sales?


#correlations


#lag 0
pearsonr(sales_covar.sales, sales_covar.trans)
#0.12 pearson
spearmanr(sales_covar.sales, sales_covar.trans)
#0.13 spearman
"{:.6f}".format(adjusted_mutual_info_score(sales_covar.sales, sales_covar.trans))
#zero MI


#cross correlations for 365 lags
sales_trans_ccf = ccf(sales_covar.sales, sales_covar.trans, adjusted=False)[0:730]
ax=sns.lineplot(x=range(0, len(sales_trans_ccf)), y=sales_trans_ccf)
ax.set_title("cross correlation of sales and trans")
ax.set_xlabel("lag")
ax.set_ylabel("corr")
plt.show()
plt.close()
#highest correlation of 0.11 at lag 0, then drops and stays low




#FIT FULL MODEL WITH LAGGED PREDICTORS

#outcome series: y_decomp

#features:
  #target lag 1, 2, 3, 5, 11, 28
  #oil_21

#function to make lags
def make_lags(ts, lags, prefix):
  return pd.concat(
  {
    f"{prefix}_lag_{i}": ts.shift(i)
    for i in range(1, lags+1)
  },
  axis=1
  )


#make lags (MISTAKE: )
res_lags = make_lags(res_decomp, lags=28, prefix="sales")
res_lags = res_lags.iloc[:,[0,1,2,4,10,27]]


#combine lags with oil_21
x_train = res_lags.merge(oil_21, how="left", left_index=True, right_index=True)
x_train = x_train.fillna(x_train.median())


#combine x_train with x_decomp
x_train = x_train.merge(x_decomp, how="left", left_index=True, right_index=True)


#linear regression
lm = LinearRegression(fit_intercept=False)


#crossvalidate model
cv_res = cross_val_score(estimator=lm, X=x_train, y=y_decomp, scoring=make_scorer(mse), cv=cv5)
np.sqrt(cv_res)
#6.65689981, 0.12895082, 0.09362015, 0.11907198, 0.09725895 rmsle respectively
np.sqrt(cv_res[1:4]).mean()
#0.1138 aggregated rmsle except first fold


#fit model on pre 2017 data, evaluate and plot it on 2017 data
x_fit = x_train.loc[x_train.index.year!=2017]
y_fit = y_decomp.loc[y_decomp.index.year!=2017]
x_valid = x_train.loc[x_train.index.year==2017]
y_valid = y_decomp.loc[y_decomp.index.year==2017]
lm.fit(x_fit, y_fit)
y_pred = pd.Series(lm.predict(x_valid), index=y_valid.index)
np.sqrt(mse(y_pred, y_valid))
#rmsle 0.092 on 2017


#plot 2017 predictions
ax = sns.lineplot(x=y_valid.index.dayofyear, y=y_valid, label="actual")
sns.lineplot(ax=ax, x=y_valid.index.dayofyear, y=y_pred, label="predicted")
ax.set_title("actual vs predicted agg sales 2017")
plt.show()
plt.close()
#not bad overall, misses the post jan 1 peak, doesn't fully capture the magnitude of some peaks and troughs


ax = sns.lineplot(x=y_valid.index.week, y=y_valid, label="actual")
sns.lineplot(ax=ax, x=y_valid.index.week, y=y_pred, label="predicted")
ax.set_title("actual vs predicted agg sales 2017")
plt.show()
plt.close()
#follows the annual trend and season well, but misses the magnitude of some peaks and troughs


ax = sns.lineplot(x=y_valid.index.day, y=y_valid, label="actual")
sns.lineplot(ax=ax, x=y_valid.index.day, y=y_pred, label="predicted")
ax.set_title("actual vs predicted agg sales 2017")
plt.show()
plt.close()
#follows the within month moves well, but doesn't fully capture the magnitudes


ax = sns.lineplot(x=y_valid.index.dayofweek, y=y_valid, label="actual")
sns.lineplot(ax=ax, x=y_valid.index.dayofweek, y=y_pred, label="predicted")
ax.set_title("actual vs predicted agg sales 2017")
plt.show()
plt.close()
#follows the within week moves well, 
#but underestimates the wednesday drop and overestimates the weekend peak




#BUILD PIPELINE, RECONCILE TOP-DOWN


#to create a hierarchy, you need to convert data to wide format, with one row per one date
#with darts:
  #first add all features you want to df_train and df_test. handle NAs
    #add oil_ma21
    #sales lags (1 2 3 5 11 28 in old attempt) will be specified in RegressionModel (remember to handle NAs)
    #instead of adding manual trend and fourier features, apply FFT with trend (due to memory issues)
  #convert all booleans to integers
  #get rid of non-features (id)
  #then convert df_train and df_test to wide (see bottom of notes). handle NAs 
  #split the data into:
    #target: darts time series with one row per date, one column per:
      #total sales that date
      #each category's total sales that date (will add up to Total)
      #each store type's total sales that date (will add up to Total)
      #each category N-store K combo's total sales that date (will add to category no N and store no K)
    #past_covariates: darts time series with one row per date, one column for:
      #oil moving average 21
      #calendar dummies
      #days of week dummies
    #static covariates: don't have any.
      #IMPORTANT, DOCUMENT THIS: static covariate=does not change over time. useful only to differentiate multiple time series
        #darts takes static covariates as: data frame with 1 row per one target time series, 1 column per static covariate
        #you have to specify a static covariate for each time series. so we don't have any applicable in our data
    #add hierarchy dictionary to the target time series https://unit8co.github.io/darts/examples/16-hierarchical-reconciliation.html?highlight=hierarchical
    

#HIERARCHICAL STRUCTURE OF THE TIME SERIES
#sales in one day
  #by category: 33
  #by location: 
    #state: 16
      #city: 22
       #store no: 54
  #by store type: 5
    #store cluster: 17
      

#levels of aggregation, from lowest to highest: 
  #category & location:
    #sale of each category in each store: 1782 series
    #sale of each category in each city: 726 series
    #sale of each category in each state: 528 series
  #category & store type
    #sale of each category in each store cluster: 561 series
    #sale of each category in each store type: 165 series
  #location:
    #all categories, each store: 54 series
    #all categories, each city: 22 series
    #all categories, each state: 16 series
  #store type:
    #all categories, each cluster: 17 series
    #all categories, each store type: 5 series
  #category: 
    #each category: 33 series

    
#load back modified data
df_train = pd.read_csv("./ModifiedData/train_modified.csv", encoding="utf-8")
df_test = pd.read_csv("./ModifiedData/test_modified.csv", encoding="utf-8")


#combine df_train and df_test to add features
df = pd.concat([df_train, df_test])
df.index = pd.RangeIndex(start=0, stop=len(df), step=1)


#set datetime index
df = df.set_index(pd.to_datetime(df.date))
df = df.drop("date", axis=1)


#set bools to int
df = df * 1


#drop id
df = df.drop("id", axis=1)


#add 21-day oil moving average (calculate from aggregated oil prices to avoid different values in same day)
df_oil = pd.read_csv("./OriginalData/oil.csv", encoding="utf-8")
df_oil.rename(columns={"dcoilwtico":"oil"}, inplace=True)
df_oil.set_index(pd.PeriodIndex(df_oil.date, freq="D"), inplace=True)
df_oil.drop("date", axis=1, inplace=True)

oil_21 = df_oil.rolling(21).mean().fillna(method="bfill")
oil_21 = oil_21.rename(columns={"oil":"oil_21"})
oil_21.index = df_oil.index
oil_21 = oil_21.set_index(pd.to_datetime(oil_21.index.to_timestamp()))
df = df.join(oil_21, how="left", on="date")
df.oil_21 = df.oil_21.interpolate(method="time")


#check NAs
pd.isnull(df).sum()
#28512 sales and transactions NAs (test data), drop after split


#add category:store_no column
df["category_store_no"] = df["category"].astype(str) + "-" + df["store_no"].astype(str)


#split df_train and df_test again
df_train = df.iloc[0:3000888,:]
df_test = df.iloc[3000888:len(df),:]


#drop sales and transactions from df_test
df_test = df_test.drop(columns=["sales", "transactions"], axis=1)


#check columns and shapes
df_train.columns
df_test.columns

df_train.shape
df_test.shape


#sales CPI adjustment, base(100)=2010
cpi = dict(
  {2013:112.8,
  2014:116.8,
  2015:121.5,
  2016:123.6,
  2017:123.6}
  )

for year in cpi.keys():
  df_train.loc[df_train.index.year==year, "sales"] = df_train.loc[df_train.index.year==year, "sales"] / cpi[year] * 100
del year


#minmax scale sales before aggregation
from sklearn.preprocessing import MinMaxScaler
scaler_sk = MinMaxScaler(feature_range=(0,1))

df_train.sales = pd.Series(
  data=scaler_sk.fit_transform(df_train.sales.values.reshape(-1,1)).reshape(1,-1).tolist()[0],
  index=df_train.sales.index
)

df_train.sales
min(df_train.sales)
max(df_train.sales)
#yes,that should work




#create wide dataframes with dates as rows, values for total, category, store and category_store sales as columns


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


#now merge it all
wide_frames = [total, category, store_no, category_store_no]
ts_train = reduce(lambda left, right: pd.merge(
  left, right, how="left", on="date"), wide_frames)
del total, category, store_no, wide_frames, category_store_no


# #minmax scale all time series, BUT ALL WITH THE TOTAL SALES' PARAMETERS
#   #if you scale every series within itself, they won't be summable
#   #0-1 results in negatives for small series. use 0-100
# from sklearn.preprocessing import MinMaxScaler
# scaler_sk = MinMaxScaler(feature_range=(0,100))
# 
# 
# #fit and transform total sales
# ts_train
# 
# ts_train.sales = pd.Series(
#   data=scaler_sk.fit_transform(ts_train.sales.values.reshape(-1,1)).reshape(1,-1).tolist()[0],
#   index=ts_train.sales.index)
# 
# 
# #transform other series
# ts_train.iloc[:,1:1870]
# 
# for i in range(1,1870):
#   ts_train.iloc[:,i] = pd.Series(
#   data=scaler_sk.transform(ts_train.iloc[:,i].values.reshape(-1,1)).reshape(1,-1).tolist()[0],
#   index=ts_train.iloc[:,i].index
#   )
# 
# ts_train



#create target series
ts_train = darts.TimeSeries.from_dataframe(
  ts_train, freq="D", fill_missing_dates=False)


#add hierarchy dictionary to target time series

#lists of hierarchy nodes
categories = df_train.category.unique().tolist()
stores = df_train.store_no.unique().astype(str).tolist()
len(categories) #33
len(stores) #54

#empty dict
ts_hierarchy = dict()

#categories to sales loop
for category in categories:
  ts_hierarchy[category] = ["sales"]

#stores to sales loop
for store in stores:
  ts_hierarchy[store] = ["sales"]

#category-store combos to category and stores
for category, store in product(categories, stores):
  ts_hierarchy["{}-{}".format(category, store)] = [category, store]
  
ts_hierarchy
#beautiful

#map hierarchy to ts_train
ts_train = ts_train.with_hierarchy(ts_hierarchy)
ts_train.hierarchy

del category, store




#create past covariates time series

#get calendar dummies
calendar_cols = ['national_holiday', 'event',
       'new_years_day', 'payday', 'earthquake', 'christmas']
past_train = df_train[calendar_cols].groupby("date").mean()

#add oil
past_train["oil_21"] = df_train.oil_21.groupby("date").mean()


#find the missing dates
data_dates = past_train.index.values.astype(str).tolist()
full_dates = pd.date_range(start="2013-01-01", end="2017-08-15").values.astype(str).tolist()
missing_dates = list(set(full_dates) - set(data_dates))
#missing dates: 25 dec 2013, 25 dec 2014, 25 dec 2015, 25 dec 2016


#add these to the future covariates manually.
  #for sales series, interpolate in darts
append_series = pd.DataFrame(
  data={
    "national_holiday":[0,0,0,0],
    "event":[0,0,0,0],
    "new_years_day":[0,0,0,0],
    "payday":[0,0,0,0],
    "earthquake":[0,0,0,0],
    "christmas":[1,1,1,1]
  },
  index=pd.to_datetime(missing_dates)
  )
past_train = past_train.append(append_series).sort_index()
past_train.index.names = ["date"]

#interpolate oil NAs for manually added dates
past_train["oil_21"] = past_train["oil_21"].interpolate(method="time")
pd.isnull(past_train).sum()  
#done


#CPI adjust oil
for year in cpi.keys():
  past_train.loc[past_train.index.year==year,"oil_21"] = past_train.loc[past_train.index.year==year, "oil_21"] / cpi[year] * 100
del year

#difference oil
differencer = Differencer(lags=1)
past_train["oil_21"] = differencer.fit_transform(past_train["oil_21"])

#minmax scale oil
from sklearn.preprocessing import MinMaxScaler
scaler_sk = MinMaxScaler()
past_train.oil_21 = pd.Series(
  data=scaler_sk.fit_transform(past_train.oil_21.values.reshape(-1,1)).reshape(1,-1).tolist()[0],
  index=past_train.oil_21.index)


#get time features
from statsmodels.tsa.deterministic import DeterministicProcess
dp = DeterministicProcess(
  index=past_train.index,
  period=365,
  constant=True,
  order=1,
  fourier=4,
  drop=True
)
past_train = past_train.merge(dp.in_sample(), how="left", on="date")


#add days of week dummies
days_week = pd.Series((past_train.index.dayofweek + 1), index=past_train.index)
past_train = past_train.merge(pd.get_dummies(
  days_week, prefix="weekday", drop_first=True), how="left", on="date"
)



#rename into future covariates because darts accepts 0 lag only for future covariates
future_train = past_train
pd.isnull(future_train).sum()


# #split into past and future covariates
# 
# #future:
#   #calendar features
#   #trend, const, fourier
# future_train = past_train.drop("oil_21", axis=1)
# 
# #past:
#   #oil_21
# past_train = pd.DataFrame(
#   data=past_train["oil_21"],
#   index=past_train.index,
#   columns=["oil_21"])
# #shift back 1 period because darts wants past covariates to have at least 1 lag
# past_train["oil_21"] = past_train["oil_21"].shift(1).fillna(method="bfill")
# 

#convert into time series
# past_train = darts.TimeSeries.from_dataframe(
#   past_train, freq="D", fill_missing_dates=False)

future_train = darts.TimeSeries.from_dataframe(
  future_train
)




#MODELING PIPELINE


#PREPROCESSING STEPS


#interpolate sales for missing dates (25 december 2013-2016)
from darts.dataprocessing.transformers import MissingValuesFiller
filler = MissingValuesFiller()
ts_train = filler.transform(ts_train)

#check missing values
from darts.utils.missing_values import missing_values_ratio
missing_values_ratio(ts_train)
missing_values_ratio(future_train)
#handled


#final check
ts_train
ts_train["sales"]
ts_train["AUTOMOTIVE"]
ts_train["1"]
ts_train["AUTOMOTIVE-1"]
#looks good


# #log transform sales series
# def log_zero(x):
#   return np.log(x+0.0001)
# 
# def log_zero_inv(x):
#   return (np.exp(x)-0.0001)
# 
# transform_log = Mapper(log_zero)
# reverse_log = Mapper(log_zero_inv)
# ts_train = transform_log.transform(ts_train)
# IMPORTANT: LOG TRANSFORMING EACH SERIES MESSES UP THE MULTIVARIATE TRAINING - RECONCILIATION
  #IT'S BECAUSE SUMMING UP TWO LOGARITHMS = MULTIPLYING THEM. ANY WORKAROUND TO THIS?


#centering and scaling of sales
#scaler_minmax = Scaler()
# ts_train["sales"] = scaler_minmax.fit_transform(ts_train["sales"])
#IMPORTANT: IF YOU SCALE THE MULTIVARIATE SERIES, YOU HAVE TO FIT ON THE TOP LEVEL ONLY!
  #otherwise every series will be 0-1 scaled within itself and won't be comparable
  
  
  
#MODELING STEPS FOR AGGREGATE SALES


#linear regression with past covariates, sales lags
from darts.models.forecasting.linear_regression_model import LinearRegressionModel
model_lm = LinearRegressionModel(
  lags=[-28, -11, -5, -3, -2, -1],
  lags_future_covariates=[0],
  output_chunk_length=15,
  fit_intercept=False
)
#the model will predict 15 days into the future at a time


#train-test crossval with pre-post 2017
  #2017: last 227 obs

#split pre-post 2017
y_train, y_val = ts_train[:-227], ts_train[-227:]
x_train_future, x_val_future = future_train[:-227], future_train[-227:]


#predict 2017 with lm
model_lm.fit(y_train["sales"], future_covariates=x_train_future)
pred_lm = model_lm.predict(future_covariates=x_val_future, n=227)
#the forecast horizon in predict is 227 (all of 2017), 
#but the model's output chunk length is 15.
#in this case, the model will predict the 227 periods 15 at a time,
  #autoregressively building on its own predictions


#score 2017 lm predictions
# from darts.metrics import rmse
from darts.metrics import rmsle
rmsle(y_val["sales"], pred_lm)
#rmsle 0.1353
  #keep in mind these metrics are for scaled data, and may underestimate the real error
  #though they can be comparable with the same scaling applied


#rolling crossval, starting with 2013 as training data, predicting next 15, moving training forward 15 days
model_lm.backtest(
  ts_train["sales"], future_covariates=future_train, start=366, forecast_horizon=15, stride=15, metric=rmsle)
#agg rmsle 0.152


#plot 2017 actual vs. fft preds
y_val["sales"].plot(label="actual")
pred_lm.plot(label="lm preds")
plt.show()
plt.close()
#accounts for the seasonality pattern well
#misses the magnitude of fluctuations, especially some peaks and drops by quite a bit without the log transform


#get and inspect inno residuals of 2017
res_lm = y_val["sales"] - pred_lm


#time plot, distribution, acf
from darts.utils.statistics import plot_residuals_analysis
plot_residuals_analysis(res_lm)
plt.show()
plt.close()
#residuals seem stationary except for the new years day drop and a few peaks
#ACF value significant for lag 1. then declines immediately
  #with log transformed sales, it was lags 1, 2, 3, 4, and an exponential decline
#distribution very centered around 0 except for the new years day outlier


#pacf
from darts.utils.statistics import plot_pacf
plot_pacf(res_lm, max_lag=48)
plt.show()
plt.close()
#PACF spike in lag 1, then a few far away significants in 27, 28, 30, 34...
  #with log transformation PACF spike in lag 1, lag 3 is barely significant, rest are insignificant


#kpss test for stationarity
from darts.utils.statistics import stationarity_test_kpss as kpss
from darts.utils.statistics import stationarity_test_adf as adf
kpss_res = kpss(res_lm)
kpss_res
#test stat 0.46, p value 0.05, data is barely stationary
  #with log transform, test stat 0.68, p val 0.015, data is non-stationary
adf_res = adf(res_lm)
adf_res
#test stat -11.34, p value very small, data is stationary
  #with log transform, test stat -5.58, p val very small, data is stationary
#the series is stationary  
  #with log transform, ADF is stationary and KPSS is not, the series is difference stationary
  #if both tests give stationary or not, the series is stationary or not,
  #if ADF gives non-stationary and KPSS gives stationary, the series is stationary around a trend - remove it.
  #if ADF gives stationary and KPSS gives non-stationary, the differenced series is stationary.




#FIT ON MULTIVARIATE SERIES, PERFORM RECONCILIATION


#check if your hierarchy is properly mapped: do components add up as they should?

##categories and total
# sum_categories = (
#   sum([ts_train[category] for category in categories])
# )
# 
# 
# sum_categories.plot(label="sum of categories")
# ts_train["sales"].plot(label="total sales")
# plt.show()
# plt.close()
# #no they don't. is this because of the log transformation?
# #try again with log transformations reversed
# 
# #reverse log transformation
# ts_train_exp = reverse_log.transform(ts_train)
# 
# sum_categories_exp = (
#   sum([ts_train_exp[category] for category in categories])
# )
# 
# 
# sum_categories_exp.plot(label="sum of categories")
# ts_train_exp["sales"].plot(label="total sales")
# plt.show()
# plt.close()
# #now they add up. don't use the log transform!



#fit and predict all nodes of the hierarchy, pre-post 2017 split
categories_stores = []
for category, store in product(categories, stores):
  categories_stores.append("{}-{}".format(category, store))

# 
# nodes = ["sales", categories, stores, categories_stores]
# y_train_nodes = [y_train[nodes[0]], y_train[nodes[1]], y_train[nodes[2]], y_train[nodes[3]]]
# 
# 
# 
# model_lm.fit(
#   series=[y_train[nodes[0]], y_train[nodes[1]], y_train[nodes[2]], y_train[nodes[3]]], 
#   future_covariates=x_train_future)
#   
#   
# pred_nodes = [
#   model_lm.predict(series=y_val[nodes][0], future_covariates=x_val_future, n=227),
#   model_lm.predict(series=y_val[nodes][1], future_covariates=x_val_future, n=227),
#   model_lm.predict(series=y_val[nodes][2], future_covariates=x_val_future, n=227),
#   model_lm.predict(series=y_val[nodes][3], future_covariates=x_val_future, n=227)
#   ]
# 
# 

#training the entire multivariate set at once
model_lm.fit(y_train, future_covariates=x_train_future)
pred_full = model_lm.predict(future_covariates=x_val_future, n=227)

y_val["sales"]
pred_full["sales"]
y_val["AUTOMOTIVE-1"]
pred_full["AUTOMOTIVE-1"]
#seems realistic


#plot some predictions by node of hierarchy

#aggregate series
y_val["sales"].plot(label="actual")
pred_full["sales"].plot(label="lm preds")
plt.title("total_sales")
plt.show()
plt.close()
#predictions are in the correct scale now, but fluctuate way too much
  #with log transformation the predictions for agg sales are in the wrong scale
  #likely due to the nature of addition with logarithms (which is multiplication)


#categories
y_val[categories[5]].plot(label="actual")
pred_full[categories[5]].plot(label="lm preds")
plt.title("category_sales")
plt.show()
plt.close()
#scale correct, seasonality mostly matches, but too much fluctuation


#stores
y_val[stores[9]].plot(label="actual")
pred_full[stores[9]].plot(label="lm preds")
plt.title("store_sales")
plt.show()
plt.close()
#scale correct, too much fluctuation, preds too low in general


#category-store
y_val["BREAD/BAKERY-9"].plot(label="actual")
pred_full["BREAD/BAKERY-9"].plot(label="lm preds")
plt.title("category_store_sales")
plt.show()
plt.close()
#scale correct, trend correct, too much fluctuation
  #though the scale is very small. maybe fluctuation is much less on the true scale?



#see how the nodes sum up before reconciliation
def plot_forecast_sums(pred_series):
    plt.figure(figsize=(10, 5))

    pred_series["sales"].plot(label="total", alpha=0.3, color="grey")
    sum([pred_series[r] for r in categories]).plot(label="sum of categories")
    sum([pred_series[r] for r in stores]).plot(label="sum of stores")
    sum([pred_series[t] for t in categories_stores]).plot(
        label="sum of categories_stores"
    )

    legend = plt.legend(loc="best", frameon=1)
    frame = legend.get_frame()
    frame.set_facecolor("white")


plot_forecast_sums(pred_full)
plt.show()
plt.close()
#the nodes sum up perfectly. how?




#score predictions for each node of the hierarchy


from statistics import stdev
from statistics import fmean
  
# def measure_rmse(val, pred, subset):
#   return rmse([val[c] for c in subset], [pred[c] for c in subset])


#rmsle throws error for negative predictions, so we 0-1 scale the predictions
  #NOTE: SCALED SERIES WON'T BE CORRECTLY COMPARABLE-SUMMABLE WITH ONE ANOTHER
  #to ensure they will be correctly comparable here, we fit it on predictions only
# scaler_minmax = Scaler()
# pred_full_scaled = scaler_minmax.fit_transform(pred_full)
# y_val_scaled = scaler_minmax.transform(y_val)

#sanity check
# y_val["sales"][0] #9775.48624353
# pred_full["sales"][0] #839241.83111653
# y_val_scaled["sales"][0] #0.11570099
# pred_full_scaled["sales"][0] #0.52329089
#yes, that should work


def measure_rmsle(val, pred, subset):
  return rmsle([val[c] for c in subset], [pred[c] for c in subset])


#scores

measure_rmsle(y_val, pred_full, ["sales"])
#mean rmsle of total sales 0.5763

fmean(measure_rmsle(y_val, pred_full, categories))
stdev(measure_rmsle(y_val, pred_full, categories))
#mean rmsle across categories 0.08, sd 0.13, +2sd 0.34

fmean(measure_rmsle(y_val, pred_full, stores))
stdev(measure_rmsle(y_val, pred_full, stores))
#mean rmsle across stores 0.076, sd 0.04, +2sd 0.156

fmean(measure_rmsle(y_val, pred_full, categories_stores))
stdev(measure_rmsle(y_val, pred_full, categories_stores))
#mean rmsle across category-store combos 0.0037, sd 0.0095, +2sd = 0.023




#WITHOUT RECONCILIATION
#the hierarchy nodes sum perfectly,
  #error is very large at the top node, gets much smaller down the hierarchy
  #does this mean darts uses automatic bottom-up reconciliation?
  #or maybe it only fits on the lowest nodes of the hierarchy?







#TOP DOWN RECONCILIATION
from darts.dataprocessing.transformers.reconciliation import TopDownReconciliator
topdown_reconciler = TopDownReconciliator()
topdown_reconciler.fit(y_train)
pred_topdown = topdown_reconciler.transform(pred_full)


#see how nodes sum up after reconciliation
plot_forecast_sums(pred_topdown)
plt.show()
plt.close()
#nothing changed?




#score preds of each node
measure_rmsle(y_val, pred_topdown, ["sales"])
#0.5763 rmsle on total sales
  #same as unreconciled


fmean(measure_rmsle(y_val, pred_topdown, categories))
stdev(measure_rmsle(y_val, pred_topdown, categories))
#mean rmsle across categories 0.06, sd 0.1, +2sd 0.26
  #lower than unreconciled


fmean(measure_rmsle(y_val, pred_topdown, stores))
stdev(measure_rmsle(y_val, pred_topdown, stores))
#mean rmsle across stores 0.06, sd 0.03, +2sd 0.12
  #lower than unreconciled


fmean(measure_rmsle(y_val, pred_topdown, categories_stores))
stdev(measure_rmsle(y_val, pred_topdown, categories_stores))
#mean rmsle across category-store combos 0.0024, sd 0.006, +2sd 0.014
  #lower than unreconciled








































# #apply FFT
# from darts.models import FFT
# model_fft = FFT(
#   nr_freqs_to_keep=4,
#   trend="poly",
#   trend_poly_degree=1
# )
# 
# 
# #train-test crossval with pre-post 2017
#   #2017: last 227 obs
# 
# #split pre-post 2017
# y_train, y_val = ts_train[:-227], ts_train[-227:]
# x_train, x_val = past_train[:-227], past_train[-227:]
# 
# #predict 2017 with fft
# model_fft.fit(y_train["sales"])
# pred_fft = model_fft.predict(n=227)
# 
# 
# #score 2017 fft predictions
# from darts.metrics import rmse
# rmse(y_val["sales"], pred_fft)
# #rmsle 0.0607
#   #better than manual approach's decomposition and hybrid model
# 
# 
# #5 fold rolling crossval
# model_fft.backtest(
#   ts_train["sales"], start=0.2, forecast_horizon=15, stride=337, metric=rmse)
# #agg rmsle 0.03646
# 
# 
# #plot 2017 actual vs. fft preds
# y_val["sales"].plot(label="actual")
# pred_fft.plot(label="fft preds")
# plt.show()
# plt.close()
# #follows the annual seasonality pattern, but sometimes early-late
# #seems to have a slightly increasing trend compared to actual, possibly due to spikes
# #not comparable directly with previous composition as it also included calendar features
# 
# 
# #get and inspect inno residuals of 2017
# res_fft = y_val["sales"] - pred_fft
# 
# 
# #time plot, distribution, acf
# from darts.utils.statistics import plot_residuals_analysis
# plot_residuals_analysis(res_fft)
# plt.show()
# plt.close()
# #looks stationary and patternless except for NY drop
# #acf shows clear sinusoid pattern, no spikes, few significant lags
# #distribution normal except for NY outlier
# 
# 
# #pacf
# from darts.utils.statistics import plot_pacf
# plot_pacf(res_fft, max_lag=48)
# plt.show()
# plt.close()
# #declining sinusoidal pattern, almost completely disappears by lag 50
# #only slightly significant lags: 3, 7
# #looks like the manual fourier transform didn't account for all the seasonality?
# 
# 
# #kpss test for stationarity
# from darts.utils.statistics import stationarity_test_kpss as kpss
# kpss(res_fft)
# #test stat 0.19, p value +0.1, data is stationary
# 
# 
# 
# 
# #apply decomposition to all training data, total sales
#   #unlike the manual approach, this gets the residuals for time index i with a model trained from all previous time indexes
#   #this generates missing values for a few obs at the start, as a model couldn't be fit for them. drop these from the training set
# res_train = model_fft.residuals(ts_train["sales"])
# 
# 
# #time plot, acf, distribution
# plot_residuals_analysis(res_train)
# plt.show()
# plt.close()
# 
# 
# #pacf
# plot_pacf(res_train, max_lag=24)
# plt.show()
# plt.close()
# 
# 
# #kpss test for stationarity
# kpss(res_train)
# #not stationary





#split res_train as res_train, res_valid





  #apply linear regression with past_covariates on fft residuals
    #crossvalidate top level, 
  #reconcile top-down
    #crossvalidate top level and bottom level,
  #reversing transformations:
    #sales: inverse minmax - exponential - inverse CPI
    #oil: inverse minmax - inverse difference - inverse CPI












#NOTES FOR SECOND MODELING ATTEMPT:
  #CPI adjust sales and oil at the start
  #difference oil, onpromotion, transactions at the start
  #log transform sales as part of pipeline, after ts conversion
  #utilize the darts statistics and plots to evaluate time series characteristics such a seasonality
  #decompose sales with a STL, FFT or another advanced method, ideally non-linear trend
  #reevaluate lag features and rolling features with ewm or another advanced method, utilise darts plots and stats
  #fit an advanced suitable method to stationary data, in a hybrid model

















































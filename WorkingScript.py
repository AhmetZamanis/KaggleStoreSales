#KAGGLE STORE SALES FORECASTING COMPETITION

import pandas as pd
import numpy as np


import seaborn as sns
import matplotlib.pyplot as plt

from sktime.utils.plotting import plot_series
from sktime.utils.plotting import plot_lags
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.deterministic import DeterministicProcess
from scipy.signal import periodogram

from sklearn.preprocessing import StandardScaler

from sktime.transformations.series.detrend import STLTransformer
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.seasonal import DecomposeResult


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


#rename holiday_types column to avoid clashes at merge
df_holidays.rename(columns={"type":"holiday_type"}, inplace=True)


#inspect unique date and duplicate date holidays
df_holidays_unique = df_holidays.drop_duplicates(subset="date", keep=False)
df_holidays_duplicate = df_holidays[df_holidays.duplicated(["date"], keep=False)]



#add calendar_holiday and actual_holiday columns
  #calendar_holiday=1 if: Holiday, Additional, Work Day
  #actual_holiday=1 if: (Holiday & Transferred=FALSE), Transfer, Additional, Bridge
df_holidays["calendar_holiday"] = df_holidays.holiday_type.isin(["Holiday", "Additional", "Work Day"])
df_holidays["actual_holiday"] = (
  df_holidays.holiday_type.isin(["Transfer", "Additional", "Bridge"]) | 
((df_holidays.holiday_type=="Holiday") & (df_holidays.transferred==False))
)


#split special events
df_events = df_holidays[df_holidays.holiday_type=="Event"]
df_holidays = df_holidays.drop(labels=(df_events.index), axis=0)


#split holidays into local, regional, national
df_local = df_holidays.loc[df_holidays.locale=="Local"]
df_local.rename(
  columns={
    "calendar_holiday":"local_calendar_holiday",
    "actual_holiday":"local_actual_holiday"}, inplace=True)

df_regional = df_holidays.loc[df_holidays.locale=="Regional"]
df_regional.rename(
  columns={
    "calendar_holiday":"regional_calendar_holiday",
    "actual_holiday":"regional_actual_holiday"}, inplace=True)


df_national = df_holidays.loc[df_holidays.locale=="National"]
df_national.rename(
  columns={
    "calendar_holiday":"national_calendar_holiday",
    "actual_holiday":"national_actual_holiday"}, inplace=True)



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
df_events.drop("calendar_holiday", axis=1, inplace=True)
df_events.rename(columns={"actual_holiday":"event"}, inplace=True)
df_events["event"] = ~df_events["event"]


df_events_duplicated = df_events[df_events.duplicated(["date"], keep=False)]
#2 duplicates, one is earthquake, other is mother's day. drop earthquake
df_events.loc[244]
df_events.drop(244, axis=0, inplace=True)
#no more date duplicates in events
del df_events_duplicated




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
df_oil.rename(columns={"dcoilwtico":"oil"}, inplace=True)
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
    "holiday_type", "locale", "description", "transferred", 
    "regional_actual_holiday"], axis=1
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




#flag first day of year with binary feature, set national holidays columns to false at this date
df_train["new_years_day"] = (df_train.index.dayofyear==1)
df_test["new_years_day"] = (df_test.index.dayofyear==1)
df_train.national_calendar_holiday.loc[df_train.new_years_day==True] = False
df_train.national_actual_holiday.loc[df_train.new_years_day==True] = False  


#flag christmas, 21-26, set national holidays columns to false at christmas
df_train["christmas"] = (df_train.index.month==12) & (df_train.index.day.isin(
  [21,22,23,24,25,26])) 
df_test["christmas"] = (df_test.index.month==12) & (df_test.index.day.isin(
  [21,22,23,24,25,26]))
df_train.national_calendar_holiday.loc[df_train.christmas==True] = False
df_train.national_actual_holiday.loc[df_train.christmas==True] = False  


#flag paydays: 15th and last day of each month
df_train["payday"] = ((df_train.index.day==15) | (df_train.index.to_timestamp().is_month_end))
df_test["payday"] = ((df_test.index.day==15) | (df_test.index.to_timestamp().is_month_end))


#flag earthquakes: 2016-04-16 to 2016-05-16. set event to False for earthquake dates
earthquake_dates = pd.period_range(start="2016-04-16", end="2016-05-16")
df_train["earthquake"] = (df_train.index.isin(earthquake_dates))
df_train.loc[df_train.earthquake==True]
df_test["earthquake"] = False
df_train.event.loc[df_train.earthquake==True] = False



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
       'store_type', 'store_cluster', 'oil', 'local_calendar_holiday',
       'local_actual_holiday', 'regional_calendar_holiday',
       'national_calendar_holiday', 'national_actual_holiday', 'event',
       'new_years_day', 'payday', 'earthquake', "christmas"]]


df_train = df_train[['id', "sales", 'category', 'onpromotion', 'city', 'state', 'store_no',
       'store_type', 'store_cluster', 'oil', 'local_calendar_holiday',
       'local_actual_holiday', 'regional_calendar_holiday',
       'national_calendar_holiday', 'national_actual_holiday', 'event',
       'new_years_day', 'payday', 'earthquake', "christmas"]]
       



#handle missing values (before lags-indicators)


#check NAs
pd.isnull(df_train).sum()
#928422 in oil
#close to row number in holiday and event cols

pd.isnull(df_test).sum()
#7128 in oil
#close to row number in holiday and event cols




#set NA holiday values to False
holiday_cols = ['local_calendar_holiday', 'local_actual_holiday',
       'regional_calendar_holiday', 'national_calendar_holiday',
       'national_actual_holiday', 'event']

df_test[holiday_cols] = df_test[holiday_cols].fillna(value=False, inplace=False)
#worked for test data

df_train[holiday_cols] = df_train[holiday_cols].fillna(value=False)
#worked for train data




#fill in oil NAs with time interpolation
df_train["oil"] = df_train["oil"].interpolate("time")
df_test["oil"] = df_test["oil"].interpolate("time")
#got rid of all test NAs. 1782 remain for train


#check remaining train oil NAs 
df_train[pd.isnull(df_train["oil"])]
#all belong to the first day. fill them with the next day oil price


#fill first day oil NAs
df_train.oil = df_train.oil.fillna(method="bfill")


#check if it worked
pd.isnull(df_train).sum()
pd.isnull(df_test).sum()
#all done


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
       
       




#final checks


#shapes
df_train.shape
#(3000888, 20)
df_test.shape
#(28512, 19)


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
    
  #alternatives: 
    #try to group some categories together?
    df_test.category.unique()
    #try to apply a clustering based on location + store type?




#TIME COMPONENTS OF AGGREGATE SALES SERIES


sales_agg = df_train.groupby("date").sales.mean()
sales_agg = pd.DataFrame(data={
  "sales": sales_agg.values,
  "time": np.arange((len(sales_agg)))
}, index=sales_agg.index)
sales_agg.time = sales_agg.time + 1 

sns.set_theme(style="darkgrid")
#average daily sales across time, across all categories
ax = sns.lineplot(x="time", y="sales", legend=False, data=sales_agg)
ax.set_title("agg sales over time")
plt.show()
plt.close()
#generally increasing sales across time
#dips at time 1, 365, 1458, 729, 1093
sales_agg.index[sales_agg.time.isin([1,365,729,1093,1458])]
#these are all jan 1, already flagged with a binary feature


#avg sales each month of the year, across all categories
ax = sns.lineplot(x=sales_agg.index.month, y="sales", hue=sales_agg.index.year, data=sales_agg, legend="brief")
ax.set_title("month of year")
plt.show()
plt.close()
#sales peak at december
#they are generally stable before december, except for 2015 which had a sharp increase in june
#sales fluctuated considerably month to month in 2014



#avg sales each week of the year, across all categories
sns.lineplot(x=sales_agg.index.week, y="sales", hue=sales_agg.index.year, data=sales_agg)
ax.set_title("week of year")
plt.show()
plt.close()
#similar weekly seasonality to month. drop in 2017 after week 32-33 due to data ending
sales_agg.sales.loc[sales_agg.index.week==32]


#avg sales each day of the year, across all categories
sns.lineplot(x=sales_agg.index.dayofyear, y="sales", hue=sales_agg.index.year, data=sales_agg)
ax.set_title("day of year")
plt.show()
#dip at new years day, stable fluctuation across year, peak towards end of year
plt.close()





#avg sales each day of the month, across all categories
sns.lineplot(x=sales_agg.index.day, y="sales", hue=sales_agg.index.month, data=sales_agg)
ax.set_title("day of month")
plt.show()
plt.close()
#peak sales at start of month, days 1-3
#then drops until the 15th, slight recovery after 15th
#then drups until roughly 26-27, starts to recover towards the end of month
#different for december: the biggest peak happens at days 21-24 - christmas.
sales_agg.loc[(sales_agg.index.month==12) & (sales_agg.index.day.isin([20,21,22,23,24,30])), :]


#avg sales each day of the week, across all categories
sns.lineplot(x=sales_agg.index.dayofweek, y="sales", hue=sales_agg.index.month, data=sales_agg)
ax.set_title("day of week")
plt.show()
#dips until wednesday, ramps up afterwards and peaks at sunday
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
    #months of year (categorical) or weeks of year (fourier). or both? days of year too unstable.
    #days of month (fourier)
    #days of week (categorical)
    #plot a periodogram and decide?


#TIME DECOMPOSITION, AGGREGATE SALES

#detrend and deseasonalize
#use multiplicative decomposition (or log transform sales and use additive)
y_train_agg =  np.log(df_train.groupby("date").sales.mean() + 0.00001)



#STL decomposition
decomp_stl = STLTransformer(seasonal=31, robust=True, return_components=True)
y_train_decomped = decomp_stl.fit_transform(y_train_agg)
y_train_decomped["time"] = np.arange(len(y_train_decomped)) + 1
y_train_decomped.iloc[:,0:4] = np.exp(y_train_decomped.iloc[:,0:4])


#plot components


ax = sns.lineplot(x="time", y="trend", data=y_train_decomped)
plt.show()
plt.close()
#trend is still super seasonal. doesn't account for days of week, the end of year peaks, or new year drops

ax = sns.lineplot(x="time", y="seasonal", data=y_train_decomped)
plt.show()
plt.close()
#seasonal component fairly balanced over the years, though some peaks

ax = sns.lineplot(x="time", y="resid", data=y_train_decomped)
plt.show()
plt.close()
#the residual is subject to new year drops and short-term seasonality




#decompose with MSTL
  #update statsmodels module












#lag features (RE-EVALUATE AFTER DECOMPOSING)


#sales

#scatter
fig, ax = plot_lags(sales_agg.sales, lags=[1,2,3,4,5,6])
plt.show()
plt.close()


#pacf
plot_pacf(sales_agg.sales, lags=45)
plt.show()
#lags up to 9 all significant
#most significant ones:
  #1, 3, 5, 6, 7, 8, 9, 13, 14
plt.close()




#onpromotion
def make_lags(ts, lags, prefix):
  return pd.concat(
  {
    f"{prefix}_lag_{i}": ts.shift(i)
    for i in range(1, lags+1)
  },
  axis=1
  )

sales_agg_prom = df_train.groupby(df_train.index.day).sales.mean()
sales_agg_prom = pd.DataFrame(data={
  "sales": sales_agg_prom.values,
  "time": np.arange((len(sales_agg_prom)))
}, index=sales_agg_prom.index)
sales_agg_prom.time = sales_agg_prom.time + 1 
sales_agg_prom["onpromotion"] = df_train.groupby(df_train.index.day).onpromotion.mean()
lags_onpromotion = make_lags(sales_agg_prom.onpromotion, lags=5, prefix="prom")
sales_agg_prom = sales_agg_prom.merge(lags_onpromotion, how="left", on="date")
scaler_std = StandardScaler()


#lineplot of scaled agg sales and onpromotion
scaled_sales_prom = scaler_std.fit_transform(sales_agg_prom.values)
scaled_sales_prom = pd.DataFrame(scaled_sales_prom, columns=sales_agg_prom.columns)
scaled_sales_prom["time"] = sales_agg_prom.time.values


#plots
ax = sns.lineplot(data=scaled_sales_prom, x="time", y="sales", label="sales")
sns.lineplot(ax=ax, data=sales_agg_prom, x="time", y="onpromotion", label="lag 0")
sns.lineplot(ax=ax, data=sales_agg_prom, x="time", y="prom_lag_1", label="lag 1")
sns.lineplot(ax=ax, data=sales_agg_prom, x="time", y="prom_lag_2", label="lag 2")
sns.lineplot(ax=ax, data=sales_agg_prom, x="time", y="prom_lag_3", label="lag 3")
sns.lineplot(ax=ax, data=sales_agg_prom, x="time", y="prom_lag_4", label="lag 4")
sns.lineplot(ax=ax, data=sales_agg_prom, x="time", y="prom_lag_5", label="lag 5")
plt.show()
plt.close()



#scatter

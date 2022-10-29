#KAGGLE STORE SALES FORECASTING COMPETITION

import pandas as pd
import numpy as np


import seaborn as sns
import matplotlib.pyplot as plt

import sktime
from sktime.utils.plotting import plot_series
from sktime.utils.plotting import plot_lags
from sktime.transformations.series.difference import Differencer
from sktime.transformations.hierarchical.aggregate import Aggregator
# from sktime.forecasting.arima import AutoARIMA
# from sktime.forecasting.arima import ARIMA


from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.deterministic import DeterministicProcess
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import ccf

from scipy.signal import periodogram
from scipy.stats import pearsonr
from scipy.stats import spearmanr

from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression


import darts
# from darts.timeseries import TimeSeries
# from darts.models.forecasting.arima import ARIMA

# from sktime.transformations.series.detrend import STLTransformer
# from statsmodels.tsa.seasonal import MSTL
# from statsmodels.tsa.seasonal import DecomposeResult





np.set_printoptions(suppress=True, precision=8)
pd.options.display.float_format = '{:.8f}'.format


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


#merge transactions data to df_train
df_trans.set_index(pd.PeriodIndex(df_trans.date, freq="D"), inplace=True)
df_trans.drop("date", axis=1, inplace=True)      
df_trans.rename(columns={"store_nbr":"store_no"}, inplace=True)

df_train2 = df_train.merge(df_trans, how="left", on=["date", "store_no"])
#no rows added
#no wrong columns added
df_train2 = df_train2[['id', "sales", "transactions", 'category', 'onpromotion', 'city', 'state', 'store_no',
       'store_type', 'store_cluster', 'oil', 'local_calendar_holiday',
       'local_actual_holiday', 'regional_calendar_holiday',
       'national_calendar_holiday', 'national_actual_holiday', 'event',
       'new_years_day', 'payday', 'earthquake', "christmas"]]

pd.isnull(df_train2.transactions).sum()
#245784 NAs 

df_train2.transactions.fillna(0, inplace=True)
#none left

df_train = df_train2



#final checks


#shapes
df_train.shape
#(3000888, 21)
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




#FIT LAGGED PREDICTORS MODEL ON DECOMPOSED RESIDUALS

#outcome series: res_decomp

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


#make lags
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
  #get rid of non-features (id, city, state, store_type, store_cluster)
  #convert all booleans to integers
  #then convert df_train and df_test to wide (see bottom of notes). handle NAs 
  #split the data into:
    #target: darts time series with one row per date, one column per:
      #total sales that date
      #each category's total sales that date (will add up to Total)
      #each store's total sales that date (will add up to Total)
      #each category-store combo's sales that date (will add up to Total)
      #more hierarchy elements? will they improve performance in reconciliation?
        #store-cluster-type-total?
        #city-state-total?
      #sales lags 1 2 3 5 11 28 will be specified in RegressionModel (remember to handle NAs)
    #past_covariates: darts time series with one row per date, one column per:
      #trend dummy order 1
      #fourier features, period 365, order 4
      #oil moving average 21
    #static covariates: add to target time series when creating
      #calendar dummies
      #days of week dummies
    #how to handle the data:
      #create target time series with static covariates from long data using:
        #TimeSeries.from_group_dataframe() https://unit8co.github.io/darts/examples/15-static-covariates.html
      #create past covariates separately as another time series
      #add hierarchy dictionary to the target time series https://unit8co.github.io/darts/examples/16-hierarchical-reconciliation.html?highlight=hierarchical
    
    
    
    
#load back modified data
df_train = pd.read_csv("./ModifiedData/train_modified.csv", encoding="utf-8")
df_test = pd.read_csv("./ModifiedData/test_modified.csv", encoding="utf-8")
















# #convert dates to indexes again
# df_train.set_index(pd.PeriodIndex(df_train.date, freq="D"), inplace=True)
# df_train.drop("date", axis=1, inplace=True)
# 
# df_test.set_index(pd.PeriodIndex(df_test.date, freq="D"), inplace=True)
# df_test.drop("date", axis=1, inplace=True)


# #combine category and store_no cols
# df_test["category"] = df_test["category"].astype(str) + "_" + df_test["store_no"].astype(str)
# df_train["category"] = df_train["category"].astype(str) + "_" + df_train["store_no"].astype(str)
# df_test["category"].nunique()
# df_train["category"].nunique()
# #1782 unique category-store combos


# #set date and category to multiindex
# test_multiindex = df_test[["category", "date"]]
# test_multiindex.date = pd.PeriodIndex(test_multiindex.date, freq="D")
# test_multiindex = pd.MultiIndex.from_frame(test_multiindex)
# df_test.index = test_multiindex
# 
# 
# #transform to hierarchical
# agg = Aggregator()
# df_test = agg.fit_transform(df_test)

#KAGGLE STORE SALES TIME SERIES FORECASTING - SECOND ATTEMPT

#LIBRARIES

#data handling
library(tidyverse)
library(tsibble)
library(lubridate)

#TS modeling
library(feasts)
library(fable)



#DATA PREP


#load previously modified data
df_train = read.csv("./ModifiedData/train_modified.csv", encoding="UTF-8", header=TRUE)
df_test = read.csv("./ModifiedData/test_modified.csv", encoding="UTF-8", header=TRUE)


#summarize raw data
names(df_train)
dim(df_train)
#3000888 rows, 20 cols

names(df_test)
dim(df_test)
#28512 rows, 18 cols


#combine train and test data for data handling operations
df_test$sales = NA
df_test$transactions = NA
df = rbind(df_train, df_test)
names(df)
dim(df)
#3029400 rows, 20 columns


#coerce df to tibble
df$date = as_date(df$date)
ts = as_tsibble(df, key=c("category", "city", "state", "store_no", "store_type",
                         "store_cluster"), index="date")


#check missing values
colSums(is.na(ts))
#none except the 28512 sales and transactions for test data


#check missing dates
missing_dates = scan_gaps(ts)$date
unique(missing_dates)
#december 25 missing in 2013-2016


#add rows for missing dates
ts = fill_gaps(ts, .full=FALSE)
dim(ts)
#we now have 3036528 rows
scan_gaps(ts)$date
#no more missing dates


#replace NA values for missing dates
  #sales, transactions, onpromotion, oil: spline interpolate
  #calendar features: all false, except christmas true
ts_missing = ts %>% filter_index("2013-12-25", "2014-12-25", "2015-12-25", "2016-12-25")
ts_missing[,13:19] = "False"
ts_missing$christmas = "True"
ts 


#CPI adjust sales and oil
cpi = c(112.8,
        116.8,
        121.5,
        123.6,
        123.6)
names(cpi) = c("2013", "2014", "2015", "2016", "2017")

for (years in names(cpi)){
  ts$sales[ts$date]
}  




#PREPROCESSING - DECOMPOSITION


#coerce df_train and df_test to tsibble
  #index=date
  #key=category, store_no, store_cluster, store_type, city, state
#log transformation of sales
#scaling of sales





#PREPROCESSING - STATIONARY MODEL


#impute NAs in lags and rolling features
#differencing of features (and decomposed sales?)
#scaling of features


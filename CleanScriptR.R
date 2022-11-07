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


#CPI adjust sales and oil
cpi = c(112.8,
        116.8,
        121.5,
        123.6,
        123.6)
names(cpi) = c("2013", "2014", "2015", "2016", "2017")

ts = ts %>% mutate(sales = ifelse(year(date)=="2013", sales / cpi["2013"] * 100, 
                             ifelse(year(date)=="2014", sales / cpi["2014"] * 100,
                                    ifelse(year(date)=="2015", sales / cpi["2015"] * 100,
                                           sales / cpi["2016"] * 100))))

# max(na.omit(df$sales))
# max(na.omit(ts$sales))
# max(na.omit(df$oil))
# max(na.omit(ts$oil))




#evaluate features to add
  #pre-post holiday, new years, christmas etc. features

#plot aggregated sales, grouped by holiday vs no holiday
ts %>%
  group_by(christmas) %>%
  summarise(agg_sales = sum(sales)) %>%
  autoplot()








#replace NA values for missing dates in train data, after splitting train-test: 
  #msales: spline interpolate.
  #transactions, onpromotion, oil: spline interpolate
  #calendar features: christmas true, all others false


#drop sales from test data









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


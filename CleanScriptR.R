#KAGGLE STORE SALES TIME SERIES FORECASTING - SECOND ATTEMPT

#LIBRARIES####

#data handling
library(tidyverse)
library(tsibble) #time series tibbles
library(lubridate) #date objects

#visualization
library(patchwork)

#TS modeling
library(feasts)
library(fable)


options(scipen=999)


#DATA PREP 1####


#load previously modified data
df_train = read.csv("./ModifiedData/train_modified.csv", encoding="UTF-8", header=TRUE)
df_test = read.csv("./ModifiedData/test_modified.csv", encoding="UTF-8", header=TRUE)


# #summarize raw data
# names(df_train)
# dim(df_train)
# #3000888 rows, 20 cols
# 
# names(df_test)
# dim(df_test)
# #28512 rows, 18 cols


#combine train and test data for data handling operations
df_test$sales = NA
df_test$transactions = NA
df = rbind(df_train, df_test)
# names(df)
# dim(df)
#3029400 rows, 20 columns


#coerce df to tibble
df$date = as_date(df$date)
ts = as_tsibble(df, key=c("category", "city", "state", "store_no", "store_type",
                         "store_cluster"), index="date")


# #check missing values
# colSums(is.na(ts))
# #none except the 28512 sales and transactions for test data


# #check missing dates
# missing_dates = scan_gaps(ts)$date
# unique(missing_dates)
# #december 25 missing in 2013-2016


#add rows for missing dates
ts = fill_gaps(ts, .full=FALSE)


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


#drop id column
ts = dplyr::select(ts, -c("id"))


#split ts_train and ts_test
ts_train = ts %>%
  filter_index(~ "2017-08-15")

ts_test = ts %>%
  filter_index("2017-08-16"~.)


#drop sales, transactions from ts test
ts_test = dplyr::select(ts_test, -c("sales", "transactions"))


# #check NAs
# colSums(is.na(ts_train))
# #7128 each in sales, transactions, onpromotion, oil, calendar columns
# 
# colSums(is.na(ts_test))
# #none



#NA IMPUTATION 1####
#replace NA values for missing dates in train data: 

#calendar features: christmas true, all others false
ts_train$christmas = replace_na(ts_train$christmas, "True")
calendar_cols = names(ts_train)[12:18]
ts_train = ts_train %>% 
  mutate_at(calendar_cols, ~ replace_na(.,"False"))


#transactions, onpromotion, oil, sales: interpolate with linear

#transactions
imp_trns = ts_train %>%
  model(lm = TSLM(transactions ~ trend())) %>%
  interpolate(ts_train)
ts_train$transactions = imp_trns$transactions
rm(imp_trns)


#onpromotion
imp_onp = ts_train %>%
  model(lm = TSLM(onpromotion ~ trend())) %>%
  interpolate(ts_train)
ts_train$onpromotion = imp_onp$onpromotion
rm(imp_onp)


#oil
imp_oil = ts_train %>%
  model(lm = TSLM(oil ~ trend())) %>%
  interpolate(ts_train)
ts_train$oil = imp_oil$oil
rm(imp_oil)


#sales
imp_sales = ts_train %>%
  model(lm = TSLM(sales ~ trend())) %>%
  interpolate(ts_train)
ts_train$sales = imp_sales$sales
rm(imp_sales)


#ALTERNATIVE: CONVERT TO ts, USE imputeTS TO IMPUTE WITH ADVANCED METHODS AND PLOT?
  #too much work for 4 missing periods?


colSums(is.na(ts_train))
#yes, that should work




#EDA 1#### 
#TREND & SEASONALITY, CALENDAR EFFECTS


##Time plots####

#overall trend
ts_train %>%
  summarise(agg_sales=sum(sales)) %>%
  autoplot() +
  labs(title="overall trend",
       y="total sales",
       x="year")


#seasonality plots:


#annual seasonality
ts_train %>% 
  #summarise(agg_sales=sum(sales)) %>%
  gg_season() +
  labs(title="annual seasonality",
       y="aggregated sales",
       color="year") +
  theme_bw()

#monthly seasonality
ts_train %>% 
  summarise(agg_sales=sum(sales)) %>%
  gg_season(period="week") +
  labs(title="annual seasonality",
       y="aggregated sales",
       color="year") +
  theme_bw()

ts_train %>% 
  summarise(agg_sales=sum(sales)) %>%
  gg_subseries(period="year")





#seasonality plots, aggregated:


#annual seasonality

#   #across months of year
# ts_train %>% summarise(agg_sales=sum(sales)) %>%
#   fill_gaps(.full=FALSE) %>%
#   gg_season(period="year") +
#   labs(title="annual seasonality across months")


#across months of year
ts_train %>%
  summarise(agg_sales=sum(sales)) %>%
  mutate(month=month(date)) %>%
  mutate(year=year(date)) %>%
  as_tibble() %>%
  select(-date) %>%
  group_by(month, year) %>% summarise(monthly_sales=sum(agg_sales)) %>%
  mutate(year=as.factor(year)) %>%
  ggplot(aes(x=month, y=monthly_sales, color=year)) +
  geom_line(size=1) +
  geom_point(size=2) +
  scale_x_continuous(breaks=seq(1,12,1)) +
  labs(title="annual seasonality across months") +
  theme_bw()

#across weeks of year
ts_train %>%
  summarise(agg_sales=sum(sales)) %>%
  mutate(week=week(date)) %>%
  mutate(year=year(date)) %>%
  as_tibble() %>%
  select(-date) %>%
  group_by(week, year) %>% summarise(weekly_sales=sum(agg_sales)) %>%
  mutate(year=as.factor(year)) %>%
  ggplot(aes(x=week, y=weekly_sales, color=year)) + 
  geom_line(size=1) +
  geom_point(size=2) +
  scale_x_continuous(breaks=seq(1,52,2)) +
  labs(title="annual seasonality across weeks") +
  theme_bw()



#monthly seasonality
  #across days

#weekly seasonality
  #across days




#evaluate features to add
  #pre-post holiday, new years, christmas etc. features

#plot aggregated sales, grouped by holiday vs no holiday
ts_train %>%
  summarise(agg_sales = sum(sales)) %>%
  group_by(national_holiday) %>%
  gg_season(agg_sales, period="day")


















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


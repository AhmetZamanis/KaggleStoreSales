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


#load holidays data
df_holidays = read.csv("./OriginalData/holidays_events.csv", encoding="UTF-8", header=TRUE)



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


colSums(is.na(ts_train))
#yes, that should work




#EDA 1 - TIME & CALENDAR EFFECTS#### 


##Time plots####


###Total sales####

#overall trend
ts_train %>%
  summarise(agg_sales=sum(sales)) %>%
  autoplot() +
  labs(title="overall trend",
       y="total sales",
       x="year")
#sales generally peak at the end of each year, drop at the start of each year
#overall increasing trend, will be non linear if not fit smoothly


#annual seasonality
ts_train %>% 
  summarise(agg_sales=sum(sales)) %>%
  gg_season(period="year") +
  labs(title="annual seasonality",
       y="total sales",
       color="year") +
  theme_bw()
#too noisy to read


#annual seasonality, months aggregated
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
  labs(title="annual seasonality, months aggregated",
       y="total sales",
       color="year") +
  theme_bw()
#sales peak in december
#drop from january to february
#increase from feb to march


#annual seasonality, weeks aggregated
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
  scale_x_continuous(breaks=seq(1,52,1)) +
  labs(title="annual seasonality, weeks aggregated",
       y="total sales",
       color="year") +
  theme_bw()
#holiday rise starts from wk 47, peaks in wk 51, crashes in wk 1-2
#sales increase over years. strong annual pattern that holds across years, except:
  #2014: strong drop between weeks 6-8, 14-26, 31-35 (why?)
    #or: is it a strong spike between wk 1-4, 9-13, 27?
  #2015: sales dropped lower than 2014 until week 21, 
    #suddenly spiked in week 22, stayed elevated after that. (why?)
  #2016: strong peak between weeks 14-18 (earthquake)
    #earthquake previously flagged from apr 16 to may 16 (weeks 15, 16, 17, 18)




#monthly seasonality, days aggregated
ts_train %>%
  summarise(agg_sales=sum(sales)) %>%
  mutate(month=month(date)) %>%
  mutate(day=day(date)) %>%
  as_tibble() %>%
  select(-date) %>%
  group_by(month, day) %>% summarise(daily_sales=sum(agg_sales)) %>%
  mutate(month=as.factor(month)) %>%
  ggplot(aes(x=day, y=daily_sales, color=month)) + 
  geom_line(size=1) +
  geom_point(size=2) +
  scale_x_continuous(breaks=seq(1,31,1)) +
  labs(title="monthly seasonality, days aggregated",
       y="total sales",
       color="month") +
  theme_bw()
#sales generally decline from day 1 to 15, increase in day 16 (payday +1 effect)
  #flag day 16 for payday instead of 15
#stable from 16-29, another increase in day 30 (payday)
  #flag day 31 as payday +1
  #highest sales in days 1-6, flag as post_payday
#unusually high sales in days 20-24 in december, flag as christmas eve
  #drop in day 25 december, flag as christmas day
#very sharp drop at day 1 jan, marked as new years day
  #stronger than usual recovery in day 2 jan, mark as new years +1
#very sharp drop on feb 29, this is misleading because there are fewer february 29s



#weekly seasonality
ts_train %>% 
  # mutate(month=month(date), year=year(date)) %>%
  # group_by(month, year) %>%
  summarise(agg_sales=sum(sales)) %>%
  gg_season(period="week") +
  labs(title="weekly seasonality",
       y="total sales",
       color="year") +
  theme_bw()
#non-linear decline into thursday
#non linear increase into sunday


#sales lag plots
ts_train %>%
  summarise(agg_sales=sum(sales)) %>%
  gg_lag(geom="point")
#lag 7 and lag 1 are particularly strong
  #this shows strong weekly seasonality


#ACF and PACF for sales
ts_train %>%
  summarise(agg_sales=sum(sales)) %>%
  ACF(agg_sales, lag_max = 60) %>%
  autoplot() +
  labs(title="ACF total sales")
#strong at lag 1, declines until lag 4, increases again and peaks at lag 7,
  #pattern repeats weekly, declining increasingly
  #the decline is due to trend. the pattern is due to weekly seasonality


ts_train %>%
  summarise(agg_sales=sum(sales)) %>%
  PACF(agg_sales) %>%
  autoplot() +
  labs(title="PACF total sales")
#lag 1 strongest.
  #other strong lags: 3, 5, 6, 7, 8, 9, 13, 14
  #other numerous slightly significant lags
  #declining weekly pattern



#examine the sudden changes in 2014-2015. is it due to oil perchance?
ts_train %>%
  summarise(agg_sales=sum(sales), oil=mean(oil)) %>%
  mutate(week=week(date)) %>%
  mutate(year=year(date)) %>%
  as_tibble() %>%
  select(-date) %>%
  group_by(week, year) %>% summarise(weekly_sales=sum(agg_sales), oil=mean(oil)) %>%
  mutate(weekly_sales = scale(weekly_sales), oil=scale(oil)) %>%
  filter(year %in% c(2014, 2015)) %>%
  mutate(week=ifelse(year==2015, week + 52, week)) %>%
  ggplot(aes(x=week, y=weekly_sales, color=as.factor(year))) + 
  geom_line(size=1) +
  geom_point(size=2) +
  geom_line(aes(x=week, y=oil), size=1, linetype=2) +
  geom_point(aes(x=week, y=oil), size=1, shape=5) +
  scale_x_continuous(breaks=seq(1,104,2)) +
  labs(title="sales vs oil 2014-2015, weeks aggregated, values scaled",
       y="total sales, oil") +
  theme_bw()


#oil vs overall trend
ts_train %>%
  summarise(agg_sales=sum(sales), oil=mean(oil)) %>%
  autoplot(agg_sales) +
  ts_train %>%
  summarise(agg_sales=sum(sales), oil=mean(oil)) %>%
  autoplot(oil)
#oil price was around 90-110 until mid 2014
#after that, it sharply declined to 30-60, and stayed there
#THEORY: elevated-dropped oil prices for a 3-6 month period leads to elevated-dropped sales 4-6 months ahead
  #oil spike: 2013 M6-9. sales spike: 2014 M1-3


#scatterplot of oil vs sales, without time
o1 <- ts_train %>%
  summarise(agg_sales=sum(sales), oil=mean(oil)) %>%
  ggplot(aes(x=oil, y=agg_sales)) + geom_point() + geom_smooth() +
  labs(x="oil at T")
#two distinct groups, 30-60 oil and 90-110 oil. few obs between
#considerable decline in sales with high CURRENT oil prices


#scatterplot of oil lagged vs sales, without time
o2 <- ts_train %>%
  summarise(agg_sales=sum(sales), oil=mean(oil)) %>%
  mutate(oil=dplyr::lag(oil, n=60)) %>%
  ggplot(aes(x=oil, y=agg_sales)) + geom_point() + geom_smooth() +
  labs(x="oil at T-60")
  

#scatterplot of oil lagged vs sales, without time
o3 <- ts_train %>%
  summarise(agg_sales=sum(sales), oil=mean(oil)) %>%
  mutate(oil=dplyr::lag(oil, n=120)) %>%
  ggplot(aes(x=oil, y=agg_sales)) + geom_point() + geom_smooth() +
  labs(x="oil at T-120")


#scatterplot of oil lagged vs sales, without time
o4 <- ts_train %>%
  summarise(agg_sales=sum(sales), oil=mean(oil)) %>%
  mutate(oil=dplyr::lag(oil, n=180)) %>%
  ggplot(aes(x=oil, y=agg_sales)) + geom_point() + geom_smooth() +
  labs(x="oil at T-180")


#all oil scatterplots together
(o1 | o2) / (o3 | o4)
#the relationship generally holds at all T values
#however, it is most linear at T-60. it becomes non-linear/sigmoidal at 120 and 180
#re-evaluate this with rolling stats instead of actual values
  #evaluate after you remove trend & seasonality




#zoom in on the earthquake days. which days to flag?
ts_train %>%
  summarise(agg_sales=sum(sales)) %>%
  filter(year(date)==2016) %>%
  autoplot()

ts_train %>%
  summarise(agg_sales=sum(sales)) %>%
  filter(year(date)==2016 & month(date) %in% c(3,4,5,6)) %>%
  autoplot()
#the start will definitely be apr 16
  #earthquake: apr 16
  #+n features until and including may 15


 

#FEATURE ENGINEERING 1####


###Modify calendar features####

#drop old versions
ts_train = ts_train %>% select(-c("payday", "new_years_day", "christmas", "earthquake"))


#convert holiday features to dummies
ts_train$local_holiday = ts_train$local_holiday %>%
  recode("True" = 1, "False" = 0)

ts_train$regional_holiday = ts_train$regional_holiday %>%
  recode("True" = 1, "False" = 0)

ts_train$national_holiday = ts_train$national_holiday %>%
  recode("True" = 1, "False" = 0)



#payday features:
  #payday_16: day 16
  #payday_31: day 31
  #payday_n: day 1-6
ts_train = ts_train %>%
  mutate(payday_16 = ifelse(day(date)==16, 1, 0),
         payday_31 = ifelse(day(date)==31, 1, 0),
         payday_1 = ifelse(day(date)==1, 1, 0),
         payday_2 = ifelse(day(date)==2, 1, 0),
         payday_3 = ifelse(day(date)==3, 1, 0),
         payday_4 = ifelse(day(date)==4, 1, 0),
         payday_5 = ifelse(day(date)==5, 1, 0),
         payday_6 = ifelse(day(date)==6, 1, 0))


#christmas features:
  #christmas_eve: 20-24 december
  #christmas_day: 25 december
ts_train = ts_train %>%
  mutate(
    christmas_eve = ifelse(
      day(date) %in% c(20:24) &
        month(date) == 12,
      1, 0
    ),
    christmas_day = ifelse(
      day(date) == 25 &
        month(date) == 12,
      1, 0
    )
  )




#new year features:
  #new_year_1: january 1
  #new_year_2: january 2
ts_train = ts_train %>%
  mutate(
    new_year_1 = ifelse(
      day(date) == 1 &
        month(date) == 1,
      1, 0
    ),
    new_year_2 = ifelse(
      day(date) == 2 &
        month(date) == 1,
      1, 0
    )
  )



#earthquake features:
  #earthquake_1 = apr 16
  #earthquake_2,3,4,5,6,7 = apr 17, 18, 19, 20, 21
  #earthquake_16 = may 1

ts_train = ts_train %>%
  mutate(
    earthquake_1 = ifelse(year(date) == 2016 & month(date) == 4 & day(date) == 
                            16,
                          1, 0
                          ),
    earthquake_2 = ifelse(year(date) == 2016 & month(date) == 4 & day(date) == 
                            17,
                          1, 0
    ),
    earthquake_3 = ifelse(year(date) == 2016 & month(date) == 4 & day(date) ==
                            18,
                          1, 0
    ),
    earthquake_7 = ifelse(year(date) == 2016 & month(date) == 4 & day(date) %in% 
                            c(19:22),
                          1, 0
    ),
    earthquake_15 = ifelse(year(date) == 2016 & month(date) == 4 & day(date) %in% 
                             c(23:30),
                            1, 0
                           ),
    earthquake_30 = ifelse(year(date) == 2016 & month(date) == 5 & day(date) %in% 
                              c(1:16),
                            1, 0
    )
  )



#addition to holiday features:
  #x_holiday_n: n days before x holiday
ts_train = ts_train %>%
  mutate(local_holiday_lead1 = lead(local_holiday, 1),
         local_holiday_lead2 = lead(local_holiday, 2),
         local_holiday_lead3 = lead(local_holiday, 3),
         regional_holiday_lead1 = lead(regional_holiday, 1),
         regional_holiday_lead2 = lead(regional_holiday, 2),
         regional_holiday_lead3 = lead(regional_holiday, 3),
         national_holiday_lead1 = lead(national_holiday, 1),
         national_holiday_lead2 = lead(national_holiday, 2),
         national_holiday_lead3 = lead(national_holiday, 3), .after=national_holiday)
#yes, that should work


#events: split into different event types
  #dia_madre: events = TRUE & date in may 8, 10, 11, 12, 14
  #futbol: events = TRUE & date in 2014-06-12 / 2014-07-13
  #black_friday: event = TRUE & (date in 2014-11-28, 2015-11-27, 2016-11-25)
  #cyber_monday: event = TRUe & (date in 2014-12-01, 2015-11-30, 2016-11-28)
  #drop event afterwards
ts_train = ts_train %>%
  mutate(dia_madre = ifelse(
    event=="True" & month(date) == 5 & day(date) %in% c(8, 10, 11, 12, 14),
    1, 0
  ),
  futbol = ifelse(
    event=="True" & date %within% interval("2014-06-12", "2014-07-13"),
    1, 0
  ),
  black_friday = ifelse(
    event=="True" & as.character(date) %in% c("2014-11-28", "2015-11-27", "2016-11-25"),
    1, 0
  ),
  cyber_monday = ifelse(
    event=="True" & as.character(date) %in% c("2014-12-01", "2015-11-30", "2016-11-28"),
    1, 0
  ),
  .after=event)
#yes, that should work


ts_train = ts_train %>%
  select(-event)


#days of week dummies: use monday as intercept, because it has the least fluctuation-outliers
ts_train = ts_train %>%
  mutate(
    tuesday = ifelse(
      wday(date) == 2, 1, 0
    ),
    wednesday = ifelse(
      wday(date) == 3, 1, 0
    ),
    thursday = ifelse(
      wday(date) == 4, 1, 0
    ),
    friday = ifelse(
      wday(date) == 5, 1, 0
    ),
    saturday = ifelse(
      wday(date) == 6, 1, 0
    ),
    sunday = ifelse(
      wday(date) == 7, 1, 0
    )
  )


#check NAs
colSums(is.na(ts_train))
#we have a few NAs in the holiday leads columns
  #2017-08-13, 14, 15
#check if there are any holidays in 2017-08-16

df_holidays = read.csv("./OriginalData/holidays_events.csv", encoding="UTF-8", header=TRUE)
#no holidays in 2017-08-16. replace the na's with 0's
ts_train[is.na(ts_train)] = 0
#yes, that should work




#save new version of ts_train
write.csv(ts_train, "./ModifiedData/train_modified2.csv", row.names = FALSE)


##Reload new version of ts_train####
ts_train = read.csv("./ModifiedData/train_modified2.csv", encoding="UTF-8", header=TRUE)
ts_train$date = as_date(ts_train$date)
ts_train = as_tsibble(ts_train, key=c("category", "city", "state", "store_no", "store_type",
                          "store_cluster"), index="date")




#MODEL 1 - DECOMPOSITION####


##STL decomposition, total sales####

#NOTE: STL in feasts cannot make predictions. therefore it can't be part of the model.
  #use STL to examine the trend and seasonality, and then model them in the LM
  #use a piecewise linear trend with knots. derive the number of knots from STL


#default trend, monthly and weekly annual seasonality
stl_model = ts_train %>%
  summarise(agg_sales=sum(sales)) %>%
  model(STL(
    log(agg_sales) ~ trend() +
                     season(period=28) +
                     season(period=7)
    )
  )
#7-period seasonality only doesn't rid the trend from the seasonality
#28-period only, and 28-period together with 7-period have similar trends
  #but the residuals for the multiseason decomposition are smaller. go with that


#retrieve components
stl_components = stl_model %>%
  components() 
#actual - trend - season 7 - season 28 =  remainder


#plot all components
stl_model %>% components %>% autoplot()
#trend: 2-piece linear, after adjusting for holidays, new years, and the cylicality?
#seasonality: multiseason captures it better than single season.
  #two separate sets of fourier features, for period 28 and period 7?


#rename log(agg_sales) in the components tsibble
stl_components = stl_components %>%
  rename(agg_sales = "log(agg_sales)")


#plot trend vs actual
stl_components %>%
  ggplot(aes(x=date, y=agg_sales)) +
  geom_line() +
  geom_line(aes(x=date, y=trend), color="#F8766D", size=1) +
  labs(title="STL decomposed trend vs actual values",
       y="total sales") +
  theme_bw()
#trend follows the actual prices very well,
  #except for ny 2017 where it drops too early
#without the 28-period seasonality, the trend becomes way too wiggly


# #plot seasonal adjusted vs actual, 2017
# stl_components %>%
#   filter(year(date)==2016) %>%
#   ggplot(aes(x=date, y=agg_sales)) +
#   geom_line() +
#   geom_line(aes(x=date, y=season_adjust), color="#F8766D") +
#   labs(title="STL seasonally adjusted values vs actual values",
#        y="total sales") +
#   theme_bw()


#plot seasonal components vs actual values, only 2016 for ease of viewing
stl_components %>%
  filter(year(date)==2016) %>%
  ggplot(aes(x=date, y=scale(agg_sales))) +
  geom_line() +
  geom_line(aes(x=date, y=scale(season_7 + season_28)), color="#00BFC4") +
  labs(title="STL decomposed seasonality vs actual values",
       subtitle="seasonality = sum of period_7 and period_28 seasonality",
       y="total sales") +
  theme_bw()
#catches the seasonality waveshape well, but not the magnitudes
  #1 peak-trough pair per 1 week. 52 movements in total



#STL decomped innovation residuals analysis
stl_model %>%
  gg_tsresiduals()
#residuals are normally distributed around 0, except for new years outliers
#strong ACF correlation with lags 7 and its multiples
  #day of week seasonality is still present. will be accounted for with dummies
#slightly significant acf correlation with lags 1, 8, 9, 10, 11
  #sigmoidal decline. not significant past lag 11. is this the day of week seasonality?


#test stationarity

#kpss test
stl_components %>%
  features(remainder, unitroot_kpss)
#null hypothesis: data is stationary around a linear trend
  #the null is accepted with a p value of 0.1 or higher
  #the residuals are stationary with a possible trend


#philips-perron test
stl_components %>%
  features(remainder, unitroot_pp)
#null hypothesis: data is non-stationary around a constant trend
  #the null is rejected with a p value of 0.01 or lower
  #the residuals are stationary with no trend






##LM time & calendar model, total sales####


#trend: piecewise linear 
  #knots at:
    #mid 2015?
#seasonality: fourier features
  #one set for period 7, or two for period 7 and 28?
#calendar effects: dummies


#create tsibble with all dummy predictors and total sales
  #log transform to get multiplicative decomposition
cal_cols = names(ts_train)[12:51]
ts_lm = ts_train %>%
  summarise(agg_sales=sum(sales),
            across(cal_cols, mean)) %>%
  mutate(agg_sales=log(agg_sales))


#create validation set
ts_lm_val = ts_lm %>%
  filter(year(date)==2017)


#fit LM on 2013-2016
lm_model = ts_lm %>%
  filter(year(date)!=2017) %>%
  model(TSLM(agg_sales ~ . + 
               trend(knots=c(date("2015-06-01"))) +
               fourier(period=28, K=7) +
               fourier(period=7, K=2) -
               date - agg_sales
             )
        )




#predict LM on 2017
lm_preds = lm_model %>%
  forecast(ts_lm_val) 


#plot actual vs predicted, time plot
lm_preds %>%
  autoplot(ts_lm_val) +
  labs(title="predicted vs actual total sales",
       subtitle="model: lm with 2-piece trend, fourier pairs, calendar dummies",
       y="total sales") +
  theme_bw()
#looks good overall
  #catches the waveshape every time except for the last wave
    #might indicate some special effect in the last week of the training data,
      #may apply to the testing data
  #the magnitude of the wave is close to actual, 
    #matches the actual values exactly in some weeks,
    #misses them considerably in other weeks
    #the timing is slightly early or late for some waves
      #possibly due to cyclicality
  #matches the new years' drop, and the subsequent recovery very well


#plot actual vs predicted, scatterplot
ts_lm_val %>%
  ggplot(aes(x=agg_sales, y=lm_preds$.mean)) +
  geom_point() +
  geom_abline(slope=1, intercept=0, color="red") +
  labs(title="predicted vs actual total sales",
       subtitle="model: lm with 2-piece trend, fourier pairs, calendar dummies",
       y="total sales, predicted",
       x="total sales, actual") +
  theme_bw()



#score the predictions with rmse, mape and rmsle
  #reverse the log transform for mape
lm_scores = lm_preds %>% 
  mutate(agg_sales=exp(agg_sales)) %>%
  accuracy(ts_lm_val %>%
             mutate(agg_sales=exp(agg_sales)))

#get the RMSLE as well
lm_scores$RMSLE = lm_preds %>%
  accuracy(ts_lm_val) %>%
  select(RMSE)

#scores
lm_scores$RMSE
lm_scores$RMSLE
lm_scores$MAPE
#rmse 61338.21
#rmsle 0.0836
#mape 6.434653




#diagnose the model fit


#retrieve LM model report
lm_model %>% report()
#r square 85%
#large and significant coefs:
#regional_holiday_lead3,
#payday dummies:
#payday + 1, 2, 3, 4, 5 all have larger magnitude than both paydays
#sd's are low as well
#christmas eve,
#new year 1
#earthquake 2,3,7


#innovation residuals analysis
lm_model %>% gg_tsresiduals() +
  labs(title="innovation residuals diagnostics, lm model")
#generally seems statonary, except for:
  #start of 2014 - mid 2015 cyclicality is not captured
  #christmas peaks are not well captured, or they are backed by other cyclical components?
  #the increase in the end of the training set (2017 jul-aug) is not well captured
    #figure out the cause of this
#ACF lags persist all the way to lag 28, linearish decline
  #also a slight weekly sigmoidal pattern persists, but only between lag 1-7
  #strongest at lag 1 with 0.8, weakest at lag 28 with 0.2
#residual distribution centered normally around 0, except for few outliers


#test stationarity

#kpss test
augment(lm_model) %>%
  features(.innov, unitroot_kpss)
#null hypothesis: data is stationary around a linear trend
  #the null is accepted with a p value of 0.1 or higher
  #the residuals are stationary with a possible trend
    #although the test stat is 0.2, less than half of the STL decomp


#philips-perron test
augment(lm_model) %>%
  features(.innov, unitroot_pp)
#null hypothesis: data is non-stationary around a constant trend
  #the null is rejected with a p value of 0.01 or lower
  #the residuals are stationary with no trend


#plot residuals against fitted values
augment(lm_model) %>%
  ggplot(aes(x=.fitted, y=.resid)) +
  geom_point() +
  labs(title="fitted vs residuals plot, lm model",
    x="fitted", y="residuals")
#4 huge outliers (new years days)
  #besides the outliers, the residuals are larger around fitted 0,
    #and get smaller with lower and higher fitted values
    #maybe the model isn't great at predicting zero or small sales




#MODEL 1 RESULT:
  #it seems the trend and seasonality is captured well, in shape if not magnitude
  #predictions are good overall, 6.5% MAPE, 0.083 RMSLE, 61338.21 RMSE
  #preds sometimes miss the magnitude of waves, sometimes the timing
  #ACF autocorrelations remain, cyclic components such as christmas and 2014-2015 cycles remain




#rolling CV the LM model
ts_lm_cv = ts_lm %>%
  stretch_tsibble(.step=15, .init=365)

table(ts_lm_cv$.id)
#training sets start from first 365, progresses as n+15, ends with first 1685 rows
#won't be able to predict set id 89 with a window of 15
  #filter the data and keep only first 88 sets as training (first 1670 rows)
  #give the original data as new_data
    #create matching .id column if needed
ts_lm_cv = ts_lm_cv %>%
  filter(.id!=89)
#88 training sets, starting from first 365, ending with first 1670


#create validation data with matching .id col
  #drop 2013 rows because they won't be predicted
ts_lm_cv_val = ts_lm %>%
  filter(year(date)!=2013) %>%
  tile_tsibble(.size=15) %>%
  filter(.id!=89)

table(ts_lm_cv_val$.id)
#89 testing sets with window 15, starting from 2014


#perform cv
lm_cv = ts_lm_cv %>%
  model(TSLM(agg_sales ~ . +
               trend(knots=c(date("2015-06-01"))) +
               fourier(period=28, K=7) +
               fourier(period=7, K=2) -
               date - agg_sales
  )
  ) %>%
  forecast(new_data=ts_lm_cv_val)


#score the predictions with rmse, mape and rmsle
  #reverse the log transform for mape
lm_cv_scores = lm_cv %>% 
  mutate(agg_sales=exp(agg_sales)) %>%
  accuracy(ts_lm_cv_val %>%
             mutate(agg_sales=exp(agg_sales)))

#get the RMSLE as well
lm_cv_scores$RMSLE = lm_cv %>%
  accuracy(ts_lm_cv_val) %>%
  select(RMSE)

#scores
mean(lm_cv_scores$RMSE)
mean(lm_cv_scores$RMSLE$RMSE)
mean(lm_cv_scores$MAPE)
#rmse 98834.61
#rmsle 0.164
#mape 16.37
  

#plot the mape
ggplot(data=lm_cv_scores, aes(x=.id, y=MAPE)) + geom_line() + geom_point() +
  labs(title="rolling CV results, LM model with time & calendar features",
       subtitle="starts from 2014, each id is 15 days of test data")
#scores decline linearly from 35 MAPE to 5-10
  #except for the peaks in error likely in late 2014-2015 due to cyclicality




###Get residuals from model 1####

#fit LM on 2013-2017
lm_model_full = ts_lm %>%
  model(TSLM(agg_sales ~ . + 
               trend(knots=c(date("2015-06-01"))) +
               fourier(period=28, K=7) +
               fourier(period=7, K=2) -
               date - agg_sales
  )
  )


#get decomped resids from LM
total_sales_decomped = lm_model_full %>%
  residuals()


#put decomped resids in a tsibble, reverse log transformation
ts_decomped = ts_lm %>%
  mutate(agg_sales=total_sales_decomped$.resid)




#EDA 2 - LAGS AND COVARIATES####

#perform EDA for sales lags, covariates, covariate lags and rolling features,
  #use the residuals from the LM model
    #all of them, or just 2017 residuals???


#add in oil, onpromotion, transactions
covariates_total = ts_train %>%
  summarise(oil=mean(oil), onpromotion=sum(onpromotion), transactions=sum(transactions)) 
ts_decomped = left_join(x=ts_decomped, y=covariates_total, by="date")


#test covariates' stationarity
  #kpss null: data is stationary around a linear trend
  #pp null: data is non-stationary around a constant trend

unitroot_kpss(ts_decomped$oil)
#null rejected, oil not stationary around a linear trend
unitroot_pp(ts_decomped$oil)
#null accepted, oil not stationary around a constant trend


unitroot_kpss(ts_decomped$onpromotion)
#null rejected, onpromotion not stationary around a linear trend
unitroot_pp(ts_decomped$onpromotion)
#null rejected, onpromotion stationary around a constant trend


unitroot_kpss(ts_decomped$transactions)
#null rejected, transactions not stationary around a linear trend
unitroot_pp(ts_decomped$transactions)
#null rejected, transactions stationary around a constant trend


#determine number of differences necessary to make covariates stationary
unitroot_ndiffs(ts_decomped$oil) #once
unitroot_ndiffs(ts_decomped$onpromotion) #once
unitroot_ndiffs(ts_decomped$transactions) #once


#difference all covariates
ts_decomped = ts_decomped %>%
  mutate(oil=difference(oil),
         onpromotion=difference(onpromotion),
         transactions=difference(transactions))
ts_decomped[is.na(ts_decomped)] = 0




##sales lags X total sales####


#PACF
ts_decomped %>%
  PACF(agg_sales) %>% autoplot() +
  labs(title="PACF of decomposed residuals, total sales")
#lag 1 0.8 PACF
#lags 2-7 0.1-0.2 PACF
#last significant lag at ~61


#scatterplots
ts_decomped %>%
  gg_lag(agg_sales, geom="point")
#lag 1 super linear




##oil X total sales####


#CCF (cross covariation / cross correlation)

#long term pattern
ts_decomped %>%
  CCF(x=oil, y=agg_sales, lag_max=28, type="correlation") %>% autoplot()
#sigmoidal pattern in CCF plots, declines as it gets farther from origin point
#few significant spikes, though all low at 0.06-0.04 corr
  #biggest one in lag 7, others 2, 3, 8
  #several in leads 1-60
#LOOK INTO INTERPRETATION, MAYBE SCALING?


#plots




##onpromotion X total sales####


#transactions X total_sales####








#FEATURE ENGINEERING 2####




#MODEL 2 - LAGS AND COVARIATES####



























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


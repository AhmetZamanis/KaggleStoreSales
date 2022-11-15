#KAGGLE STORE SALES TIME SERIES FORECASTING - SECOND ATTEMPT

#LIBRARIES####

#data handling
library(tidyverse)
library(tsibble) #time series tibbles
library(lubridate) #date objects
library(pracma) #exponential weighted moving averages

#visualization
library(patchwork)
library(GGally)

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
#WARNING: THIS VERSION IS CPI ADJUSTED!


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






##LM1 time & calendar model, total sales####


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
       subtitle="model: lm1 with 2-piece trend, fourier pairs, calendar dummies",
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
       subtitle="model: lm1 with 2-piece trend, fourier pairs, calendar dummies",
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
  labs(title="innovation residuals diagnostics, lm1 model")
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
  labs(title="fitted vs residuals plot, lm1 model",
    x="fitted", y="residuals")
#4 huge outliers (new years days)
  #besides the outliers, the residuals are larger around fitted 0,
    #and get smaller with lower and higher fitted values
    #maybe the model isn't great at predicting zero or small sales




###LM1 RESULT####
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


#rolling stat, scaterplot
ts_decomped %>%
  mutate(sales_ma7 = movavg(ts_decomped$agg_sales, n=7, type="e")) %>%
  ggplot(aes(x=sales_ma7, y=agg_sales)) + geom_point() + geom_smooth() +
  labs(title="total decomposed sales and its EMA7",
        x="sales EMA7",
        y="sales at 0")
#very linear. maybe even better than single lags? check correlations


#rolling stat, timeplot
ts_decomped %>%
  mutate(sales_ma7 = movavg(ts_decomped$agg_sales, n=7, type="e")) %>%
  select(agg_sales, sales_ma7) %>%
  pivot_longer(cols=agg_sales:sales_ma7, names_to="series") %>%
  ggplot(aes(x=date, y=value, color=series)) + geom_line() +
  labs(title="total decomposed sales and its EMA7",
       x="date",
       y="total sales and its EMA7")
#follows the cylicality well



#spearman correlations with lag 1 and ma7
sales_lags = ts_decomped %>%
  mutate(sales_ma7 = movavg(ts_decomped$agg_sales, n=7, type="e"),
         sales_1 = lag(agg_sales, n=1)) %>%
  select(agg_sales, sales_ma7, sales_1)

cor(na.omit(sales_lags[1:3]), method="spearman")
#0.86 between sales and sales ma7
#0.75 between sales and sales lag 1
#0.87 between sales lag 1 and sales ma 7
#use the MA. or an ARIMA component?



##oil X total sales####


#timeplot of total sales and oil, scaled
ts_decomped %>%
  autoplot(scale(agg_sales)) /
  ts_decomped %>%
  autoplot(scale(oil)) +
  plot_annotation(title="total sales (decomposed) vs. oil prices (differenced)")
#it looks like sharp movements in oil price leads to sharp movements in sales in ~ 7 days
#still doesn't fully explain the cyclicality in 2014-2015




#scatterplot of total sales and oil at time T, scaled
ts_decomped %>%
  ggplot(aes(x=scale(oil), y=scale(agg_sales))) +
  geom_point()
#no apparent relation




#CCF (cross covariation / cross correlation)
ts_decomped %>%
  CCF(x=oil, y=agg_sales, lag_max=365, type="correlation") %>% autoplot() +
  labs(title="cross correlation of total sales (decomposed) and oil (differenced)",
       x="oil lags & leads",
       y="cross-correlation with total sales")


ts_decomped %>%
  CCF(x=oil, y=agg_sales, type="correlation") %>% autoplot() +
  labs(title="cross correlation of total sales (decomposed) and oil (differenced)",
       x="oil lags & leads",
       y="cross-correlation with total sales")
#sigmoidal pattern in CCF plots, declines as it gets farther from origin point
#few significant spikes, though all low at 0.06-0.04 corr
  #biggest one in lag 7, others 2, 3, 8
  #several in leads 1-60




#plots of significant lags

#lag 7
ts_decomped %>%
  mutate(oil_7 = lag(oil, 7)) %>%
  ggplot(aes(x=scale(oil_7), y=scale(agg_sales))) +
  geom_point() + geom_smooth() +
  labs(title="total sales (decomposed) vs. oil prices at lag 7 (differenced)",
       x="oil price change, scaled",
       y="total sales, scaled")
#no relation for small movements in oil price
#but linear increase in sales with large increase in oil price, for the few outliers
#non-linear, tangent-like relationship


#same plot, with extreme values only
ts_decomped %>%
  mutate(oil_7 = scale(lag(oil, 7)),
         agg_sales = scale(agg_sales)) %>%
  filter(oil_7 < -2.5 | oil_7 > 2.5) %>%
  ggplot(aes(x=oil_7, y=agg_sales)) +
  geom_point() + geom_smooth() +
  labs(title="total sales (decomposed) vs. oil prices at lag 7 (differenced)",
       subtitle="extreme values only",
       x="oil price change, scaled",
       y="total sales, scaled")
#linearish increase in sales with large increase in oil price 




#lags 2, 3, 8


#lag 2
ts_decomped %>%
  mutate(oil_2 = scale(lag(oil, 2)),
         agg_sales = scale(agg_sales)) %>%
  ggplot(aes(x=oil_2, y=agg_sales)) +
  geom_point() + geom_smooth() +
  labs(title="total sales (decomposed) vs. oil prices at lag 2 (differenced)",
       x="oil price change, scaled",
       y="total sales, scaled")
#slight linear decline in sales as oil price increases


#lag 3
ts_decomped %>%
  mutate(oil_3 = scale(lag(oil, 3)),
         agg_sales = scale(agg_sales)) %>%
  ggplot(aes(x=oil_3, y=agg_sales)) +
  geom_point() + geom_smooth() +
  labs(title="total sales (decomposed) vs. oil prices at lag 3 (differenced)",
       x="oil price change, scaled",
       y="total sales, scaled")
#decline in sales as oil price increases, only for negative outliers
#increase in sales as oil price increases, only for positive outliers
#neutral for most observations
#non-linear reverse U shaped relationship


#lag 8
ts_decomped %>%
  mutate(oil_8 = scale(lag(oil, 8)),
         agg_sales = scale(agg_sales)) %>%
  ggplot(aes(x=oil_8, y=agg_sales)) +
  geom_point() + geom_smooth() +
  labs(title="total sales (decomposed) vs. oil prices at lag 8 (differenced)",
       x="oil price change, scaled",
       y="total sales, scaled")
#slight linear decline in sales as oil price increases




#compute some sort of rolling stat from lags 2-8? or another range?
  #consider 21 and 28 day rolling stats, as they had the highest corr in the first analysis
  #especially look at the relationship for extreme oil price change values

#time plot with sales and various oil rolling MAs
ts_decomped %>%
  mutate(oil_ma7 = scale(movavg(ts_decomped$oil, n=7, type="e")),
         #oil_ma14 = scale(movavg(ts_decomped$oil, n=14, type="e")),
         oil_ma21 = scale(movavg(ts_decomped$oil, n=21, type="e")),
         #oil_ma28 = scale(movavg(ts_decomped$oil, n=28, type="e")),
         oil_ma54 = scale(movavg(ts_decomped$oil, n=54, type="e")),
         #oil_ma112 = scale(movavg(ts_decomped$oil, n=112, type="e")),
         oil_ma168 = scale(movavg(ts_decomped$oil, n=168, type="e")),
         agg_sales = scale(agg_sales)) %>%
  select(agg_sales,
         oil_ma7, 
         #oil_ma14, 
         oil_ma21, 
         #oil_ma28, 
         oil_ma54, 
         #oil_ma112, 
         oil_ma168) %>%
  pivot_longer(cols = agg_sales:oil_ma168,
               names_to = "series") %>%
  ggplot(aes(x=date, y=value, color=series)) + geom_line() + geom_smooth()
#looks like oil prices changes MAs and total sales diverge at cyclical times like 2014-2015
#otherwise, they fluctuate around 0
  #narrower periods seem more stable, while larger periods fluctuate more?
  #hard to read this plot


#scatterplot with sales and oil rolling MAs
ts_decomped %>%
  mutate(oil_ma7 = scale(movavg(ts_decomped$oil, n=7, type="e")),
         oil_ma14 = scale(movavg(ts_decomped$oil, n=14, type="e")),
         oil_ma21 = scale(movavg(ts_decomped$oil, n=21, type="e")),
         oil_ma28 = scale(movavg(ts_decomped$oil, n=28, type="e")),
         oil_ma54 = scale(movavg(ts_decomped$oil, n=54, type="e")),
         oil_ma112 = scale(movavg(ts_decomped$oil, n=112, type="e")),
         oil_ma168 = scale(movavg(ts_decomped$oil, n=168, type="e")),
         agg_sales = scale(agg_sales)) %>%
  select(agg_sales,
         oil_ma7, 
         oil_ma14, 
         oil_ma21, 
         oil_ma28, 
         oil_ma54, 
         oil_ma112, 
         oil_ma168) %>%
  ggplot(aes(x=oil_ma54, y=agg_sales)) + geom_point() + geom_smooth() +
  labs(title="decomposed total sales and differenced oil EMA",
       x="54-period EMA of oil, scaled",
       y="total sales, scaled")
#ma7: slight decline in sales as oil increases, uncertain relationship
#ma14: pronounced decline in sales as oil increases, more certain relationship than 7
  #influenced by 2 large outliers to the right
#ma21: more certain version of 14
#ma28: bit less certain version of 21
#ma54: decline in sales as oil increases, even more certain and linear relationship
#ma112: close to no relationship
#ma168: close to no relationship




#calculate spearman correlations for the MAs
ts_oil_ma = ts_decomped %>%
  mutate(oil_ma7 = scale(movavg(ts_decomped$oil, n=7, type="e")),
         oil_ma14 = scale(movavg(ts_decomped$oil, n=14, type="e")),
         oil_ma21 = scale(movavg(ts_decomped$oil, n=21, type="e")),
         oil_ma28 = scale(movavg(ts_decomped$oil, n=28, type="e")),
         oil_ma54 = scale(movavg(ts_decomped$oil, n=54, type="e")),
         oil_ma112 = scale(movavg(ts_decomped$oil, n=112, type="e")),
         oil_ma168 = scale(movavg(ts_decomped$oil, n=168, type="e")),
         agg_sales = scale(agg_sales)) %>%
  select(agg_sales,
         oil_ma7, 
         oil_ma14, 
         oil_ma21, 
         oil_ma28, 
         oil_ma54, 
         oil_ma112, 
         oil_ma168) 

cor(ts_oil_ma[1:8], method="spearman")
#strongest corr with MA 51, -0.1836
  #54 correlated 0.66 with 7. others all high
  #maybe use MA54, + lags from 2-8?


#check the spearman correlations of oil lags 2, 3, 7, 8
oil_lags = ts_decomped %>%
  mutate(oil_2 = lag(oil, 2),
         oil_3 = lag(oil, 3),
         oil_7 = lag(oil, 7),
         oil_8 = lag(oil, 8)) %>%
  select(oil_2:oil_8)

cor(na.omit(oil_lags[1:4]), method="spearman")
#they are not correlated at all








##onpromotion X total sales####


#timeplot of total sales and onpromotion, scaled
ts_decomped %>%
  autoplot(scale(agg_sales)) /
  ts_decomped %>%
  autoplot(scale(onpromotion)) +
  plot_annotation(title="total sales (decomposed) vs. onpromotion (differenced)")
#noisy plot, but 2 observations:
  #the relative number of change of items on promotion is very low before mid 2014
  #then it ramps up until mid 2015
  #afterward it fluctuates around a very steady constant
  #possibly because promotions became more utilized after the 2014-2015 cyclicality?

  
#scatterplot of total sales and onpromotion at time T, scaled
  ts_decomped %>%
    ggplot(aes(x=scale(onpromotion), y=scale(agg_sales))) +
    geom_point() + geom_smooth() +
    labs(title="total decomposed sales and differenced onpromotion",
         x="onpromotion, scaled",
         y="total sales, scaled")
#very slight linearish increase in sales with increase in onpromotion
  #could do decently with LM


  
#CCF (cross covariation / cross correlation)
  ts_decomped %>%
    CCF(x=onpromotion, y=agg_sales, lag_max=28, type="correlation") %>% autoplot() +
    labs(title="cross correlation of total sales (decomposed) and onpromotion (differenced)",
         x="onpromotion lags & leads",
         y="cross-correlation with total sales")
#clear weekly pattern
#lags & leads that are the multiple of 7 are significant, but very low correlation
  #for lags, it's 0 to 119
  #for leads, it's up until 56
  #only use the most recent ones, if any
#most significant correlation is at T 0,
  #followed by lead 7, lag 7, lead 14...
#consider using current value of onpromotion + lag 7
  #would lead 7 be hard to impute for the test data?
  #also consider rolling stats?

  
#plots of significant lags / leads

    
#lag 7
ts_decomped %>%
  mutate(onp_7 = lag(onpromotion, 7)) %>%
  ggplot(aes(x=scale(onp_7), y=scale(agg_sales))) +
  geom_point() + geom_smooth() +
  labs(title="total sales (decomposed) vs. onpromotion at lag 7 (differenced)",
         x="onpromotion change, scaled",
         y="total sales, scaled")  
#slight linear increase with a large increase in onpromotion,
#otherwise no effect. 
  #it's likely a large increase in promotion increases sales, but a decrease doesn't drop them


#lead 7
ts_decomped %>%
  mutate(onp_7_lead = lead(onpromotion, 7)) %>%
  ggplot(aes(x=scale(onp_7_lead), y=scale(agg_sales))) +
  geom_point() + geom_smooth() +
  labs(title="total sales (decomposed) vs. onpromotion at lead 7 (differenced)",
       x="onpromotion change, scaled",
       y="total sales, scaled")  
#same relationship as lag 7
  #this is likely because of the weekly seasonality in onpromotion
  #maybe promotions are announced 1 week in advance all the time?




#time plot with sales and various onpromotion rolling MAs
ts_decomped %>%
  mutate(onp_ma7 = scale(movavg(ts_decomped$onpromotion, n=7, type="e")),
         onp_ma14 = scale(movavg(ts_decomped$onpromotion, n=14, type="e")),
         onp_ma21 = scale(movavg(ts_decomped$onpromotion, n=21, type="e")),
         agg_sales = scale(agg_sales)) %>%
  select(agg_sales,
         onp_ma7,
         onp_ma14,
         onp_ma21) %>%
  pivot_longer(cols = agg_sales:onp_ma21,
               names_to = "series") %>%
  ggplot(aes(x=date, y=value, color=series)) + geom_line() 
#too noisy to read




#scatterplot with sales and various onpromotion rolling MAs 
ts_decomped %>%
  mutate(onp_ma7 = scale(movavg(ts_decomped$onpromotion, n=7, type="e")),
         onp_ma14 = scale(movavg(ts_decomped$onpromotion, n=14, type="e")),
         onp_ma21 = scale(movavg(ts_decomped$onpromotion, n=21, type="e")),
         agg_sales = scale(agg_sales)) %>%
  select(agg_sales,
         onp_ma7,
         onp_ma14,
         onp_ma21) %>%
  ggplot(aes(x=onp_ma7, y=agg_sales)) + geom_point() + geom_smooth() +
  labs(title="total decomposed sales vs EMA7 of differenced onpromotion",
       x="EMA7 of differenced onpromotion",
       y="total decomposed sales")
#for extremely low values of ma7, an increase leads to a strong increase in sales
  #but this is strongly impacted by few outliers
#for the rest, ma7 has no effect on sales, except:
  #a small decrease from around 0 will cause a sharp drop in sales
  #can't likely model this with LM
#same story for MA 14 and 21
#MA 7 is the most linear one, so use that if you will









##transactions X total_sales####

#consider only lag 15+ as that's what you'll have for test data


#timeplot of total sales and transactions, scaled
ts_decomped %>%
  autoplot(scale(agg_sales)) /
  ts_decomped %>%
  autoplot(scale(transactions)) +
  plot_annotation(title="total sales (decomposed) vs. transactions (differenced)")
#transactions are very stationary with seasonality similar to sales
#there are drops and peaks around new years




#scatterplot of total sales and transactions at time T, scaled
ts_decomped %>%
  ggplot(aes(x=scale(transactions), y=scale(agg_sales))) +
  geom_point() + geom_smooth() +
  labs(title="total decomposed sales and differenced transactions",
       x="transactions, scaled",
       y="total sales, scaled")
#for the vast majority of obs, more transactions means a linearish increase in sales
#when you factor in outliers, the relationship is sin-like



#CCF (cross covariation / cross correlation)
ts_decomped %>%
  CCF(x=transactions, y=agg_sales, lag_max=28, type="correlation") %>% autoplot() +
  labs(title="cross correlation of total sales (decomposed) and transactions (differenced)",
       x="transactions lags & leads",
       y="cross-correlation with total sales")
#only significant for a few lags and leads around T 0
  #most significant at lead 2, then 0, then lead 1 and 7
  #this is likely caused by some other factor affecting sales & transactions for a few days
#likely not a significant predictor for total sales as leads are unusable in test data.
#still worth to consider for subcategories and for insight



#plots of significant lags / leads


#lag 2
ts_decomped %>%
  mutate(trans_7 = lag(transactions, 2)) %>%
  ggplot(aes(x=scale(trans_7), y=scale(agg_sales))) +
  geom_point() + geom_smooth() +
  labs(title="total sales (decomposed) vs. transactions at lag 2 (differenced)",
       x="transactions change, scaled",
       y="total sales, scaled")
#no relationship for vast majority of obs









#FEATURE ENGINEERING 2####


#add oil features to ts_decomped:
#lags 2, 3, 7, 8
#EMA 54
ts_decomped = ts_decomped %>%
  mutate(oil_2 = lag(oil, 2),
         oil_3 = lag(oil, 3),
         oil_7 = lag(oil, 7),
         oil_8 = lag(oil, 8),
         oil_ma54 = movavg(ts_decomped$oil, n=54, type="e"))

#check NAs
colSums(is.na(ts_decomped))
#a few in the oil lags
#fill them with their medians
ts_decomped = ts_decomped %>%
  mutate(oil_2 = ifelse(is.na(oil_2), median(na.omit(oil_2)), oil_2),
         oil_3 = ifelse(is.na(oil_3), median(na.omit(oil_3)), oil_3),
         oil_7 = ifelse(is.na(oil_7), median(na.omit(oil_7)), oil_7),
         oil_8 = ifelse(is.na(oil_8), median(na.omit(oil_8)), oil_8))


#drop original oil column
ts_decomped = ts_decomped %>%
  select(-oil)


#do keep the onpromotion features, but:
  #they likely won't do well in LM. don't use them in LMs
    #except onpromotion at time T. it may be modeled decently with LM
  #they probably matter less for total sales and more for specific categories
ts_decomped = ts_decomped %>% 
  mutate(onp_7 = lag(onpromotion, n=7),
         onp_ma7 = movavg(ts_decomped$onpromotion, n=7, type="e"))
ts_decomped[is.na(ts_decomped)] = 0


#don't use transactions as a predictor, at least for total sales
ts_decomped = ts_decomped  %>%
  select(-transactions)




#create tsibble for the second model
ts_lm2 = ts_decomped


#add sales MA to ts_decomped
ts_lm2 = ts_lm2 %>%
  mutate(sales_ma7 = movavg(ts_lm2$agg_sales, n=7, type="e"))



#check NAs
colSums(is.na(ts_lm2)) > 0
#no issue


#drop time and calendar features
ts_lm2 = ts_lm2 %>%
  select(!(local_holiday:sunday))



#MODEL 2 - LAGS AND COVARIATES####


#use sales_ma7 to account for autoregression, or use an ARIMA component?
#how to scale and center the train-test data? with a recipe?
  #don't bother with this for fable. even if you do it for a train-valid split, it won't work for CV

##LM2 lags & covariates model, total sales####


#create validation set
ts_lm2_val = ts_lm2 %>%
  filter(year(date)==2017)


#fit LM2 on 2013-2016
lm_model2 = ts_lm2 %>%
  filter(year(date)!=2017) %>%
  model(TSLM(agg_sales ~ . - onp_7 - onp_ma7 -
               date - agg_sales
  )
  )


#predict LM on 2017
lm_preds2 = lm_model2 %>%
  forecast(ts_lm2_val) 


#add back lm_preds to get the hybrid final predictions
lm_preds_final = lm_preds2 %>%
  mutate(agg_sales = agg_sales + lm_preds$agg_sales,
         .mean = .mean + lm_preds$.mean)



#plot actual vs predicted, time plot
lm_preds_final %>%
  autoplot(ts_lm_val) +
  labs(title="predicted vs actual total sales",
       subtitle="hybrid results of lm1 and lm2",
       y="total sales") +
  theme_bw()
#catches the magnitude of most waves much better, still misses a few
  #still a few slight timing mistakes
#accounts for start of year and end of data better, though not perfect



#plot actual vs predicted, scatterplot
ts_lm_val %>%
  ggplot(aes(x=agg_sales, y=lm_preds_final$.mean)) +
  geom_point() +
  geom_abline(slope=1, intercept=0, color="red") +
  labs(title="predicted vs actual total sales",
       subtitle="hybrid results of lm1 and lm2",
       y="total sales, predicted",
       x="total sales, actual") +
  theme_bw()




#score the predictions with rmse, mape and rmsle
#reverse the log transform for mape
lm_scores_final = lm_preds_final %>% 
  mutate(agg_sales=exp(agg_sales)) %>%
  accuracy(ts_lm_val %>%
             mutate(agg_sales=exp(agg_sales)))

#get the RMSLE as well
lm_scores_final$RMSLE = lm_preds_final %>%
  accuracy(ts_lm_val) %>%
  select(RMSE)

#scores
lm_scores_final$RMSE
lm_scores_final$RMSLE
lm_scores_final$MAPE
#RMSE 45600
#RMSLE 0.061
#MAPE 5.1%
#bit better than lm1 alone



#diagnose the lm2 model fit


#retrieve LM2 model report
lm_model2 %>% report()
#r square 79%
#large and significant coefs:
#onpromotion, oil_7, sales_ma7 ***
#oil_2, oil_ma54 *
#oil_3 .
#oil_8 insignificant


#innovation residuals analysis
lm_model2 %>% gg_tsresiduals() +
  labs(title="innovation residuals diagnostics, lm2 model")
#much more stationary
  #generally deals with 2014-2015 cyclicality, still misses some small peaks and drops
  #misses NY peak at the start of 2013 (because no precedent)
  #still misses the christmas-NY effects a little
#significant ACF lags: 1, 7, 27, 28
  #probably still not capturing the weakly seasonality fully. maybe increase fourier orders?


#test stationarity

#kpss test
augment(lm_model2) %>%
  features(.innov, unitroot_kpss)
#null hypothesis: data is stationary around a linear trend
  #the null is accepted with a p value of 0.1 or higher
  #the residuals are stationary with a possible trend
  #test stat 0.084


#philips-perron test
augment(lm_model2) %>%
  features(.innov, unitroot_pp)
#null hypothesis: data is non-stationary around a constant trend
  #the null is rejected with a p value of 0.01 or lower
  #the residuals are stationary with no trend
  #test stat -26.1


#plot residuals against fitted values
augment(lm_model2) %>%
  ggplot(aes(x=.fitted, y=.resid)) +
  geom_point() +
  labs(title="fitted vs residuals plot, lm2 model",
       x="fitted", y="residuals")
#few huge outliers to the left
#main mass of residuals distributed much more normally compared to lm1
  #still some large negative residuals for small sales predictions



##LM1-LM2 HYBRID RESULT####
#improves the fit and generally accounts for the 2014-2015 cyclicality
#predictions are improved, RMSE 45600, RMSLE 0.061, MAPE 5.1%
#ACF autocorrelations remain for lags 1, 7 and 28. likely a little weekly seasonality left






#RECONCILIATION#### 


# #figure out aggregation
# ts_test = ts_train %>%
#   aggregate_key(
#     category * store_no,
#     sales = sum(sales)
#   )
# #the above code results in:
# #total sales in each day, 1688 rows (category: aggregated, store_no:aggregated)
# #sales in each category, for each day and store (category: category name, store_no:aggregated)
# #sales in each store, for each category and each day (category: aggregated, store_no:store name)
# #sales in each store:category combo, in each day (disaggregated data)
# #3156560 rows compared to 3008016 original rows = 148544 new rows
# 
# 
# ts_test %>%
#   filter(is_aggregated(category)) %>%
#   autoplot(sales)
# #this filters and plots total sales + category totals for each store
# 
# ts_test %>%
#   filter(is_aggregated(store_no), !is_aggregated(category)) %>%
#   autoplot(sales)
# #this filters and plots category totals
# 
# ts_test %>%
#   filter(!is_aggregated(category), is_aggregated(store_no)) %>%
#   autoplot(sales) +
#   facet_wrap(vars(category))
# #this puts each category total on a different plot
# 
# ts_test %>%
#   filter(category=="GROCERY I", is_aggregated(store_no)) %>%
#   autoplot(sales)
# #this filters and plots a single category total
# 
# ts_test %>%
#   filter(is_aggregated(store_no),
#          (category %in% c("GROCERY I", "BREAD/BAKERY", "MEATS", "SEAFOOD"))) %>%
#   autoplot(sales) +
#   facet_wrap(vars(category))
# #this should filter and plot 4 category totals, but it's taking way too long?
#   #throws error:
# # Error in `filter()`:
# #   ! Problem while computing `..2 = category %in% ...`.
# # ✖ Input `..2` must be of size 3156560 or 1, not size 2.
# # Run `rlang::last_error()` to see where the error occurred.
# 
# ts_test %>%
#   filter(category %in% c("GROCERY I", "BREAD/BAKERY", "MEATS", "SEAFOOD"))
# #why doesn't this work???




##Create hierarchy mapped ts####
rm(ts_test)
ts_hier = ts_train %>%
  select(-transactions) %>%
  aggregate_key(category * store_no,
    sales = sum(sales),
    onpromotion = sum(onpromotion),
    oil = mean(oil),
    across(local_holiday:sunday, mean)
  )
#this is causing R to freeze, taking up 12GB of memory
#move to darts






#OUTPUT MODIFIED DATA####


#Reload modified version 2 of ts_train
ts_train = read.csv("./ModifiedData/train_modified2.csv", encoding="UTF-8", header=TRUE)
ts_train$date = as_date(ts_train$date)
ts_train = as_tsibble(ts_train, key=c("category", "city", "state", "store_no", "store_type",
                                      "store_cluster"), index="date")


#reload modified version 1 of test data
ts_test = read.csv("./ModifiedData/test_modified.csv", encoding="UTF-8", header=TRUE)
ts_test$date = as_date(ts_test$date)
ts_test = as_tsibble(ts_test, key=c("category", "city", "state", "store_no", "store_type",
                                      "store_cluster"), index="date")


#reorder columns
ts_train = ts_train %>%
  relocate(category, .before = sales) %>%
  relocate(store_no, .before = sales) %>%
  relocate(store_cluster, .before = sales) %>%
  relocate(store_type, .before = sales) %>%
  relocate(city, .before = sales) %>%
  relocate(state, .before = sales)


ts_test = ts_test %>%
  relocate(category, .before = id) %>%
  relocate(store_no, .before = id) %>%
  relocate(store_cluster, .before = id) %>%
  relocate(store_type, .before = id) %>%
  relocate(city, .before = id) %>%
  relocate(state, .before = id)




##Modify test data to match with train data####


#drop old versions of calendar features
ts_test = ts_test %>% select(-c("payday", "new_years_day", "christmas", "earthquake"))


#convert holiday features to dummies
ts_test$local_holiday = ts_test$local_holiday %>%
  recode("True" = 1, "False" = 0)

ts_test$regional_holiday = ts_test$regional_holiday %>%
  recode("True" = 1, "False" = 0)

ts_test$national_holiday = ts_test$national_holiday %>%
  recode("True" = 1, "False" = 0)



#payday features:
  #payday_16: day 16
  #payday_31: day 31
  #payday_n: day 1-6
ts_test = ts_test %>%
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
ts_test = ts_test %>%
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
ts_test = ts_test %>%
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

ts_test = ts_test %>%
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
ts_test = ts_test %>%
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
ts_test = ts_test %>%
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


ts_test = ts_test %>%
  select(-event)


#days of week dummies: use monday as intercept, because it has the least fluctuation-outliers
ts_test = ts_test %>%
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
colSums(is.na(ts_test)) 
#we have a few NAs in the holiday leads columns
  #2017-08-29, 30, 31
  #check if there are any holidays in 2017-09-01

df_holidays = read.csv("./OriginalData/holidays_events.csv", encoding="UTF-8", header=TRUE)
#no holidays in 2017-09-01. replace the na's with 0's
ts_test[is.na(ts_test)] = 0
#yes, that should work


#drop id from test data
ts_test = ts_test %>%
  select(-id)



##Save modified version 2 of train and test####
  #actions performed on this data:
    #merge with stores, holidays-events, oil, transactions data (python iter 1)
    #missing oil values interpolated (python iter 1)
    #missing dates in train filled in and interpolated  (R iter 1)
      #(december 25, sales, oil, transactions, onpromotion)
    #CPI adjusted (sales and oil, both train and test) (R iter 1)
    #holidays-events-calendar features readjusted (R iter 1)

#check gaps
scan_gaps(ts_train)    
scan_gaps(ts_test) 
#none

#check NAs
colSums(is.na(ts_train)) > 0
colSums(is.na(ts_test)) > 0
#none

#check column names
names(ts_train)
names(ts_test)
setdiff(names(ts_train), names(ts_test))
#only difference: sales and transactions


#save data
write.csv(ts_train, "./ModifiedData/train_modified2.csv", row.names = FALSE)
write.csv(ts_test, "./ModifiedData/test_modified2.csv", row.names = FALSE)








# #add lags and covariates:
#   #THESE ARE SPECIFIC TO THE HIERARCHY NODE. CAN'T DO THIS AT THIS POINT.
# 
# 
# #oil
# ts_train = ts_train %>%
#   mutate(oil_2 = lag(oil, 2),
#        oil_3 = lag(oil, 3),
#        oil_7 = lag(oil, 7),
#        oil_8 = lag(oil, 8),
#        oil_ma54 = movavg(ts_train$oil, n=54, type="e"), .after=oil)
# ts_train[is.na(ts_train)] = 93.14
# 
# 
# #onpromotion
# ts_train = ts_train %>%
#     mutate(onp_7 = lag(onpromotion, n=7),
#        onp_ma7 = movavg(ts_train$onpromotion, n=7, type="e"), .after=onpromotion)
# ts_train[is.na(ts_train)] = 0
# 
# 
# #sales
# ts_train = ts_train %>%
#   mutate(sales_ma7 = movavg(ts_train$sales, n=7, type="e"), .after=sales)

#UNUSED CODE





# #first get them for total sales
# res_lm1 = model_lm1.residuals(
#   series = y_train["sales"],
#   future_covariates = time_covariates[0],
#   forecast_horizon=15
# )
# #gives residuals from 2013-02-06 and onwards
# 
# 
# #then loop over all target components except first, and stack them with res_lm1
# for i in range(1, len(y_train.components)):
#   res_comp = model_lm1.residuals(
#     series = y_train[y_train.components[i]],
#     future_covariates = time_covariates[i],
#     forecast_horizon = 15
#     )
#   res_lm1 = res_lm1.stack(res_comp)
# 
# del res_comp, i
# #takes a very long time, consider saving these if they work
#   #res_lm1.to_csv()
#   #alternative: go back to wide pandas df with targets, and get the time covariates as list of pandas df's
#     #then fit and predict on all series using sklearn and a loop. y=df_targets[,i], x=list_time_covars[i]
#     #join predictions into a wide pandas df just like targets, and turn them into residuals
#     #make the residuals a darts ts and map hieararchy.
#     #this way, you can also avoid specifying sales lags
#   #alternative 2: get historical predictions for 2014-2017, all nodes (already done for top level),
#     #join them into 1 series, for all nodes (already done for top level),
#     #slice y_train into 2014-2017
#     #get residuals
#   #alternative 3: train on entire series, then call residuals with retrain=False
#     #does this work tho?









# #create log transform and backtransform invertible mapper
# from darts.dataprocessing.transformers.mappers import InvertibleMapper
# 
# def log_func(x):
#   if x == 0:
#     return x.map(np.log(x+0.0001))
#   else:
#     return x.map(np.log(x))
# 
# 
# def exp_func(x):
#   if np.exp(x)==0.0001:
#     return 0
#   else:
#     return np.exp(x)
# 
# trafo_log = InvertibleMapper(log_func, exp_func)
# 
# 

# for component in ts_train.components:
#   y_train1[component] = trafo_log.transform(ts_train[component])
#   
#   
# #throws error: ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()


# #log throws warnings. try box-cox instead
# from darts.dataprocessing.transformers import BoxCox
# trafo_boxcox = BoxCox()
# 
# #boxcox transform the target series
# y_train1 = trafo_boxcox.fit_transform(ts_train)
# #throwing errors.

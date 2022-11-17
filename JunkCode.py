#UNUSED CODE



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

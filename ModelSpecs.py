Store sales prediction scores
--------
Model = Naive drift
MAE: mean = 6363.13, sd = 3694.66, min = 1156.91, max = 21050.07
MSE: mean = 64192549.51, sd = 76134854.1, min = 2097909.98, max = 457287381.6
RMSE: mean = 7013.42, sd = 3873.57, min = 1448.42, max = 21384.28
RMSLE: mean = 0.95, sd = 0.86, min = 0.62, max = 7.07
--------
Model = Naive seasonal
MAE: mean = 3966.55, sd = 2940.74, min = 1012.22, max = 14603.02
MSE: mean = 33918822.51, sd = 50677321.75, min = 1744277.29, max = 234667386.71
RMSE: mean = 4777.28, sd = 3331.13, min = 1320.71, max = 15318.86
RMSLE: mean = 0.84, sd = 0.86, min = 0.63, max = 7.07
--------
Model = Linear
MAE: mean = 2002.82, sd = 1504.09, min = 585.77, max = 9609.55
MSE: mean = 11410492.23, sd = 25843486.45, min = 596159.8, max = 186283233.07
RMSE: mean = 2684.66, sd = 2050.14, min = 772.11, max = 13648.56
RMSLE: mean = 0.72, sd = 0.88, min = 0.23, max = 7.07
--------
Model = Linear + RF
MAE: mean = 2103.61, sd = 1540.66, min = 577.5, max = 9609.55
MSE: mean = 11758632.46, sd = 25904173.41, min = 569031.59, max = 186283233.07
RMSE: mean = 2749.23, sd = 2049.48, min = 754.34, max = 13648.56
RMSLE: mean = 0.59, sd = 0.91, min = 0.11, max = 7.07
--------
Model = Linear + RNN (global)
MAE: mean = 2008.33, sd = 1498.65, min = 579.57, max = 9575.43
MSE: mean = 11402095.54, sd = 25638472.35, min = 590154.86, max = 184616098.11
RMSE: mean = 2688.82, sd = 2042.63, min = 768.22, max = 13587.35
RMSLE: mean = 0.69, sd = 0.59, min = 0.23, max = 4.95
--------
Model = D-linear (global)
MAE: mean = 2899.19, sd = 1863.69, min = 888.22, max = 9557.04
MSE: mean = 19424505.91, sd = 30125865.27, min = 1374606.52, max = 183539618.23
RMSE: mean = 3682.08, sd = 2422.15, min = 1172.44, max = 13547.68
RMSLE: mean = 0.76, sd = 0.49, min = 0.6, max = 4.34
--------
Model = D-linear (global) + RNN (global)
MAE: mean = 2918.89, sd = 1855.89, min = 893.44, max = 9522.03
MSE: mean = 19570486.82, sd = 29983195.08, min = 1415213.83, max = 181886586.25
RMSE: mean = 3707.05, sd = 2414.18, min = 1189.63, max = 13486.53
RMSLE: mean = 0.77, sd = 0.48, min = 0.61, max = 4.24
--------

Model = Linear (trend only)
MAE: mean = 3131.3, sd = 2347.77, min = 752.36, max = 10208.38
MSE: mean = 23417424.1, sd = 37249525.55, min = 1074358.28, max = 186283233.07
RMSE: mean = 3905.48, sd = 2857.39, min = 1036.51, max = 13648.56
RMSLE: mean = 0.79, sd = 0.86, min = 0.38, max = 7.07
--------
Model = Linear (trend only) + RF (only trend removed, all covars except trend)
MAE: mean = 2312.4, sd = 1696.02, min = 635.87, max = 9609.55
MSE: mean = 14055413.6, sd = 27619496.76, min = 815966.5, max = 186283233.07
RMSE: mean = 3017.0, sd = 2225.56, min = 903.31, max = 13648.56
RMSLE: mean = 0.77, sd = 0.87, min = 0.3, max = 7.07
--------
Model = Linear + RNN (global, only trend removed, all covars except trend)
MAE: mean = 3161.25, sd = 2299.0, min = 893.75, max = 10149.53
MSE: mean = 23440051.16, sd = 36149416.73, min = 1472506.43, max = 174243039.23
RMSE: mean = 3953.56, sd = 2794.54, min = 1213.47, max = 13200.12
RMSLE: mean = 0.76, sd = 0.63, min = 0.4, max = 5.34
--------







Category sales prediction scores with separate linear model and STL + second model
--------
Model = Linear
MAE: mean = 2490.68, sd = 4729.57, min = 4.91, max = 18161.36
MSE: mean = 53357646.8, sd = 157023733.27, min = 33.11, max = 682423288.06
RMSE: mean = 3328.56, sd = 6502.18, min = 5.75, max = 26123.23
RMSLE: mean = 0.46, sd = 0.27, min = 0.2, max = 1.28
--------
Model = Linear + Random forest
MAE: mean = 2689.02, sd = 5158.25, min = 4.97, max = 21686.3
MSE: mean = 58245076.48, sd = 176222421.86, min = 33.96, max = 849969897.75
RMSE: mean = 3483.65, sd = 6790.38, min = 5.83, max = 29154.24
RMSLE: mean = 0.44, sd = 0.28, min = 0.15, max = 1.31
--------





trend & seasonal terms only:

Model=Linear
MAE: mean=2621.4, sd=4923.58, min=5.03, max=19160.13
MSE: mean=70499625.75, sd=212692761.21, min=34.44, max=994111456.04
RMSE: mean=3806.15, sd=7484.17, min=5.87, max=31529.53
RMSLE: mean=0.49, sd=0.25, min=0.3, max=1.2x

+ calendar features (without replacing negative preds with zeroes)

Model=Linear
MAE: mean=2490.91, sd=4729.51, min=4.91, max=18161.36
MSE: mean=53358370.35, sd=157023502.31, min=33.11, max=682423288.06
RMSE: mean=3328.76, sd=6502.13, min=5.75, max=26123.23
RMSLE: mean=0.43, sd=0.23, min=0.2, max=1.28


Model=AutoARIMA (all features)
MAE: mean=3177.22, sd=5938.97, min=3.59, max=24151.07
MSE: mean=65090234.22, sd=174104280.35, min=22.94, max=817507351.93
RMSE: mean=3825.76, sd=7103.09, min=4.79, max=28592.09
RMSLE: mean=0.52, sd=0.61, min=0.2, max=3.67
--------

Model = Linear (time & calendar)
MAE: mean=2490.68, sd=4729.57, min=4.91, max=18161.36
MSE: mean=53357646.8, sd=157023733.27, min=33.11, max=682423288.06
RMSE: mean=3328.56, sd=6502.18, min=5.75, max=26123.23
RMSLE: mean=0.46, sd=0.27, min=0.2, max=1.28

Model = Linear + Random forest (linear decomp, covariate features)
MAE: mean=2949.54, sd=5760.42, min=6.22, max=23345.68
MSE: mean=67404319.58, sd=199819874.33, min=51.01, max=927516253.58
RMSE: mean=3729.04, sd=7314.27, min=7.14, max=30455.15
RMSLE: mean=0.5, sd=0.4, min=0.16, max=1.9
--------

Model = Linear + XGBoost untuned (linear decomp, covariate features)
MAE: mean=3060.46, sd=5984.15, min=6.35, max=24577.0
MSE: mean=76354996.81, sd=223252746.14, min=54.88, max=1061127656.54
RMSE: mean=3989.13, sd=7774.43, min=7.41, max=32574.95
RMSLE: mean=0.53, sd=0.47, min=0.19, max=1.89
--------

Model = Linear (time only)
MAE: mean = 2621.4, sd = 4923.58, min = 5.03, max = 19160.13
MSE: mean = 70499625.75, sd = 212692761.21, min = 34.44, max = 994111456.04
RMSE: mean = 3806.15, sd = 7484.17, min = 5.87, max = 31529.53
RMSLE: mean = 0.49, sd = 0.25, min = 0.3, max = 1.26
--------
Model = Linear + Random forest (STL decomp, covariate + calendar features)
MAE: mean = 2775.22, sd = 5294.9, min = 5.11, max = 22540.5
MSE: mean = 68710801.31, sd = 212215204.9, min = 35.71, max = 1044254758.16
RMSE: mean = 3741.69, sd = 7396.66, min = 5.98, max = 32314.93
RMSLE: mean = 0.48, sd = 0.27, min = 0.26, max = 1.3
--------
Model = Linear + XGBoost untuned (STL decomp, covariate + calendar features)
MAE: mean = 2986.53, sd = 5742.79, min = 5.15, max = 25868.4
MSE: mean = 79903065.08, sd = 259141086.04, min = 36.09, max = 1394927088.03
RMSE: mean = 4095.4, sd = 7945.49, min = 6.01, max = 37348.72
RMSLE: mean = 0.51, sd = 0.36, min = 0.26, max = 1.82
--------

Model = Linear (time & calendar)
MAE: mean=2490.68, sd=4729.57, min=4.91, max=18161.36
MSE: mean=53357646.8, sd=157023733.27, min=33.11, max=682423288.06
RMSE: mean=3328.56, sd=6502.18, min=5.75, max=26123.23
RMSLE: mean=0.46, sd=0.27, min=0.2, max=1.28
--------
Model = Linear + Random forest (STL decomp, covariate features)
MAE: mean = 2689.02, sd = 5158.25, min = 4.97, max = 21686.3
MSE: mean = 58245076.48, sd = 176222421.86, min = 33.96, max = 849969897.75
RMSE: mean = 3483.65, sd = 6790.38, min = 5.83, max = 29154.24
RMSLE: mean = 0.44, sd = 0.28, min = 0.15, max = 1.31
--------
Model = Linear + XGBoost untuned (STL decomp, covariate features)
MAE: mean = 3031.09, sd = 5880.44, min = 4.92, max = 25721.48
MSE: mean = 78410131.67, sd = 243713513.61, min = 33.87, max = 1260756454.48
RMSE: mean = 4075.08, sd = 7861.54, min = 5.82, max = 35507.13
RMSLE: mean = 0.53, sd = 0.48, min = 0.19, max = 2.69
--------







# LINEAR




# XGBOOST

# Category global

# Fit XGB model
model_xgb_cat.fit(
  [decomp_cat["AUTOMOTIVE":"SEAFOOD"][category] for category in categories],
  future_covariates = [x[xgb_covars] for x in x_cat]
)

# Predict validation data for each category
pred_xgb_cat_list = model_xgb_cat.predict(
  n = 227,
  series = [decomp_cat["AUTOMOTIVE":"SEAFOOD"][category] for category in categories],
  future_covariates = [x[xgb_covars] for x in x_cat]
  )

# Stack predictions to get multivariate series
pred_xgb_cat = pred_xgb_cat_list[0].stack(pred_xgb_cat_list[1])
for pred in pred_xgb_cat_list[2:]:
  pred_xgb_cat = pred_xgb_cat.stack(pred)






# D-LINEAR

# 1
model_dlinear_cat = DLinear(
  input_chunk_length = 30,
  output_chunk_length = 1,
  kernel_size = 25,
  batch_size = 32,
  n_epochs = 500,
  model_name = "DLinearCat1",
  log_tensorboard = True,
  save_checkpoints = True,
  random_state = 1923,
  pl_trainer_kwargs = {
    "callbacks": [early_stopper, progress_bar],
    "accelerator": "gpu",
    "devices": [0]
    },
  show_warnings = True,
  force_reset = True
)



# RNN

# 1
model_rnn_cat = RNN(
  model = "LSTM",
  input_chunk_length = 30,
  training_length = 45,
  batch_size = 32,
  n_epochs = 500,
  n_rnn_layers = 2,
  hidden_dim = 32,
  dropout = 0.2,
  model_name = "RNNCat1",
  log_tensorboard = True,
  save_checkpoints = True,
  random_state = 1923,
  pl_trainer_kwargs = {
    "callbacks": [early_stopper, progress_bar],
    "accelerator": "gpu",
    "devices": [0]
    },
  show_warnings = True,
  force_reset = True
)



from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import RichProgressBar, RichModelSummary

# Create early stopper
early_stopper = EarlyStopping(
  monitor = "val_loss",
  min_delta = 5000, # 1% of min. MSE of best model so far
  patience = 10
)

# Progress bar
progress_bar = RichProgressBar()

# Rich model summary
model_summary = RichModelSummary(max_depth = -1)


# Specify D-Linear model
model_dlinear = DLinear(
  input_chunk_length = 30,
  output_chunk_length = 15,
  kernel_size = 25,
  batch_size = 32,
  n_epochs = 500,
  model_name = "DLinearDisagg2.0",
  log_tensorboard = True,
  save_checkpoints = True,
  show_warnings = True,
  optimizer_kwargs = {"lr": 0.002},
  lr_scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau,
  lr_scheduler_kwargs = {"patience": 5},
  pl_trainer_kwargs = {
    "callbacks": [early_stopper, progress_bar, model_summary],
    "accelerator": "gpu",
    "devices": [0]
    }
)


# All covariates, future & past
dlinear2_futcovars = ['tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'day_sin', 'day_cos', "month_sin", "month_cos", 'oil', 'oil_ma28', 'onpromotion', 'onp_ma28', 'local_holiday', 'regional_holiday', 'national_holiday', 'ny1', 'ny2', 'ny_eve31', 'ny_eve30', 'xmas_before', 'xmas_after', 'quake_after', 'dia_madre', 'futbol', 'black_friday', 'cyber_monday']
dlinear2_pastcovars = ["sales_ema7", "transactions", "trns_ma7"]


# Fit DLinear model
model_dlinear.fit(
  series = [y[:-45] for y in y_train_disagg],
  future_covariates = [x[dlinear2_futcovars] for x in x_disagg],
  past_covariates = [x[dlinear2_pastcovars] for x in x_disagg],
  val_series = [y[-45:] for y in y_train_disagg],
  val_future_covariates = [x[dlinear2_futcovars] for x in x_disagg],
  val_past_covariates = [x[dlinear2_pastcovars] for x in x_disagg],
  verbose = True,
  num_loader_workers = 10
)

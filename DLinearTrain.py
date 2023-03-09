


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


# Fit DLinear model
model_dlinear.fit(
  series = [y[:-45] for y in y_train_disagg],
  future_covariates = [x[dlinear2_futcovars] for x in x_disagg],
  past_covariates = [x[dlinear2_pastcovars] for x in x_disagg],
  val_series = [y[-45:] for y in y_train_disagg],
  val_future_covariates = [x[dlinear2_futcovars] for x in x_disagg],
  val_past_covariates = [x[dlinear2_pastcovars] for x in x_disagg],
  verbose = True,
  num_loader_workers = 20
)

# KaggleStoreSales
Time series regression modeling on a dataset of supermarket sales across years, with the Darts library in Python. 

Part 1:
Predicting the total national sales, aggregated across all product categories & stores. Performing time decomposition & hybrid modeling, trying methods such as linear regression with custom features, AutoARIMA and random forest.

[Markdown report, part 1](https://github.com/AhmetZamanis/KaggleStoreSales/blob/main/ReportPart1.md)

Part 2:
Predicting the sales across all hierarchy levels (total sales, store totals and disaggregated series) and performing hierarchical reconciliation. Using Darts implementations of PyTorch global neural networks / deep learning models tailored for time series forecasting, such as D-Linear, LSTM and Temporal Fusion Transformer.

[Markdown report, part 2](https://github.com/AhmetZamanis/KaggleStoreSales/blob/main/ReportPart2.md)

Kaggle competition submission:
A simple submission using an AutoETS model for each of the 1782 disaggregated series resulted in a score of 0.42505 RMSLE, placing 61th out of 612 (top 10%) in the leaderboard at submission time (March 2023).

[Kaggle competition notebook](https://www.kaggle.com/code/ahmetzamanis/store-sales-autoets-with-darts)

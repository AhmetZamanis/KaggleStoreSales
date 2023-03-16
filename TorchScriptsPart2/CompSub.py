
# Load df_train and df_test

# CPI adjust sales














# Wide dataframe of x = date, cols = categories-stores
test_data = ts_sales[categories_stores].pd_dataframe()

# Long format, first all Automotives, then other categories...
pd.melt(test_data, var_name = "category_store_nbr", value_name = "sales", ignore_index = False) 

# Merge the long dataframe above with the id column from the original data, on date and category_store_nbr


# This doesn't work / breaks the original order: 
pd.melt(test_data, var_name = "category_store_nbr", value_name = "sales", ignore_index = False).sort_values(by="date")

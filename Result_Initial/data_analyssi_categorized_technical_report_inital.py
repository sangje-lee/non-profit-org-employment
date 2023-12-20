# %% [markdown]
# ## Data analysis for employment in non-profit organization

# %% [markdown]
# <h3> First portion of the code </h3>

# %% [markdown]
# Portion of the code involved, including,<br />
# - Import entire dataset<br />
# - Filtering some of Indicators<br />
# - Importing it into Panda Profiling files. (Three of them)<br />

# %% [markdown]
# Import all requirement,

# %%
import pandas as pd
import numpy as np
import ydata_profiling as pp  
from ydata_profiling import ProfileReport 
import warnings
import os

warnings.filterwarnings('ignore')

# %%
import seaborn as sns
import matplotlib.pyplot as plt
from fitter import Fitter, get_common_distributions, get_distributions

from sklearn.linear_model import LinearRegression
from scipy.stats import chi2_contingency

import datetime as dt
# import theseus_growth as thg

# %% [markdown]
# Import unemployment dataset.

# %%
df = pd.read_csv('36100651.csv')

print(df.info())
print(df.head(10))

# %% [markdown]
# Filter only the essential columns of the original dataset.

# %%
print("Grab the only the essential part of database.")

# From the original, 
# UOM_ID, SCALAR_ID, VECTOR, COORDINATE, STATUS, SYMBOL, TERMINATED, and DECIMALS columns are removed.

df_sorted = df[['REF_DATE','DGUID','GEO','Sector','Characteristics','Indicators','UOM','SCALAR_FACTOR','VALUE']]

print(df_sorted.head(20))
print(df_sorted.info())

print("Sort by Characteristics")
grouped = df_sorted.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.size]))

print("Sort by Indicator")
grouped = df_sorted.groupby(['Indicators'])
print(grouped['VALUE'].agg([np.size]))


# %% [markdown]
# Check for the missing value from the sorted dataset done above.
# * Notice there is missing value in this dataset.
# * Based on "VALUE" records, there's are 2.86% of the data are missing.

# %%
# Ratio instead of number out ob 
# https://stackoverflow.com/questions/51070985/find-out-the-percentage-of-missing-values-in-each-column-in-the-given-dataset

# Value for "STATUS", "SYMBOL", and "TERMINATED" will be removed after this analysis.
# They contains non-meanful data inside.

percent_missing_df = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'percent_in_na': percent_missing_df,
                                 'num_of_na': df.isnull().sum(),
                                 'total_sample': len(df)})
print("Original database null counter")
print(missing_value_df)

# Noticed that, there's 2.85% of the data (VALUE) is missing.
# To straight forward those missing data, I have decided to further removed some of the missing values.

percent_missing_df_sorted = df_sorted.isnull().sum() * 100 / len(df_sorted)
missing_value_df_sorted = pd.DataFrame({'percent_in_na': percent_missing_df_sorted,
                                 'num_of_na': df_sorted.isnull().sum(),
                                 'total_sample': len(df_sorted)})
print("\nModified dataset null counter.")
print(missing_value_df_sorted)

# %% [markdown]
# Dropping missing value from the sorted dataset.

# %%
df_sorted_na = df_sorted.dropna()

# %% [markdown]
# Check now if there's still a missing data inside modified sorted dataset done above.

# %%
print("Modified dataset modification after removing missing value and it's total counter")

percent_missing_df_sorted_na = df_sorted_na.isnull().sum() * 100 / len(df_sorted_na)
missing_value_df_sorted_na = pd.DataFrame({'percent_in_na': percent_missing_df_sorted_na})
print(missing_value_df_sorted_na)
# print(df_sorted_na.head(20))

print(df_sorted_na.info())
grouped = df_sorted_na.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.size]))

grouped = df_sorted_na.groupby(['Indicators'])
print(grouped['VALUE'].agg([np.size]))

# %% [markdown]
# Panda Profiling for original dataset (CSV file),

# %%
# https://medium.com/analytics-vidhya/pandas-profiling-5ecd0b977ecd

pp = ProfileReport(df, title="Pandas Profiling Report")
pp_df = pp.to_html()

f = open("df_NoMod.html", "a")  # Expert into html file without modifying any columns in dataset.
f.write(pp_df)
f.close()

# %% [markdown]
# Panda Profiling for sorted dataset,

# %%
pp_sorted = ProfileReport(df_sorted, title="Pandas Profiling Report with Columns Sorted")
pp_df_sorted = pp_sorted.to_html()

f = open("df_Sorted.html", "a") # Expert modifying data into html file.
f.write(pp_df_sorted)
f.close()

# %% [markdown]
# Panda Profiling for modified sorted dataset (missing data removed),

# %%
pp = ProfileReport(df_sorted_na, title="Pandas Profiling Report with Columned Sorted and NA Removed")
pp_df_sorted = pp.to_html()

f = open("df_Sorted-no-na.html", "a") # Expert modifying data into html file.
f.write(pp_df_sorted)
f.close()

# %%
# Differences should be, there will be less data to work on.
# Particularly business non-profit organizations and community organizations haven't given more accurate data (more missing values).



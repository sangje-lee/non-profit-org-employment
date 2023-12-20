# %% [markdown]
# <h3> Second part of the code </h3>

# %% [markdown]
# Running the main code required.

# %% [markdown]
# Portion of the code involved, including,<br />
# - Portion of finishing code.<br />
# - Spliting by indicators.<br />
# - Histogram of each indicators.<br />
# - Best fit using package called 'fitter'

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from fitter import Fitter, get_common_distributions, get_distributions

import warnings
import os

warnings.filterwarnings('ignore')

# %%
# If the dataset are missing for this components, it will give a error.

if os.path.isfile("df_byIndicator.csv"):
    df_sorted_na = pd.read_csv('df_byIndicator.csv')

    print(df_sorted_na.info())
    print(df_sorted_na.head(10))
else:
    print("Run main code first before running this.")

# %% [markdown]
# Average annual hours worked from modified sorted dataset.

# %%
# Average annual hours worked        15120
print("\nAverage annual hours worked")
df_AvgAnnHrsWrk = df_sorted_na.loc[
    (df_sorted_na['Indicators'] == 'Average annual hours worked')
]
# grouped = df_AvgAnnHrsWrk.groupby(['GEO'])
grouped = df_AvgAnnHrsWrk.groupby(['Indicators'])
print(grouped['VALUE'].agg([np.median, np.mean, np.std, np.size]))
print("The total number of this one is ",len(df_AvgAnnHrsWrk.index))

sns.displot(data=df_AvgAnnHrsWrk, x="VALUE", kind="hist", bins = 100, aspect = 1.5)
plt.show()

# %% [markdown]
# Average annual wages and salaries from modified sorted dataset. (Mention above)

# %%
# Average annual wages and salaries  15120
print("\nAverage annual wages and salaries")
df_AvgAnnWages = df_sorted_na.loc[
    (df_sorted_na['Indicators'] == 'Average annual wages and salaries')
]
grouped = df_AvgAnnWages.groupby(['Indicators'])
print(grouped['VALUE'].agg([np.median, np.mean, np.std, np.size]))
print("The total number of this one is ",len(df_AvgAnnWages.index))

sns.displot(data=df_AvgAnnWages, x="VALUE", kind="hist", bins = 100, aspect = 1.5)
plt.show()

# %% [markdown]
# Average hourly wage from modified sorted dataset. (Mentions above)

# %%
# Average hourly wage                15120
print("\nAverage hourly wage")
df_AvgHrsWages = df_sorted_na.loc[
    (df_sorted_na['Indicators'] == 'Average hourly wage')
]
grouped = df_AvgHrsWages.groupby(['Indicators'])
print(grouped['VALUE'].agg([np.median, np.mean, np.std, np.size]))
print("The total number of this one is ",len(df_AvgHrsWages.index))

sns.displot(data=df_AvgHrsWages, x="VALUE", kind="hist", bins = 100, aspect = 1.5)
plt.show()

# %% [markdown]
# Average weekly hours worked from modified sorted dataset.

# %%
# Average weekly hours worked        15120
print("\nAverage weekly hours worked")
df_AvgWeekHrsWrked = df_sorted_na.loc[
    (df_sorted_na['Indicators'] == 'Average weekly hours worked')
]
grouped = df_AvgWeekHrsWrked.groupby(['Indicators'])
print(grouped['VALUE'].agg([np.median, np.mean, np.std, np.size]))
print("The total number of this one is ",len(df_AvgWeekHrsWrked.index))

sns.displot(data=df_AvgWeekHrsWrked, x="VALUE", kind="hist", bins = 100, aspect = 1.5)
plt.show()

# %% [markdown]
# Hours worked from modified sorted dataset.
# * Notice, Skewed left.

# %%
# Hours worked                       15120
print("\nHours worked")
df_Hrs_Wrked = df_sorted_na.loc[
    (df_sorted_na['Indicators'] == 'Hours worked')
]
grouped = df_Hrs_Wrked.groupby(['Indicators'])
print(grouped['VALUE'].agg([np.median, np.mean, np.std, np.size]))
print(grouped['VALUE'].agg([np.amin, np.amax]))
print("The total number of this one is ",len(df_Hrs_Wrked.index))

sns.displot(data=df_Hrs_Wrked, x="VALUE", kind="hist", bins = 100, aspect = 1.5)
plt.show()

# %% [markdown]
# Number of jobs from modified sorted dataset.
# * Notice, skewed left.

# %%
# Number of jobs                     15120
print("\nNumber of jobs")
df_NumOfJob = df_sorted_na.loc[
    (df_sorted_na['Indicators'] == 'Number of jobs')
]
grouped = df_NumOfJob.groupby(['Indicators'])
print(grouped['VALUE'].agg([np.median, np.mean, np.std, np.size]))
print(grouped['VALUE'].agg([np.amin, np.amax]))
print("The total number of this one is ",len(df_NumOfJob.index))

sns.displot(data=df_NumOfJob, x="VALUE", kind="hist", bins = 100, aspect = 1.5)
plt.show()

# %% [markdown]
# Wages and salaries from modified sorted dataset.
# 
# * Noticed skewed left.

# %%
# Wages and salaries                 15120
print("\nWages and salaries")
df_WagesAndSalaries = df_sorted_na.loc[
    (df_sorted_na['Indicators'] == 'Wages and salaries')
]
grouped = df_WagesAndSalaries.groupby(['Indicators'])
print(grouped['VALUE'].agg([np.median, np.mean, np.std, np.size]))
print(grouped['VALUE'].agg([np.amin, np.amax]))
print("The total number of this one is ",len(df_WagesAndSalaries.index))

sns.displot(data=df_WagesAndSalaries, x="VALUE", kind="hist", bins = 100, aspect = 1.5)
plt.show()

# %% [markdown]
# Next step as analysis, I use "Fitter" to analysis the best fit for the values. The values are distributed correctly. However, it is not normally distributed.

# %% [markdown]
# Best distribution for "Average annual hours worked"

# %%
# Not noramlly distirbuted, skewed toward right

# https://medium.com/the-researchers-guide/finding-the-best-distribution-that-fits-your-data-using-pythons-fitter-library-319a5a0972e9
# https://www.datacamp.com/tutorial/probability-distributions-python
# https://realpython.com/python-histograms/
# https://seaborn.pydata.org/tutorial/introduction.html
# https://aminazahid45.medium.com/seaborn-in-python-76f44752a7c8
# https://stackoverflow.com/questions/26597116/seaborn-plots-not-showing-up
# https://www.analyticsvidhya.com/blog/2021/02/statistics-101-beginners-guide-to-continuous-probability-distributions/

fa = Fitter(df_AvgAnnHrsWrk["VALUE"].values,
           distributions=['gamma',
                          'lognorm',
                          "beta",
                          "burr",
                          "norm"])
fa.fit()
fa.summary()
fa.get_best(method = 'sumsquare_error')

# %% [markdown]
# Best distribution for "Average annual wages and salaries"

# %%
fa = Fitter(df_AvgAnnWages["VALUE"].values,
           distributions=['gamma',
                          'lognorm',
                          "beta",
                          "burr",
                          "norm"])
fa.fit()
fa.summary()
fa.get_best(method = 'sumsquare_error')

# %% [markdown]
# Best distribution for "Average hourly wage"

# %%
fa = Fitter(df_AvgHrsWages["VALUE"].values,
           distributions=['gamma',
                          'lognorm',
                          "beta",
                          "burr",
                          "norm"])
fa.fit()
fa.summary()
fa.get_best(method = 'sumsquare_error')

# %% [markdown]
# Best distribution for "Average weekly hours worked"

# %%
fa = Fitter(df_AvgWeekHrsWrked["VALUE"].values,
           distributions=['gamma',
                          'lognorm',
                          "beta",
                          "burr",
                          "norm"])
fa.fit()
fa.summary()
fa.get_best(method = 'sumsquare_error')

# %% [markdown]
# Best distribution for "Hours Worked"

# %%
fa = Fitter(df_Hrs_Wrked["VALUE"].values,
           distributions=['gamma',
                          'lognorm',
                          "beta",
                          "burr",
                          "norm"])
fa.fit()
fa.summary()
fa.get_best(method = 'sumsquare_error')

# %% [markdown]
# Best distrubution for "Number of jobs"

# %%
fa = Fitter(df_NumOfJob["VALUE"].values,
           distributions=['gamma',
                          'lognorm',
                          "beta",
                          "burr",
                          "norm"])
fa.fit()
fa.summary()
fa.get_best(method = 'sumsquare_error')

# %% [markdown]
# Best distribution for "Wages and Salaries"

# %%
fa = Fitter(df_WagesAndSalaries["VALUE"].values,
           distributions=['gamma',
                          'lognorm',
                          "beta",
                          "burr",
                          "norm"])
fa.fit()
fa.summary()
fa.get_best(method = 'sumsquare_error')



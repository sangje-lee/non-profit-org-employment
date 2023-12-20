# %% [markdown]
# ## Data analysis for employment in non-profit organization

# %% [markdown]
# Summarized of this process, <br />
# 1. Import CSV file contain the dataset <br />
# 2. Remove all the Null (NA) dataset from the original dataset. <br />
# 3. Divide by seven indicators. <br />
# 4. Measure the best fit of each indicators. <br />
# 5. Drop (2010-2012) dataset and split (2013-2015), (2016-2018), and (2019-20021). <br />
# 6. Divide by training (2013-2018) and testing (2019-2021) dataset. (Only testing set will be used) <br />
# 7. Divide by characteristic types (based on age, gender, education, and immigrant). <br />
# 8. Other characteristic types will be dropped. <br />
# 9. Divide dataset by provinces but only select five provinces and merge into previous divided dataset.

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

# %% [markdown]
# <h3> Part 1 - Import CSV </h3>

# %% [markdown]
# Import unemployment dataset.

# %%
df = pd.read_csv('36100651.csv')

print(df.info())
print(df.head(10))

# %% [markdown]
# <h3> Part 2 - Filter NA </h3>

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
# <h3> Panda profiling from the datasets</h3>

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

# %% [markdown]
# <h3> This part of code will be used for export into CSV files to demonstrate output in splited portional of code. </h3>

# %% [markdown]
# The code below will create directory to organized the result and structured of the file output from this script.

# %%
# I created the class method to prevent access the whole system.
# To use this one,
# CreatedTheFile = toOrganizedOutputFiles('name')

# Code done by here, https://www.geeksforgeeks.org/create-a-directory-in-python/
# Example 1

class toOrganizedOutputFiles: # Creating folder for each section
  def __init__(self, name):

    import os

    # Leaf directory  
    directory = name
        
    # Parent Directories  
    parent_dir = ""
        
    # Path  
    path = os.path.join(parent_dir, directory)  

    if os.path.isdir(path):
      print("Directory '% s' is ALREADY created" % directory)  
    else:
      # Create the directory  
      os.makedirs(path)  
      print("Directory '% s' created" % directory)  
    


# %% [markdown]
# For clarify, there will be a new directory that stored the result in file based on Indicators columns.

# %%
CreatedTheFile = toOrganizedOutputFiles('Result_By_Indicators')

# %% [markdown]
# <h3> Part 3 - Divide datasets by 'Indicators' </h3>

# %% [markdown]
# Next step, I will filtered the dataset by all the 'Indicators' given below. All of them done with modified sorted dataset (filtered missing value)
# * Notice there will be seven indicators data inside.
# * Notice there will be divided by seven datasets based on indicators.

# %%
# All columns
print(df_sorted_na.info())

# All indicators
grouped = df_sorted_na.groupby(['Indicators'])
print(grouped['VALUE'].agg([np.size]))

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
# Panda Profiling only for "Average annual hours worked"

# %%
pp = ProfileReport(df_AvgAnnHrsWrk, title="Average annual hours worked")
pp_df = pp.to_html()

f = open("Result_By_Indicators/Average annual hours worked.html", "a")  # Expert into html file without modifying any columns in dataset.
f.write(pp_df)
f.close()

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
# Panda Profiling only for "Average annual wages and salaries"

# %%
pp = ProfileReport(df_AvgAnnWages, title="Average annual wages and salaries")
pp_df = pp.to_html()

f = open("Result_By_Indicators/Average annual wages and salaries.html", "a")  # Expert into html file without modifying any columns in dataset.
f.write(pp_df)
f.close()

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
# Panda Profiling only for "Average hourly wages"

# %%
pp = ProfileReport(df_AvgHrsWages, title="Average hourly wage")
pp_df = pp.to_html()

f = open("Result_By_Indicators/Average hourly wages.html", "a")  # Expert into html file without modifying any columns in dataset.
f.write(pp_df)
f.close()

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
# Panda Profiling only for "Average weekly hours worked"

# %%
pp = ProfileReport(df_AvgWeekHrsWrked, title="Average weekly hours worked")
pp_df = pp.to_html()

f = open("Result_By_Indicators/Average weekly hours worked.html", "a")  # Expert into html file without modifying any columns in dataset.
f.write(pp_df)
f.close()

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
# Panda Profiling only for "Hours worked" (Skewed left, noticed)

# %%
pp = ProfileReport(df_Hrs_Wrked, title="Hours Worked")
pp_df = pp.to_html()

f = open("Result_By_Indicators/Hours worked.html", "a")  # Expert into html file without modifying any columns in dataset.
f.write(pp_df)
f.close()

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
# Panda Profiling only for "Number of the jobs" (Stewed toward left)

# %%
pp = ProfileReport(df_NumOfJob, title="Number of jobs")
pp_df = pp.to_html()

f = open("Result_By_Indicators/Number of jobs.html", "a")  # Expert into html file without modifying any columns in dataset.
f.write(pp_df)
f.close()

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
# Panda Profiling only for "Wages and salaries" (Strewed toward left)

# %%
pp = ProfileReport(df_WagesAndSalaries, title="Wages and Salaries")
pp_df = pp.to_html()

f = open("Result_By_Indicators/Wages and salaries.html", "a")  # Expert into html file without modifying any columns in dataset.
f.write(pp_df)
f.close()

# %% [markdown]
# <h3> Part 4 - Best fit for each 'Indicators' </h3>

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

# %% [markdown]
# Before, I go more further in the exercise, I will going to export the process that I have finished. This csv file can be used to regenerate this components of the code again.

# %%
# Save the dataframe to a CSV file
df_sorted_na.to_csv('Result_By_Indicators/df_byIndicator.csv', index=False)


# %% [markdown]
# <h3> Cohort Analysis, 'commented' </h3>

# %% [markdown]
# <b> Cohort Analaysis work is performed both Excel and Python <br />
# However, output result is not showing the result what I am expected <br />
# Therefore, they are commented. However, the excel work is posted in Github </b>

# %% [markdown]
# Cohert Analysis before modifying the whole dataset.<br />
# Excel work is done in separate files.

# %%
# # Excel work is done via this Youtube.
# # https://www.youtube.com/watch?v=dEhlop5ekYM&t=11s

# data_galaxy = df_AvgAnnHrsWrk.copy()
# data_galaxy['REF_DATE'] = data_galaxy['REF_DATE'].astype(str)
# # data_galaxy['REF_DATE'] = pd.to_datetime(data_galaxy["REF_DATE"])

# data_galaxy['FAKE_DATE'] = df.loc[:, 'REF_DATE']
# data_galaxy['FAKE_DATE'] = data_galaxy['FAKE_DATE'].astype(str)

# data_galaxy = data_galaxy.sort_values('FAKE_DATE')

# print(data_galaxy['FAKE_DATE'].unique())
# print(data_galaxy.info())

# # Create fake date as following.
# data_galaxy['FAKE_DATE'] = data_galaxy['FAKE_DATE'].str.replace('2010', '01-01-2010')
# data_galaxy['FAKE_DATE'] = data_galaxy['FAKE_DATE'].str.replace('2011', '02-01-2010')
# data_galaxy['FAKE_DATE'] = data_galaxy['FAKE_DATE'].str.replace('2012', '03-01-2010')
# data_galaxy['FAKE_DATE'] = data_galaxy['FAKE_DATE'].str.replace('2013', '04-01-2010')
# data_galaxy['FAKE_DATE'] = data_galaxy['FAKE_DATE'].str.replace('2014', '05-01-2010')
# data_galaxy['FAKE_DATE'] = data_galaxy['FAKE_DATE'].str.replace('2015', '06-01-2010')
# data_galaxy['FAKE_DATE'] = data_galaxy['FAKE_DATE'].str.replace('2016', '07-01-2010')
# data_galaxy['FAKE_DATE'] = data_galaxy['FAKE_DATE'].str.replace('2017', '08-01-2010')
# data_galaxy['FAKE_DATE'] = data_galaxy['FAKE_DATE'].str.replace('2018', '09-01-2010')
# data_galaxy['FAKE_DATE'] = data_galaxy['FAKE_DATE'].str.replace('2019', '10-01-2010')
# data_galaxy['FAKE_DATE'] = data_galaxy['FAKE_DATE'].str.replace('2020', '11-01-2010')
# data_galaxy['FAKE_DATE'] = data_galaxy['FAKE_DATE'].str.replace('2021', '12-01-2010')

# print(data_galaxy.info())

# data_galaxy['REF_DATE'] = pd.to_datetime(data_galaxy["REF_DATE"])
# data_galaxy['FAKE_DATE'] = pd.to_datetime(data_galaxy["FAKE_DATE"])

# # Export into CSV file, before Cohort Analysis begins.
# # data_galaxy.to_csv('df_AvgAnnHrsWrk_test.csv', index=False)

# %%
# # https://saturncloud.io/blog/converting-a-column-to-date-format-in-pandas-dataframe/
# # https://sparkbyexamples.com/pandas/pandas-extract-year-from-datetime/
# # https://saturncloud.io/blog/converting-a-column-to-date-format-in-pandas-dataframe/

# # https://www.askpython.com/python/examples/cohort-analysis
# # https://www.activestate.com/blog/cohort-analysis-with-python/


# # Make a copy
# data_galaxy = df_AvgAnnHrsWrk .copy()

        
# data_galaxy['REF_DATE'] = data_galaxy['REF_DATE'].astype(str)
# data_galaxy['REF_DATE'] = pd.to_datetime(data_galaxy["REF_DATE"])

# data_galaxy = data_galaxy.sort_values('REF_DATE')

# data_galaxy = data_galaxy.sort_values('FAKE_DATE')

# def getting_months(m):
#     return dt.datetime(m.year, m.month,1)

# #function for data to create a series
# def get_date_elements(df, column):
#     day = df[column].dt.day
#     month = df[column].dt.month
#     year = df[column].dt.year
#     return day, month, year 

# # using the above function
# data_galaxy['Invoice-Month'] = data_galaxy['FAKE_DATE'].apply(getting_months) # data_galaxy['REF_DATE'].apply(getting_months)
# # self.data_galaxy['Invoice-Month'] = self.data_galaxy['Invoice-Month']

# # indexing a column for the first month visit of the customer
# data_galaxy['Cohort-Month'] = data_galaxy.groupby('DGUID')['Invoice-Month'].transform('min')
# # data_galaxy['Cohort-Month'] = data_galaxy['Cohort-Month'].dt.to_period('Y')
# data_galaxy.head(30)

# # getting date for columns and invoice
# _,Invoiceofmonth,Invoiceofyear = get_date_elements(data_galaxy,'Invoice-Month')
# _,Cohortofmonth,Cohortofyear =  get_date_elements(data_galaxy,'Cohort-Month')

# # cohort index creation
# yeardifference = Invoiceofyear -Cohortofyear
# monthdifference = Invoiceofmonth - Cohortofmonth
# data_galaxy['Cohort-Index'] = yeardifference*12 +monthdifference+1

# # counting customer ID 
# cohort_data = data_galaxy.groupby(['Cohort-Month','Cohort-Index'])['DGUID'].apply(pd.Series.nunique).reset_index()
        
# # pivot table creation
# cohort_table = cohort_data.pivot(index='Cohort-Month', columns=['Cohort-Index'],values='DGUID')
        
# # changing index of the cohort table
# cohort_table.index = cohort_table.index.strftime('%Y') # ('%B %Y')
        
# # creation of heatmap and visualization
# plt.figure(figsize=(21,10))
# sns.heatmap(cohort_table,annot=True,cmap='Greens')
        
# # cohort for percentage analysis
# new_cohort_table = cohort_table.divide(cohort_table.iloc[:,0],axis=0)
        
# # creating a percentage visualization
# plt.figure(figsize=(21,10))
# colormap=sns.color_palette("mako", as_cmap=True)
# sns.heatmap(new_cohort_table,annot=True,fmt='.0%',cmap=colormap)
# # display the heatmaps.
# plt.show()

# %% [markdown]
# <h3> Step 5 - Division by years </h3>
# Each year is divided by three years, (2010-2012), (2013-2015), (2016-2018), (2019-2021)

# %% [markdown]
# For next step, I will divide each Indicators dataset into four different datasets.<br />
# They are 2010-2012 (dropped), 2013-2015, 2016-2018, 2019-2021.<br />
# I have dataset prepared before 2010-2012. However, it will not be used after this section, there's too much to analysis to do.<br />
# It will also demonstrate here why.

# %%
print("There are seven Indicators to analysis,")
grouped = df_sorted_na.groupby('Indicators')
print(grouped['VALUE'].agg([np.size]))

print("\nThe data inside between 2010-2013, there's are # number of data and I will be repeating this seven more time.,")
df_Avg_Sample = df_AvgAnnHrsWrk.loc[
    (df_AvgAnnHrsWrk['REF_DATE'] == 2010) |
    (df_AvgAnnHrsWrk['REF_DATE'] == 2011) |
    (df_AvgAnnHrsWrk['REF_DATE'] == 2012)
]

grouped = df_Avg_Sample.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.size]))

print("\nTo data inside above 2013 and split into three datasets, I need to repeat this analysis for "+str(7*3)+" (7x3) times.")
print("\nThis is also total of spliting into "+str(7*3)+" datasets.")

df_Avg_Sample_2013 = df_AvgAnnHrsWrk.loc[
    (df_AvgAnnHrsWrk['REF_DATE'] == 2013) |
    (df_AvgAnnHrsWrk['REF_DATE'] == 2014) |
    (df_AvgAnnHrsWrk['REF_DATE'] == 2015)
]

df_Avg_Sample_2016 = df_AvgAnnHrsWrk.loc[
    (df_AvgAnnHrsWrk['REF_DATE'] == 2016) |
    (df_AvgAnnHrsWrk['REF_DATE'] == 2017) |
    (df_AvgAnnHrsWrk['REF_DATE'] == 2018)
]

df_Avg_Sample_2019 = df_AvgAnnHrsWrk.loc[
    (df_AvgAnnHrsWrk['REF_DATE'] == 2019) |
    (df_AvgAnnHrsWrk['REF_DATE'] == 2020) |
    (df_AvgAnnHrsWrk['REF_DATE'] == 2021)
]

grouped = df_Avg_Sample_2013.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.size]))

grouped = df_Avg_Sample_2016.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.size]))

grouped = df_Avg_Sample_2019.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.size]))

# %% [markdown]
# Grabbing the year (REF_DATE) from 2010, 2013, 2016, 2018, and 2019 individually for "Average annual hours worked".

# %%
# 2010-2012
df_AvgAnnHrsWrk_2010 = df_AvgAnnHrsWrk.loc[
    (df_AvgAnnHrsWrk['REF_DATE'] == 2010) |
    (df_AvgAnnHrsWrk['REF_DATE'] == 2011) |
    (df_AvgAnnHrsWrk['REF_DATE'] == 2012)
]

grouped = df_AvgAnnHrsWrk_2010.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

# %%
print("Grabbing the data from 2013, 2016, and 2019.")

# 2013 - 2015
df_AvgAnnHrsWrk_2013 = df_AvgAnnHrsWrk.loc[
    (df_AvgAnnHrsWrk['REF_DATE'] == 2013) |
    (df_AvgAnnHrsWrk['REF_DATE'] == 2014) |
    (df_AvgAnnHrsWrk['REF_DATE'] == 2015)
]

# 2016 - 2018
df_AvgAnnHrsWrk_2016 = df_AvgAnnHrsWrk.loc[
    (df_AvgAnnHrsWrk['REF_DATE'] == 2016) |
    (df_AvgAnnHrsWrk['REF_DATE'] == 2017) |
    (df_AvgAnnHrsWrk['REF_DATE'] == 2018)
]

# 2019- 2021
df_AvgAnnHrsWrk_2019 = df_AvgAnnHrsWrk.loc[
    (df_AvgAnnHrsWrk['REF_DATE'] == 2019) |
    (df_AvgAnnHrsWrk['REF_DATE'] == 2020) |
    (df_AvgAnnHrsWrk['REF_DATE'] == 2021)
]

grouped = df_AvgAnnHrsWrk_2013.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

grouped = df_AvgAnnHrsWrk_2016.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

grouped = df_AvgAnnHrsWrk_2019.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

# %% [markdown]
# Panda Profiling for year 2016, 2018, and 2020 for "Average annual hours worked".

# %%
# # 2016-2017
# pp = ProfileReport(df_AvgAnnHrsWrk_2013, title="Average annual hours worked 2013")
# pp_df = pp.to_html()

# f = open("Average annual hours worked 2016.html", "a")  # Expert into html file without modifying any columns in dataset.
# f.write(pp_df)
# f.close()

# # 2017 - 2019
# pp = ProfileReport(df_AvgAnnHrsWrk_2016, title="Average annual hours worked 2016")
# pp_df = pp.to_html()

# f = open("Average annual hours worked 2018.html", "a")  # Expert into html file without modifying any columns in dataset.
# f.write(pp_df)
# f.close()

# # 2020 - 2021
# pp = ProfileReport(df_AvgAnnHrsWrk_2019, title="Average annual hours worked 2019")
# pp_df = pp.to_html()

# f = open("Average annual hours worked 2020.html", "a")  # Expert into html file without modifying any columns in dataset.
# f.write(pp_df)
# f.close()

# %% [markdown]
# Grabbing the year (REF_DATE) from 2010, 2013, 2016, and 2019 individually for "Average annual wages and salaries".

# %%
# 2010 - 2012
df_AvgAnnWages_2010 = df_AvgAnnWages.loc[
    (df_AvgAnnWages['REF_DATE'] == 2010) |
    (df_AvgAnnWages['REF_DATE'] == 2011) |
    (df_AvgAnnWages['REF_DATE'] == 2012)
]

grouped = df_AvgAnnWages_2010.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

# %%
print("Grabbing the data from 2017, 2019, and 2021.")

# 2013 - 2015
df_AvgAnnWages_2013 = df_AvgAnnWages.loc[
    (df_AvgAnnWages['REF_DATE'] == 2013) |
    (df_AvgAnnWages['REF_DATE'] == 2014) |
    (df_AvgAnnWages['REF_DATE'] == 2015)
]

# 2016 - 2018
df_AvgAnnWages_2016 = df_AvgAnnWages.loc[
    (df_AvgAnnWages['REF_DATE'] == 2016) |
    (df_AvgAnnWages['REF_DATE'] == 2017) |
    (df_AvgAnnWages['REF_DATE'] == 2018)
]

# 2019 - 2021
df_AvgAnnWages_2019 = df_AvgAnnWages.loc[
    (df_AvgAnnWages['REF_DATE'] == 2019) |
    (df_AvgAnnWages['REF_DATE'] == 2020) |
    (df_AvgAnnWages['REF_DATE'] == 2021)
]

grouped = df_AvgAnnWages_2013.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

grouped = df_AvgAnnWages_2016.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

grouped = df_AvgAnnWages_2019.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

# %% [markdown]
# Panda Profiling for year 2013, 2016, and 2019 for "Average annual wages and salaries".

# %%
# pp = ProfileReport(df_AvgAnnWages_2013, title="Average annual wages and salaries 2013")
# pp_df = pp.to_html()

# f = open("Average annual wages and salaries 2013.html", "a")  # Expert into html file without modifying any columns in dataset.
# f.write(pp_df)
# f.close()

# pp = ProfileReport(df_AvgAnnWages_2016, title="Average annual wages and salaries 2016")
# pp_df = pp.to_html()

# f = open("Average annual wages and salaries 2016.html", "a")  # Expert into html file without modifying any columns in dataset.
# f.write(pp_df)
# f.close()

# pp = ProfileReport(df_AvgAnnWages_2019, title="Average annual wages and salaries 2019")
# pp_df = pp.to_html()

# f = open("Average annual wages and salaries 2019.html", "a")  # Expert into html file without modifying any columns in dataset.
# f.write(pp_df)
# f.close()

# %% [markdown]
# Grabbing the year (REF_DATE) from 2010, 2013, 2016, and 2019 individually for "Average hourly wages".

# %%
# 2010 - 2012
df_AvgHrsWages_2010 = df_AvgHrsWages.loc[
    (df_AvgHrsWages['REF_DATE'] == 2010) |
    (df_AvgHrsWages['REF_DATE'] == 2011) |
    (df_AvgHrsWages['REF_DATE'] == 2012)
]

grouped = df_AvgHrsWages_2010.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

# %%
print("Grabbing the data from 2013, 2016, and 2019.")

# 2013 - 2015
df_AvgHrsWages_2013 = df_AvgHrsWages.loc[
    (df_AvgHrsWages['REF_DATE'] == 2013) |
    (df_AvgHrsWages['REF_DATE'] == 2014) |
    (df_AvgHrsWages['REF_DATE'] == 2015)
]

# 2016 - 2018
df_AvgHrsWages_2016 = df_AvgHrsWages.loc[
    (df_AvgHrsWages['REF_DATE'] == 2016) |
    (df_AvgHrsWages['REF_DATE'] == 2017) |
    (df_AvgHrsWages['REF_DATE'] == 2018)
]

# 2019 - 2021
df_AvgHrsWages_2019 = df_AvgHrsWages.loc[
    (df_AvgHrsWages['REF_DATE'] == 2019) |
    (df_AvgHrsWages['REF_DATE'] == 2020) |
    (df_AvgHrsWages['REF_DATE'] == 2021)
]

grouped = df_AvgHrsWages_2013.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

grouped = df_AvgHrsWages_2016.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

grouped = df_AvgHrsWages_2019.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

# %% [markdown]
# Panda Profiling for year 2013, 2016, and 2019 for "Average hourly wages".

# %%
# pp = ProfileReport(df_AvgHrsWages_2013, title="Average hourly wage 2013")
# pp_df = pp.to_html()

# f = open("Average hourly wages 2013.html", "a")  # Expert into html file without modifying any columns in dataset.
# f.write(pp_df)
# f.close()

# pp = ProfileReport(df_AvgHrsWages_2016, title="Average hourly wage 2016")
# pp_df = pp.to_html()

# f = open("Average hourly wages 2016.html", "a")  # Expert into html file without modifying any columns in dataset.
# f.write(pp_df)
# f.close()

# pp = ProfileReport(df_AvgHrsWages_2019, title="Average hourly wage 2019")
# pp_df = pp.to_html()

# f = open("Average hourly wages 2019.html", "a")  # Expert into html file without modifying any columns in dataset.
# f.write(pp_df)
# f.close()

# %% [markdown]
# Grabbing the year (REF_DATE) from 2010, 2013, 2016, and 2019 individually for "Average weekly hours worked".

# %%
# 2010 - 2012
df_AvgWeekHrsWrked_2010 = df_AvgWeekHrsWrked.loc[
    (df_AvgWeekHrsWrked['REF_DATE'] == 2010) |
    (df_AvgWeekHrsWrked['REF_DATE'] == 2011) |
    (df_AvgWeekHrsWrked['REF_DATE'] == 2012)
]

grouped = df_AvgWeekHrsWrked_2010.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

# %%
print("Grabbing the data from 2013, 2016, and 2019.")

# 2013 - 2015
df_AvgWeekHrsWrked_2013 = df_AvgWeekHrsWrked.loc[
    (df_AvgWeekHrsWrked['REF_DATE'] == 2013) |
    (df_AvgWeekHrsWrked['REF_DATE'] == 2014) |
    (df_AvgWeekHrsWrked['REF_DATE'] == 2015)
]

# 2016 - 2018
df_AvgWeekHrsWrked_2016 = df_AvgWeekHrsWrked.loc[
    (df_AvgWeekHrsWrked['REF_DATE'] == 2016) |
    (df_AvgWeekHrsWrked['REF_DATE'] == 2017) |
    (df_AvgWeekHrsWrked['REF_DATE'] == 2018)
]

# 2019 - 2021
df_AvgWeekHrsWrked_2019 = df_AvgWeekHrsWrked.loc[
    (df_AvgWeekHrsWrked['REF_DATE'] == 2019) |
    (df_AvgWeekHrsWrked['REF_DATE'] == 2020) |
    (df_AvgWeekHrsWrked['REF_DATE'] == 2021)
]

grouped = df_AvgWeekHrsWrked_2013.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

grouped = df_AvgWeekHrsWrked_2016.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

grouped = df_AvgWeekHrsWrked_2019.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

# %% [markdown]
# Panda Profiling for year 2013, 2016, and 2019 for "Average weekly hours worked".

# %%
# pp = ProfileReport(df_AvgWeekHrsWrked_2013, title="Average weekly hours worked 2013")
# pp_df = pp.to_html()

# f = open("Average weekly hours worked 2013.html", "a")  # Expert into html file without modifying any columns in dataset.
# f.write(pp_df)
# f.close()

# pp = ProfileReport(df_AvgWeekHrsWrked_2016, title="Average weekly hours worked 2016")
# pp_df = pp.to_html()

# f = open("Average weekly hours worked 2016.html", "a")  # Expert into html file without modifying any columns in dataset.
# f.write(pp_df)
# f.close()

# pp = ProfileReport(df_AvgWeekHrsWrked_2019, title="Average weekly hours worked 2019")
# pp_df = pp.to_html()

# f = open("Average weekly hours worked 2019.html", "a")  # Expert into html file without modifying any columns in dataset.
# f.write(pp_df)
# f.close()

# %% [markdown]
# Grabbing the year (REF_DATE) from 2010, 2013, 2016, and 2019 individually for "hours worked".

# %%
# 2010 - 2012
df_Hrs_Wrked_2010 = df_Hrs_Wrked.loc[
    (df_Hrs_Wrked['REF_DATE'] == 2010) |
    (df_Hrs_Wrked['REF_DATE'] == 2011) |
    (df_Hrs_Wrked['REF_DATE'] == 2012)
]

grouped = df_Hrs_Wrked_2010.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

# %%
print("Grabbing the data from 2013, 2016, and 2019.")

# 2013 - 2015
df_Hrs_Wrked_2013 = df_Hrs_Wrked.loc[
    (df_Hrs_Wrked['REF_DATE'] == 2013) |
    (df_Hrs_Wrked['REF_DATE'] == 2014) |
    (df_Hrs_Wrked['REF_DATE'] == 2015)
]

# 2016 - 2018
df_Hrs_Wrked_2016 = df_Hrs_Wrked.loc[
    (df_Hrs_Wrked['REF_DATE'] == 2016) |
    (df_Hrs_Wrked['REF_DATE'] == 2017) |
    (df_Hrs_Wrked['REF_DATE'] == 2018)
]

# 2019 - 2021
df_Hrs_Wrked_2019 = df_Hrs_Wrked.loc[
    (df_Hrs_Wrked['REF_DATE'] == 2019) |
    (df_Hrs_Wrked['REF_DATE'] == 2020) |
    (df_Hrs_Wrked['REF_DATE'] == 2021)
]

grouped = df_Hrs_Wrked_2013.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

grouped = df_Hrs_Wrked_2016.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

grouped = df_Hrs_Wrked_2019.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

# %% [markdown]
# Panda Profiling for year 2013, 2016, and 2019 for "hours worked".

# %%
# pp = ProfileReport(df_Hrs_Wrked_2013, title="Hours Worked 2013")
# pp_df = pp.to_html()

# f = open("Hours worked 2013.html", "a")  # Expert into html file without modifying any columns in dataset.
# f.write(pp_df)
# f.close()

# pp = ProfileReport(df_Hrs_Wrked_2016, title="Hours Worked 2016")
# pp_df = pp.to_html()

# f = open("Hours worked 2016.html", "a")  # Expert into html file without modifying any columns in dataset.
# f.write(pp_df)
# f.close()

# pp = ProfileReport(df_Hrs_Wrked_2019, title="Hours Worked 2019")
# pp_df = pp.to_html()

# f = open("Hours worked 2019.html", "a")  # Expert into html file without modifying any columns in dataset.
# f.write(pp_df)
# f.close()

# %% [markdown]
# Grabbing the year (REF_DATE) from 2010, 2013, 2016, and 2019 individually for "Number of jobs".

# %%
# 2010 - 2012
df_NumOfJob_2010 = df_NumOfJob.loc[
    (df_NumOfJob['REF_DATE'] == 2010) |
    (df_NumOfJob['REF_DATE'] == 2011) |
    (df_NumOfJob['REF_DATE'] == 2012)
]

grouped = df_NumOfJob_2010.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

# %%
print("Grabbing the data from 2013, 2016, and 2019.")

# 2013 - 2015
df_NumOfJob_2013 = df_NumOfJob.loc[
    (df_NumOfJob['REF_DATE'] == 2013) |
    (df_NumOfJob['REF_DATE'] == 2014) |
    (df_NumOfJob['REF_DATE'] == 2015)
]

# 2016 - 2018
df_NumOfJob_2016 = df_NumOfJob.loc[
    (df_NumOfJob['REF_DATE'] == 2016) |
    (df_NumOfJob['REF_DATE'] == 2017) |
    (df_NumOfJob['REF_DATE'] == 2018)
]

# 2019 - 2021
df_NumOfJob_2019 = df_NumOfJob.loc[
    (df_NumOfJob['REF_DATE'] == 2019) |
    (df_NumOfJob['REF_DATE'] == 2020) |
    (df_NumOfJob['REF_DATE'] == 2021)
]

grouped = df_NumOfJob_2013.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

grouped = df_NumOfJob_2016.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

grouped = df_NumOfJob_2019.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

# %% [markdown]
# Panda Profiling for year 2013, 2016, and 2019for "Number of jobs".

# %%
# pp = ProfileReport(df_NumOfJob_2013, title="Number of jobs 2013")
# pp_df = pp.to_html()

# f = open("Number of jobs 2013.html", "a")  # Expert into html file without modifying any columns in dataset.
# f.write(pp_df)
# f.close()

# pp = ProfileReport(df_NumOfJob_2016, title="Number of jobs 2016")
# pp_df = pp.to_html()

# f = open("Number of jobs 2016.html", "a")  # Expert into html file without modifying any columns in dataset.
# f.write(pp_df)
# f.close()

# pp = ProfileReport(df_NumOfJob_2019, title="Number of jobs 2019")
# pp_df = pp.to_html()

# f = open("Number of jobs 2019.html", "a")  # Expert into html file without modifying any columns in dataset.
# f.write(pp_df)
# f.close()

# %% [markdown]
# Grabbing the year (REF_DATE) from 2010, 2013, 2016, and 2019 individually for "Wages and Salaries".

# %%
# 2010 - 2012
df_WagesAndSalaries_2010 = df_WagesAndSalaries.loc[
    (df_WagesAndSalaries['REF_DATE'] == 2010) |
    (df_WagesAndSalaries['REF_DATE'] == 2011) |
    (df_WagesAndSalaries['REF_DATE'] == 2012)
]

grouped = df_WagesAndSalaries_2010.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

# %%
print("Grabbing the data from 2013, 2016, and 2019.")

# 2013 - 2015
df_WagesAndSalaries_2013 = df_WagesAndSalaries.loc[
    (df_WagesAndSalaries['REF_DATE'] == 2013) |
    (df_WagesAndSalaries['REF_DATE'] == 2014) |
    (df_WagesAndSalaries['REF_DATE'] == 2015)
]

# 2016 - 2018
df_WagesAndSalaries_2016 = df_WagesAndSalaries.loc[
    (df_WagesAndSalaries['REF_DATE'] == 2016) |
    (df_WagesAndSalaries['REF_DATE'] == 2017) |
    (df_WagesAndSalaries['REF_DATE'] == 2018)
]

# 2019 - 2021
df_WagesAndSalaries_2019 = df_WagesAndSalaries.loc[
    (df_WagesAndSalaries['REF_DATE'] == 2019) |
    (df_WagesAndSalaries['REF_DATE'] == 2020) |
    (df_WagesAndSalaries['REF_DATE'] == 2021)
]

grouped = df_WagesAndSalaries_2013.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

grouped = df_WagesAndSalaries_2016.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

grouped = df_WagesAndSalaries_2019.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

# %% [markdown]
# Panda Profiling for year 2013, 2016, and 2019 for "Wages and Salaries".

# %%
# pp = ProfileReport(df_WagesAndSalaries_2013, title="Wages and Salaries 2013")
# pp_df = pp.to_html()

# f = open("Wages and Salaries 2013.html", "a")  # Expert into html file without modifying any columns in dataset.
# f.write(pp_df)
# f.close()

# pp = ProfileReport(df_WagesAndSalaries_2016, title="Wages and Salaries 2016")
# pp_df = pp.to_html()

# f = open("Wages and Salaries 2016.html", "a")  # Expert into html file without modifying any columns in dataset.
# f.write(pp_df)
# f.close()

# pp = ProfileReport(df_WagesAndSalaries_2019, title="Wages and Salaries 2019")
# pp_df = pp.to_html()

# f = open("Wages and Salaries 2019.html", "a")  # Expert into html file without modifying any columns in dataset.
# f.write(pp_df)
# f.close()

# %% [markdown]
# For clarify, there will be a new directory that stored the result in file based on Training/Testing dataset.
# 

# %%
CreatedTheFile = toOrganizedOutputFiles('Result_By_Testing_Training')

# %% [markdown]
# <h3> Part 6 - Division between training and testing dataset </h3>
# Training is about 60-65% and Testing is about 40-45% of dataset.<br />
# Training (2013-2018), Testing (2019-2021).

# %% [markdown]
# Training dataset going to be after 2013 to before 2018.<br />
# Testing dataset going to be after 2019.<br />
# Analysis is still finishing year of 2013 to 2018 though.<br />
# I have decided not to use train_test_split method because I have divided dataset by the year.<br />
# Instead I have divide it manually.

# %%
# https://machinelearningmastery.com/train-test-split-for-evaluating-machine-learning-algorithms/

# This will be used if I were to use train_test_split given.

# from sklearn.model_selection import train_test_split

# train, test = train_test_split(dataset, ...)
# x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=0)    

# %%
# Average annual hours worked
# Use 2013-2015, 2016-2018 as training set
# Use 2019-2021 as testing set

frames = [df_AvgAnnHrsWrk_2013, df_AvgAnnHrsWrk_2016]
training_df_AvgAnnHrsWrk = pd.concat(frames)
testing_df_AvgAnnHrsWrk = df_AvgAnnHrsWrk_2019.copy()

grouped = training_df_AvgAnnHrsWrk.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

grouped = testing_df_AvgAnnHrsWrk.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

# %%
# pp = ProfileReport(training_df_AvgAnnHrsWrk, title="Average annual hours worked Training Datset")
# pp_df = pp.to_html()

# f = open("Result_By_Testing_Training/Average annual hours worked Training Dataaset.html", "a")  # Expert into html file without modifying any columns in dataset.
# f.write(pp_df)
# f.close()

# pp = ProfileReport(testing_df_AvgAnnHrsWrk, title="Average annual hours worked Testing Dataset")
# pp_df = pp.to_html()

# f = open("Result_By_Testing_Training/Average annual hours worked Testing Dataset.html", "a")  # Expert into html file without modifying any columns in dataset.
# f.write(pp_df)
# f.close()

# %%
# Average annual wages and salaries

frames = [df_AvgAnnWages_2013, df_AvgAnnWages_2016]
training_df_AvgAnnWages = pd.concat(frames)
testing_df_AvgAnnWages = df_AvgAnnWages_2019.copy()

grouped = training_df_AvgAnnWages.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

grouped = testing_df_AvgAnnWages.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

# %%
# pp = ProfileReport(training_df_AvgAnnHrsWrk, title="Average annual wages and salaries Training Datset")
# pp_df = pp.to_html()

# f = open("Result_By_Testing_Training/Average annual wages and salaries Training Dataaset.html", "a")  # Expert into html file without modifying any columns in dataset.
# f.write(pp_df)
# f.close()

# pp = ProfileReport(testing_df_AvgAnnHrsWrk, title="Average annual wages and salaries Testing Dataset")
# pp_df = pp.to_html()

# f = open("Result_By_Testing_Training/Average annual wages and salaries Testing Dataset.html", "a")  # Expert into html file without modifying any columns in dataset.
# f.write(pp_df)
# f.close()

# %%
# Average hourly wage

frames = [df_AvgHrsWages_2013, df_AvgHrsWages_2016]
training_df_AvgHrsWages = pd.concat(frames)
testing_df_AvgHrsWages = df_AvgHrsWages_2019.copy()

grouped = training_df_AvgHrsWages.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

grouped = testing_df_AvgHrsWages.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

# %%
# pp = ProfileReport(training_df_AvgAnnHrsWrk, title="Average hourly wage Training Datset")
# pp_df = pp.to_html()

# f = open("Result_By_Testing_Training/Average weekly hours worked Training Dataaset.html", "a")  # Expert into html file without modifying any columns in dataset.
# f.write(pp_df)
# f.close()

# pp = ProfileReport(testing_df_AvgAnnHrsWrk, title="Average hourly wage Testing Dataset")
# pp_df = pp.to_html()

# f = open("Result_By_Testing_Training/Average weekly hours worked Testing Dataset.html", "a")  # Expert into html file without modifying any columns in dataset.
# f.write(pp_df)
# f.close()

# %%
# Average weekly hours worked

frames = [df_AvgWeekHrsWrked_2013, df_AvgWeekHrsWrked_2016]
training_df_AvgWeekHrsWrked = pd.concat(frames)
testing_df_AvgWeekHrsWrked = df_AvgWeekHrsWrked_2019.copy()

grouped = training_df_AvgWeekHrsWrked.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

grouped = testing_df_AvgWeekHrsWrked.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

# %%
# pp = ProfileReport(training_df_AvgAnnHrsWrk, title="Average weekly hours worked Training Datset")
# pp_df = pp.to_html()

# f = open("Result_By_Testing_Training/Average weekly hours worked Training Dataaset.html", "a")  # Expert into html file without modifying any columns in dataset.
# f.write(pp_df)
# f.close()

# pp = ProfileReport(testing_df_AvgAnnHrsWrk, title="Average weekly hours worked Testing Dataset")
# pp_df = pp.to_html()

# f = open("Result_By_Testing_Training/Average weekly hours worked Testing Dataset.html", "a")  # Expert into html file without modifying any columns in dataset.
# f.write(pp_df)
# f.close()

# %%
# Hours Worked

frames = [df_Hrs_Wrked_2013, df_Hrs_Wrked_2016]
training_df_Hrs_Wrked = pd.concat(frames)
testing_df_Hrs_Wrked = df_Hrs_Wrked_2019.copy()

grouped = training_df_Hrs_Wrked.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

grouped = testing_df_Hrs_Wrked.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

# %%
# pp = ProfileReport(training_df_AvgAnnHrsWrk, title="Hours Worked Training Datset")
# pp_df = pp.to_html()

# f = open("Result_By_Testing_Training/Hours Worked Training Dataaset.html", "a")  # Expert into html file without modifying any columns in dataset.
# f.write(pp_df)
# f.close()

# pp = ProfileReport(testing_df_AvgAnnHrsWrk, title="Hours Worked Testing Dataset")
# pp_df = pp.to_html()

# f = open("Result_By_Testing_Training/Hours Worked Testing Dataset.html", "a")  # Expert into html file without modifying any columns in dataset.
# f.write(pp_df)
# f.close()

# %%
# Number of jobs

frames = [df_NumOfJob_2013, df_NumOfJob_2016]
training_df_NumOfJob = pd.concat(frames)
testing_df_NumOfJob = df_NumOfJob_2019.copy()

grouped = training_df_NumOfJob.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

grouped = testing_df_NumOfJob.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

# %%
# pp = ProfileReport(training_df_AvgAnnHrsWrk, title="Number of jobs Training Datset")
# pp_df = pp.to_html()

# f = open("Result_By_Testing_Training/Number of jobs Training Dataaset.html", "a")  # Expert into html file without modifying any columns in dataset.
# f.write(pp_df)
# f.close()

# pp = ProfileReport(testing_df_AvgAnnHrsWrk, title="Number of jobs Testing Dataset")
# pp_df = pp.to_html()

# f = open("Result_By_Testing_Training/Number of jobs Testing Dataset.html", "a")  # Expert into html file without modifying any columns in dataset.
# f.write(pp_df)
# f.close()

# %%
# Wages and Salaries

frames = [df_WagesAndSalaries_2013, df_WagesAndSalaries_2016]
training_df_WagesAndSalaries = pd.concat(frames)
testing_df_WagesAndSalaries = df_WagesAndSalaries_2019.copy()

grouped = training_df_WagesAndSalaries.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

grouped = testing_df_WagesAndSalaries.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

# %%
# pp = ProfileReport(training_df_AvgAnnHrsWrk, title="Wages and Salaries Training Datset")
# pp_df = pp.to_html()

# f = open("Result_By_Testing_Training/Wages and Salaries Training Dataaset.html", "a")  # Expert into html file without modifying any columns in dataset.
# f.write(pp_df)
# f.close()

# pp = ProfileReport(testing_df_AvgAnnHrsWrk, title="Wages and Salaries Testing Dataset")
# pp_df = pp.to_html()

# f = open("Result_By_Testing_Training/Wages and Salaries Testing Dataset.html", "a")  # Expert into html file without modifying any columns in dataset.
# f.write(pp_df)
# f.close()

# %% [markdown]
# Final output for Average annual hours worked

# %%
class Target_To_Analysis:

    def __init__(self, df, pd, np, pp, sns, year):
      self.dfa_Target_To_Analysis = df
      self.year = year
      self.pd = pd
      self.np = np
      self.pp = pp
      self.sns = sns
 
    # create a function
    def print_result(self):
      n = 0
      for df_Target_To_Analysis in self.dfa_Target_To_Analysis:
            grouped = df_Target_To_Analysis.groupby(['Characteristics'])
            print(self.year[n])
            print(grouped['VALUE'].agg([np.sum, np.mean, np.min, np.median, np.max, np.size]))
            print("Overall,")
            print("Sum : ",np.sum(df_Target_To_Analysis['VALUE']))
            print("Mean : ",np.mean(df_Target_To_Analysis['VALUE']))
            print("Min/median/max :",np.min(df_Target_To_Analysis['VALUE']),"/",
                  np.median(df_Target_To_Analysis['VALUE']),"/",
                  np.max(df_Target_To_Analysis['VALUE']))
            print("Standard Deviation : ",np.std(df_Target_To_Analysis['VALUE']))
            print("Skewnewss : ",df_Target_To_Analysis['VALUE'].skew())
            print("Total size : ",len(df_Target_To_Analysis.index))
            print()
            n = n + 1
    
    def print_histogram(self, n):
      sns.displot(data=self.dfa_Target_To_Analysis[int(n)], x="VALUE", kind="hist", bins = 100, aspect = 1.5)
      plt.show()

# %%
dfa_Target_To_Analysis = [training_df_AvgAnnHrsWrk, testing_df_AvgAnnHrsWrk]
dfa_Target_To_Analysis = Target_To_Analysis(dfa_Target_To_Analysis, pd, np, pp, sns, ['training set','testing set'])
dfa_Target_To_Analysis.print_result()

# %%
print("Histogram for training dataset") # testing dataset")
dfa_Target_To_Analysis.print_histogram(0)

print("Histgram for testing dataset")
dfa_Target_To_Analysis.print_histogram(1)

# %% [markdown]
# Final output for "Average annual wages and salaries"

# %%
dfa_Target_To_Analysis = [training_df_AvgAnnWages, testing_df_AvgAnnWages]
dfa_Target_To_Analysis = Target_To_Analysis(dfa_Target_To_Analysis, pd, np, pp, sns, ['training set','testing set'])
dfa_Target_To_Analysis.print_result()

# %%
print("Histogram for training dataset") # testing dataset")
dfa_Target_To_Analysis.print_histogram(0)

print("Histgram for testing dataset")
dfa_Target_To_Analysis.print_histogram(1)

# %% [markdown]
# Final output for "Average hourly wage"

# %%
dfa_Target_To_Analysis = [training_df_AvgHrsWages, testing_df_AvgHrsWages]
dfa_Target_To_Analysis = Target_To_Analysis(dfa_Target_To_Analysis, pd, np, pp, sns, ['training set','testing set'])
dfa_Target_To_Analysis.print_result()

# %%
print("Histogram for training dataset") # testing dataset")
dfa_Target_To_Analysis.print_histogram(0)

print("Histgram for testing dataset")
dfa_Target_To_Analysis.print_histogram(1)

# %% [markdown]
# Final output for "Average weekly hours worked"

# %%
dfa_Target_To_Analysis = [training_df_AvgWeekHrsWrked, testing_df_AvgWeekHrsWrked]
dfa_Target_To_Analysis = Target_To_Analysis(dfa_Target_To_Analysis, pd, np, pp, sns, ['training set','testing set'])
dfa_Target_To_Analysis.print_result()

# %%
print("Histogram for training dataset") # testing dataset")
dfa_Target_To_Analysis.print_histogram(0)

print("Histgram for testing dataset")
dfa_Target_To_Analysis.print_histogram(1)

# %% [markdown]
# Final output for "Hours Worked"

# %%
dfa_Target_To_Analysis = [training_df_Hrs_Wrked, testing_df_Hrs_Wrked]
dfa_Target_To_Analysis = Target_To_Analysis(dfa_Target_To_Analysis, pd, np, pp, sns, ['training set','testing set'])
dfa_Target_To_Analysis.print_result()

# %%
print("Histogram for training dataset") # testing dataset")
dfa_Target_To_Analysis.print_histogram(0)

print("Histgram for testing dataset")
dfa_Target_To_Analysis.print_histogram(1)

# %% [markdown]
# Final output for "Number of jobs"

# %%
dfa_Target_To_Analysis = [training_df_NumOfJob, testing_df_NumOfJob]
dfa_Target_To_Analysis = Target_To_Analysis(dfa_Target_To_Analysis, pd, np, pp, sns, ['training set','testing set'])
dfa_Target_To_Analysis.print_result()

# %%
print("Histogram for training dataset") # testing dataset")
dfa_Target_To_Analysis.print_histogram(0)

print("Histgram for testing dataset")
dfa_Target_To_Analysis.print_histogram(1)

# %% [markdown]
# Final output for "Wages and Salaries"

# %%
dfa_Target_To_Analysis = [training_df_WagesAndSalaries, testing_df_WagesAndSalaries]
dfa_Target_To_Analysis = Target_To_Analysis(dfa_Target_To_Analysis, pd, np, pp, sns, ['training set','testing set'])
dfa_Target_To_Analysis.print_result()

# %%
print("Histogram for training dataset")
dfa_Target_To_Analysis.print_histogram(0)

print("Histgram for testing dataset")
dfa_Target_To_Analysis.print_histogram(1)

# %% [markdown]
# <h3> Cha Squared result </h3>

# %% [markdown]
# All of the Panda datasets Analysis<br />
# All of these data are analysis by chi-square test.<br />
# The data I want to analysis this point are all categorical.<br />
# Two columns that are used are "REF_DATE" which is used to split are "GEO" and "Characteristics".

# %%
# https://www.geeksforgeeks.org/python-pearsons-chi-square-test/
# https://www.geeksforgeeks.org/contingency-table-in-python/
# https://www.analyticsvidhya.com/blog/2020/10/feature-selection-techniques-in-machine-learning/
# https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/


# from scipy.stats import chi2_contingency

class ChiSquareAnalysisByYear:

    def __init__(self, df, classify, pd, np):

        self.data_crosstab = pd.crosstab(df[classify], 
                            df['REF_DATE'], 
                               margins = False) 
        # print(self.data_crosstab) 

    def displayCrosstab(self):
        # Display/OUtput whole corsstab table
        print(self.data_crosstab)

    def returnCrosstab(self):
        # Return whole crosstab table itself
        return self.data_crosstab

    def doChiSquareAnalysis(self):

        # defining the table
        data = self.data_crosstab # [[207, 282, 241], [234, 242, 232]]
        stat, p, dof, expected = chi2_contingency(data)

        # interpret p-value
        alpha = 0.05
        print("p value is " + str(p))
        if p <= alpha:
            print('Dependent (reject H0)')
        else:
            print('Independent (H0 holds true)')

# %%
# Used by Chi-Square methods
# Used by spliting of training and testing dataset
# I did the analysis of training dataset anyway but only will be use testing dataset.
# Commented the analysis divided by year

df_analysis = [training_df_AvgAnnHrsWrk, testing_df_AvgAnnHrsWrk]

# First one is Training set. Second one is Testing set.
for x in df_analysis:
    data_crosstab_char = ChiSquareAnalysisByYear(x,'Characteristics', pd, np)
    data_crosstab_char.displayCrosstab()
    data_crosstab_char.doChiSquareAnalysis()

    data_crosstab_province = ChiSquareAnalysisByYear(x,'GEO', pd, np)
    data_crosstab_province.displayCrosstab()
    data_crosstab_province.doChiSquareAnalysis()

# %%
# Used by Chi-Square methods

df_analysis = [training_df_AvgAnnHrsWrk, testing_df_AvgAnnHrsWrk]

# First one is Training set. Second one is Testing set.
for x in df_analysis:
    data_crosstab_char = ChiSquareAnalysisByYear(x,'Characteristics', pd, np)
    data_crosstab_char.displayCrosstab()
    data_crosstab_char.doChiSquareAnalysis()

    data_crosstab_province = ChiSquareAnalysisByYear(x,'GEO', pd, np)
    data_crosstab_province.displayCrosstab()
    data_crosstab_province.doChiSquareAnalysis()

# %%
df_analysis = [training_df_AvgHrsWages, testing_df_AvgHrsWages]

# First one is Training set. Second one is Testing set.
for x in df_analysis:
    data_crosstab_char = ChiSquareAnalysisByYear(x,'Characteristics', pd, np)
    data_crosstab_char.displayCrosstab()
    data_crosstab_char.doChiSquareAnalysis()

    data_crosstab_province = ChiSquareAnalysisByYear(x,'GEO', pd, np)
    data_crosstab_province.displayCrosstab()
    data_crosstab_province.doChiSquareAnalysis()

# %%
df_analysis = [training_df_AvgWeekHrsWrked, testing_df_AvgWeekHrsWrked]
 
# First one is Training set. Second one is Testing set.
for x in df_analysis:
    data_crosstab_char = ChiSquareAnalysisByYear(x,'Characteristics', pd, np)
    data_crosstab_char.displayCrosstab()
    data_crosstab_char.doChiSquareAnalysis()

    data_crosstab_province = ChiSquareAnalysisByYear(x,'GEO', pd, np)
    data_crosstab_province.displayCrosstab()
    data_crosstab_province.doChiSquareAnalysis()

# %%
df_analysis = [training_df_Hrs_Wrked, testing_df_Hrs_Wrked]

# First one is Training set. Second one is Testing set.    
for x in df_analysis:
    data_crosstab_char = ChiSquareAnalysisByYear(x,'Characteristics', pd, np)
    data_crosstab_char.displayCrosstab()
    data_crosstab_char.doChiSquareAnalysis()

    data_crosstab_province = ChiSquareAnalysisByYear(x,'GEO', pd, np)
    data_crosstab_province.displayCrosstab()
    data_crosstab_province.doChiSquareAnalysis()

# %%
df_analysis = [training_df_NumOfJob, testing_df_NumOfJob]

# First one is Training set. Second one is Testing set. 
for x in df_analysis:
    data_crosstab_char = ChiSquareAnalysisByYear(x,'Characteristics', pd, np)
    data_crosstab_char.displayCrosstab()
    data_crosstab_char.doChiSquareAnalysis()

    data_crosstab_province = ChiSquareAnalysisByYear(x,'GEO', pd, np)
    data_crosstab_province.displayCrosstab()
    data_crosstab_province.doChiSquareAnalysis()

# %%
df_analysis = [training_df_WagesAndSalaries, testing_df_WagesAndSalaries]

# First one is Training set. Second one is Testing set.    
for x in df_analysis:
    data_crosstab_char = ChiSquareAnalysisByYear(x,'Characteristics', pd, np)
    data_crosstab_char.displayCrosstab()
    data_crosstab_char.doChiSquareAnalysis()

    data_crosstab_province = ChiSquareAnalysisByYear(x,'GEO', pd, np)
    data_crosstab_province.displayCrosstab()
    data_crosstab_province.doChiSquareAnalysis()

# %% [markdown]
# Export csv file that are splited between training and testing set.

# %%
# Save the dataframe to a CSV file

# df_AvgAnnHrsWrk # Average annual hours worked
df_AvgAnnHrsWrk_2013.to_csv('Result_By_Testing_Training/df_AvgAnnHrsWrk_2013.csv', index=False) # Average annual hours worked in 2017
df_AvgAnnHrsWrk_2016.to_csv('Result_By_Testing_Training/df_AvgAnnHrsWrk_2016.csv', index=False) #                                2019
df_AvgAnnHrsWrk_2019.to_csv('Result_By_Testing_Training/df_AvgAnnHrsWrk_2019.csv', index=False) #                                2021


# df_AvgAnnWages # Average annual wages and salaries
df_AvgAnnWages_2013.to_csv('Result_By_Testing_Training/df_AvgAnnWages_2013.csv', index=False) # Average annual hours worked in 2017
df_AvgAnnWages_2016.to_csv('Result_By_Testing_Training/df_AvgAnnWages_2016.csv', index=False) #                                2019
df_AvgAnnWages_2019.to_csv('Result_By_Testing_Training/df_AvgAnnWages_2019.csv', index=False) #                                2021

# df_AvgHrsWages # Average hourly wage
df_AvgHrsWages_2013.to_csv('Result_By_Testing_Training/df_AvgHrsWages_2013.csv', index=False) # Average annual hours worked in 2017
df_AvgHrsWages_2016.to_csv('Result_By_Testing_Training/df_AvgHrsWages_2016.csv', index=False) #                                2019
df_AvgHrsWages_2019.to_csv('Result_By_Testing_Training/df_AvgHrsWages_2019.csv', index=False) #                                2021

# df_AvgWeekHrsWrked # Average weekly hours worked
df_AvgWeekHrsWrked_2013.to_csv('Result_By_Testing_Training/df_AvgWeekHrsWrked_2013.csv', index=False) # Average annual hours worked in 2017
df_AvgWeekHrsWrked_2016.to_csv('Result_By_Testing_Training/df_AvgWeekHrsWrked_2016.csv', index=False) #                                2019
df_AvgWeekHrsWrked_2019.to_csv('Result_By_Testing_Training/df_AvgWeekHrsWrked_2019.csv', index=False) #                                2021

# df_Hrs_Wrked # Hours Worked
df_Hrs_Wrked_2013.to_csv('Result_By_Testing_Training/df_Hrs_Wrked_2013.csv', index=False) # Average annual hours worked in 2017
df_Hrs_Wrked_2016.to_csv('Result_By_Testing_Training/df_Hrs_Wrked_2016.csv', index=False) #                                2019
df_Hrs_Wrked_2019.to_csv('Result_By_Testing_Training/df_Hrs_Wrked_2019.csv', index=False) #                                2021

# df_NumOfJob # Number of jobs
df_NumOfJob_2013.to_csv('Result_By_Testing_Training/df_NumOfJob_2013.csv', index=False) # Average annual hours worked in 2017
df_NumOfJob_2016.to_csv('Result_By_Testing_Training/df_NumOfJob_2016.csv', index=False) #                                2019
df_NumOfJob_2019.to_csv('Result_By_Testing_Training/df_NumOfJob_2019.csv', index=False) #                                2021

# df_WagesAndSalaries # Wages and Salaries
df_WagesAndSalaries_2013.to_csv('Result_By_Testing_Training/df_WagesAndSalaries_2013.csv', index=False) # Average annual hours worked in 2017
df_WagesAndSalaries_2016.to_csv('Result_By_Testing_Training/df_WagesAndSalaries_2016.csv', index=False) #                                2019
df_WagesAndSalaries_2019.to_csv('Result_By_Testing_Training/df_WagesAndSalaries_2019.csv', index=False) #                                2021



# %% [markdown]
# For clarify, there will be a new directory that stored the result in file based on Characteristics Indicators.
# 

# %%
CreatedTheFile = toOrganizedOutputFiles('Result_By_Characteristics')

# %%
# https://www.analyticsvidhya.com/blog/2022/01/different-types-of-regression-models/

# https://online.hbs.edu/blog/post/types-of-data-analysis
# https://online.hbs.edu/blog/post/descriptive-analytics
# https://online.hbs.edu/blog/post/diagnostic-analytics
# https://online.hbs.edu/blog/post/predictive-analytics
# https://chartio.com/learn/data-analytics/types-of-data-analysis/
# https://www.simplilearn.com/data-analysis-methods-process-types-article#types_of_data_analysis

# https://builtin.com/data-science/types-of-data-analysis
# https://careerfoundry.com/en/blog/data-analytics/different-types-of-data-analysis/


# %% [markdown]
# <h3> Part 7 - Division by Characteristics </h3>
# Division by 'age', 'gender', 'educaiton' and 'immigrant'

# %% [markdown]
# For next step, I will filtered it by the following group, "age group", "gender level", "education level", "immigrant level" and "Aboriginal status" (commented).
# * I decided not to use 2010-2012 mentions before.
# * I have analysis both training and testing set. (First one is training (2013-2018) and second one is testing, 2019-2021)
# * Originally it was, (2010-2012, dropped), (2013-2015), (2016-2018), (2019-2021)
# * There's also other characteristics there as well but I decided to drop them as well.

# %% [markdown]
# Filtered for "Average annual hours worked" by following: "Age group", "Gender level", "Education level", and "Immigration status".<br />
# "Aboriginal status" has been commented.

# %%
# Dataset by training set inside Average Annual Hours Worked

print("\nAge group")
training_df_AvgAnnHrsWrk_ByAge = training_df_AvgAnnHrsWrk.loc[
    (training_df_AvgAnnHrsWrk['Characteristics'] == '15 to 24 years') |
    (training_df_AvgAnnHrsWrk['Characteristics'] == '25 to 34 years') |
    (training_df_AvgAnnHrsWrk['Characteristics'] == '35 to 44 years') |
    (training_df_AvgAnnHrsWrk['Characteristics'] == '45 to 54 years') |
    (training_df_AvgAnnHrsWrk['Characteristics'] == '55 to 64 years') |
    (training_df_AvgAnnHrsWrk['Characteristics'] == '65 years old and over')]
# print(training_df_AvgAnnHrsWrk_ByAge.head(20))
grouped = training_df_AvgAnnHrsWrk_ByAge.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("Total size : ",len(training_df_AvgAnnHrsWrk_ByAge.index))

print("\nGender group")
training_df_AvgAnnHrsWrk_ByGender = training_df_AvgAnnHrsWrk.loc[
    (training_df_AvgAnnHrsWrk['Characteristics'] == 'Female employees') |
    (training_df_AvgAnnHrsWrk['Characteristics'] == 'Male employees')
]
# print(training_df_AvgAnnHrsWrk_ByGender.head(20))
grouped = training_df_AvgAnnHrsWrk_ByGender.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("Total size : ",len(training_df_AvgAnnHrsWrk_ByGender.index))

print("\nEducation group")
training_df_AvgAnnHrsWrk_ByEducation = training_df_AvgAnnHrsWrk.loc[
    (training_df_AvgAnnHrsWrk['Characteristics'] == 'High school diploma and less') |
    (training_df_AvgAnnHrsWrk['Characteristics'] == 'Trade certificate') |
    (training_df_AvgAnnHrsWrk['Characteristics'] == 'University degree and higher')
]
# print(training_df_AvgAnnHrsWrk_ByEducation.head(20))
grouped = training_df_AvgAnnHrsWrk_ByEducation.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("Total size : ",len(training_df_AvgAnnHrsWrk_ByEducation.index))

print("\nImmigrant group")
training_df_AvgAnnHrsWrk_ByImmigrant = training_df_AvgAnnHrsWrk.loc[
    (training_df_AvgAnnHrsWrk['Characteristics'] == 'Immigrant employees') |
    (training_df_AvgAnnHrsWrk['Characteristics'] == 'Non-immigrant employees')
]
# print(training_df_AvgAnnHrsWrk_ByImmigrant.head(20))
grouped = training_df_AvgAnnHrsWrk_ByImmigrant.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("Total size : ",len(training_df_AvgAnnHrsWrk_ByImmigrant.index))

# print("\nIndigenous group")
# df_AvgAnnHrkWrk_2010_ByIndigenous = training_df_AvgAnnHrsWrk.loc[
#     (training_df_AvgAnnHrsWrk['Characteristics'] == 'Indigenous identity employees') |
#     (training_df_AvgAnnHrsWrk['Characteristics'] == 'Non-indigenous identity employees')
# ]
# print(df_AvgAnnHrkWrk_2010_ByIndigenous.head(20))
# # grouped = df_AvgAnnHrkWrk_2010_ByIndigenous.groupby(['Characteristics'])
# print(grouped['VALUE'].agg([np.sum, np.size]))
# print("The total number of this one is ",len(df_AvgAnnHrkWrk_2010_ByIndigenous.index))

# %%
# Dataset by testing set inside Average Annual Hours Worked

print("\nAge group")
testing_df_AvgAnnHrsWrk_ByAge = testing_df_AvgAnnHrsWrk.loc[
    (testing_df_AvgAnnHrsWrk['Characteristics'] == '15 to 24 years') |
    (testing_df_AvgAnnHrsWrk['Characteristics'] == '25 to 34 years') |
    (testing_df_AvgAnnHrsWrk['Characteristics'] == '35 to 44 years') |
    (testing_df_AvgAnnHrsWrk['Characteristics'] == '45 to 54 years') |
    (testing_df_AvgAnnHrsWrk['Characteristics'] == '55 to 64 years') |
    (testing_df_AvgAnnHrsWrk['Characteristics'] == '65 years old and over')]
# print(testing_df_AvgAnnHrsWrk_ByAge.head(20))
grouped = testing_df_AvgAnnHrsWrk_ByAge.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("Total size : ",len(testing_df_AvgAnnHrsWrk_ByAge.index))

print("\nGender group")
testing_df_AvgAnnHrsWrk_ByGender = testing_df_AvgAnnHrsWrk.loc[
    (testing_df_AvgAnnHrsWrk['Characteristics'] == 'Female employees') |
    (testing_df_AvgAnnHrsWrk['Characteristics'] == 'Male employees')
]
# print(testing_df_AvgAnnHrsWrk_ByGender.head(20))
grouped = testing_df_AvgAnnHrsWrk_ByGender.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("Total size : ",len(testing_df_AvgAnnHrsWrk_ByGender.index))

print("\nEducation groupa")
testing_df_AvgAnnHrsWrk_ByEducation = testing_df_AvgAnnHrsWrk.loc[
    (testing_df_AvgAnnHrsWrk['Characteristics'] == 'High school diploma and less') |
    (testing_df_AvgAnnHrsWrk['Characteristics'] == 'Trade certificate') |
    (testing_df_AvgAnnHrsWrk['Characteristics'] == 'University degree and higher')
]
# print(testing_df_AvgAnnHrsWrk_ByEducation.head(20))
grouped = testing_df_AvgAnnHrsWrk_ByEducation.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("Total size : ",len(testing_df_AvgAnnHrsWrk_ByEducation.index))

print("\nImmigrant group")
testing_df_AvgAnnHrsWrk_ByImmigrant = testing_df_AvgAnnHrsWrk.loc[
    (testing_df_AvgAnnHrsWrk['Characteristics'] == 'Immigrant employees') |
    (testing_df_AvgAnnHrsWrk['Characteristics'] == 'Non-immigrant employees')
]
# print(testing_df_AvgAnnHrsWrk_ByImmigrant.head(20))
grouped = testing_df_AvgAnnHrsWrk_ByImmigrant.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("Total size : ",len(testing_df_AvgAnnHrsWrk_ByImmigrant.index))

# print("\nIndigenous group in Alberta")
# df_AvgAnnHrkWrk_2010_ByIndigenous = testing_df_AvgAnnHrsWrk.loc[
#     (testing_df_AvgAnnHrsWrk['Characteristics'] == 'Indigenous identity employees') |
#     (testing_df_AvgAnnHrsWrk['Characteristics'] == 'Non-indigenous identity employees')
# ]
# print(df_AvgAnnHrkWrk_2010_ByIndigenous.head(20))
# # grouped = df_AvgAnnHrkWrk_2010_ByIndigenous.groupby(['Characteristics'])
# print(grouped['VALUE'].agg([np.sum, np.size]))
# print("The total number of this one is ",len(df_AvgAnnHrkWrk_2010_ByIndigenous.index))

# %% [markdown]
# Filtered for "Average annual wages and salaries" by following: "Age group", "Gender level", "Education level", and "Immigration status".<br />
# "Aboriginal status" has been commented.

# %%
# Dataset by training dataset inside Average annual wages and salaries

print("\nAge group")
training_df_AvgAnnWages_ByAge = training_df_AvgAnnWages.loc[
    (training_df_AvgAnnWages['Characteristics'] == '15 to 24 years') |
    (training_df_AvgAnnWages['Characteristics'] == '25 to 34 years') |
    (training_df_AvgAnnWages['Characteristics'] == '35 to 44 years') |
    (training_df_AvgAnnWages['Characteristics'] == '45 to 54 years') |
    (training_df_AvgAnnWages['Characteristics'] == '55 to 64 years') |
    (training_df_AvgAnnWages['Characteristics'] == '65 years old and over')]
# print(training_df_AvgAnnWages_ByAge.head(20))
grouped = training_df_AvgAnnWages_ByAge.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(training_df_AvgAnnWages_ByAge.index))

print("\nGender group")
training_df_AvgAnnWages_ByGender = training_df_AvgAnnWages.loc[
    (training_df_AvgAnnWages['Characteristics'] == 'Female employees') |
    (training_df_AvgAnnWages['Characteristics'] == 'Male employees')
]
# print(training_df_AvgAnnWages_ByGender.head(20))
grouped = training_df_AvgAnnWages_ByGender.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(training_df_AvgAnnWages_ByGender.index))

print("\nEducation group")
training_df_AvgAnnWages_ByEducation = training_df_AvgAnnWages.loc[
    (training_df_AvgAnnWages['Characteristics'] == 'High school diploma and less') |
    (training_df_AvgAnnWages['Characteristics'] == 'Trade certificate') |
    (training_df_AvgAnnWages['Characteristics'] == 'University degree and higher')
]
# print(training_df_AvgAnnWages_ByEducation.head(20))
grouped = training_df_AvgAnnWages_ByEducation.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(training_df_AvgAnnWages_ByEducation.index))

print("\nImmigrant group")
training_df_AvgAnnWages_ByImmigrant = training_df_AvgAnnWages.loc[
    (training_df_AvgAnnWages['Characteristics'] == 'Immigrant employees') |
    (training_df_AvgAnnWages['Characteristics'] == 'Non-immigrant employees')
]
# print(training_df_AvgAnnWages_ByImmigrant.head(20))
grouped = training_df_AvgAnnWages_ByImmigrant.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(training_df_AvgAnnWages_ByImmigrant.index))

# print("\nIndigenous group in Alberta")
# training_df_AvgAnnWages_ByIndigenous = training_df_AvgAnnWages.loc[
#     (training_df_AvgAnnWages['Characteristics'] == 'Indigenous identity employees') |
#     (training_df_AvgAnnWages['Characteristics'] == 'Non-indigenous identity employees')
# ]
# # print(df_AvgAnnHrk_ByIndigenous.head(20))
# grouped = training_df_AvgAnnWages_ByIndigenous.groupby(['Characteristics'])
# print(grouped['VALUE'].agg([np.sum, np.size]))
# print("The total number of this one is ",len(training_df_AvgAnnWages_ByIndigenous.index))

# %%
# Dataset by testing dataset inside Average annual wages and salaries

print("\nAge group")
testing_df_AvgAnnWages_ByAge = testing_df_AvgAnnWages.loc[
    (testing_df_AvgAnnWages['Characteristics'] == '15 to 24 years') |
    (testing_df_AvgAnnWages['Characteristics'] == '25 to 34 years') |
    (testing_df_AvgAnnWages['Characteristics'] == '35 to 44 years') |
    (testing_df_AvgAnnWages['Characteristics'] == '45 to 54 years') |
    (testing_df_AvgAnnWages['Characteristics'] == '55 to 64 years') |
    (testing_df_AvgAnnWages['Characteristics'] == '65 years old and over')]
# print(testing_df_AvgAnnWages_ByAge.head(20))
grouped = testing_df_AvgAnnWages_ByAge.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(testing_df_AvgAnnWages_ByAge.index))

print("\nGender group")
testing_df_AvgAnnWages_ByGender = testing_df_AvgAnnWages.loc[
    (testing_df_AvgAnnWages['Characteristics'] == 'Female employees') |
    (testing_df_AvgAnnWages['Characteristics'] == 'Male employees')
]
# print(testing_df_AvgAnnWages_ByGender.head(20))
grouped = testing_df_AvgAnnWages_ByGender.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(testing_df_AvgAnnWages_ByGender.index))

print("\nEducation group")
testing_df_AvgAnnWages_ByEducation = testing_df_AvgAnnWages.loc[
    (testing_df_AvgAnnWages['Characteristics'] == 'High school diploma and less') |
    (testing_df_AvgAnnWages['Characteristics'] == 'Trade certificate') |
    (testing_df_AvgAnnWages['Characteristics'] == 'University degree and higher')
]
# print(testing_df_AvgAnnWages_ByEducation.head(20))
grouped = testing_df_AvgAnnWages_ByEducation.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(testing_df_AvgAnnWages_ByEducation.index))

print("\nImmigrant group")
testing_df_AvgAnnWages_ByImmigrant = testing_df_AvgAnnWages.loc[
    (testing_df_AvgAnnWages['Characteristics'] == 'Immigrant employees') |
    (testing_df_AvgAnnWages['Characteristics'] == 'Non-immigrant employees')
]
# print(testing_df_AvgAnnWages_ByImmigrant.head(20))
grouped = testing_df_AvgAnnWages_ByImmigrant.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(testing_df_AvgAnnWages_ByImmigrant.index))

# print("\nIndigenous group in Alberta")
# testing_df_AvgAnnWages_ByIndigenous = testing_df_AvgAnnWages.loc[
#     (testing_df_AvgAnnWages['Characteristics'] == 'Indigenous identity employees') |
#     (testing_df_AvgAnnWages['Characteristics'] == 'Non-indigenous identity employees')
# ]
# # print(df_AvgAnnHrk_ByIndigenous.head(20))
# grouped = testing_df_AvgAnnWages_ByIndigenous.groupby(['Characteristics'])
# print(grouped['VALUE'].agg([np.sum, np.size]))
# print("The total number of this one is ",len(testing_df_AvgAnnWages_ByIndigenous.index))

# %% [markdown]
# Filtered for "Average hourly wage" by following: "Age group", "Gender level", "Education level", and "Immigration status". <br />
# "Aboriginal status" has been commented.

# %%
# Dataset by training dataset inside "Average hourly wage"

print("\nAge group")
training_df_AvgHrsWages_ByAge = training_df_AvgHrsWages.loc[
    (training_df_AvgHrsWages['Characteristics'] == '15 to 24 years') |
    (training_df_AvgHrsWages['Characteristics'] == '25 to 34 years') |
    (training_df_AvgHrsWages['Characteristics'] == '35 to 44 years') |
    (training_df_AvgHrsWages['Characteristics'] == '45 to 54 years') |
    (training_df_AvgHrsWages['Characteristics'] == '55 to 64 years') |
    (training_df_AvgHrsWages['Characteristics'] == '65 years old and over')]
# print(training_df_AvgHrsWages_ByAge.head(20))
grouped = training_df_AvgHrsWages_ByAge.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(training_df_AvgHrsWages_ByAge.index))

print("\nGender group")
training_df_AvgHrsWages_ByGender = training_df_AvgHrsWages.loc[
    (training_df_AvgHrsWages['Characteristics'] == 'Female employees') |
    (training_df_AvgHrsWages['Characteristics'] == 'Male employees')
]
# print(training_df_AvgHrsWages_ByGender.head(20))
grouped = training_df_AvgHrsWages_ByGender.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(training_df_AvgHrsWages_ByGender.index))

print("\nEducation group")
training_df_AvgHrsWages_ByEducation = training_df_AvgHrsWages.loc[
    (training_df_AvgHrsWages['Characteristics'] == 'High school diploma and less') |
    (training_df_AvgHrsWages['Characteristics'] == 'Trade certificate') |
    (training_df_AvgHrsWages['Characteristics'] == 'University degree and higher')
]
# print(training_df_AvgHrsWages_ByEducation.head(20))
grouped = training_df_AvgHrsWages_ByEducation.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(training_df_AvgHrsWages_ByEducation.index))

print("\nImmigrant group")
training_df_AvgHrsWages_ByImmigrant = training_df_AvgHrsWages.loc[
    (training_df_AvgHrsWages['Characteristics'] == 'Immigrant employees') |
    (training_df_AvgHrsWages['Characteristics'] == 'Non-immigrant employees')
]
# print(training_df_AvgHrsWages_ByImmigrant.head(20))
grouped = training_df_AvgHrsWages_ByImmigrant.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(training_df_AvgHrsWages_ByImmigrant.index))

# print("\nIndigenous group in Alberta")
# training_df_AvgHrsWages_ByIndigenous = training_df_AvgHrsWages.loc[
#     (training_df_AvgHrsWages['Characteristics'] == 'Indigenous identity employees') |
#     (training_df_AvgHrsWages['Characteristics'] == 'Non-indigenous identity employees')
# ]
# # print(training_df_AvgHrsWages_ByIndigenous.head(20))
# grouped = training_df_AvgHrsWages_ByIndigenous.groupby(['Characteristics'])
# print(grouped['VALUE'].agg([np.sum, np.size]))
# print("The total number of this one is ",len(training_df_AvgHrsWages_ByIndigenous.index))

# %%
# Dataset by testing dataset inside "Average hourly wage"

print("\nAge group")
testing_df_AvgHrsWages_ByAge = testing_df_AvgHrsWages.loc[
    (testing_df_AvgHrsWages['Characteristics'] == '15 to 24 years') |
    (testing_df_AvgHrsWages['Characteristics'] == '25 to 34 years') |
    (testing_df_AvgHrsWages['Characteristics'] == '35 to 44 years') |
    (testing_df_AvgHrsWages['Characteristics'] == '45 to 54 years') |
    (testing_df_AvgHrsWages['Characteristics'] == '55 to 64 years') |
    (testing_df_AvgHrsWages['Characteristics'] == '65 years old and over')]
# print(testing_df_AvgHrsWages_ByAge.head(20))
grouped = testing_df_AvgHrsWages_ByAge.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(testing_df_AvgHrsWages_ByAge.index))

print("\nGender group")
testing_df_AvgHrsWages_ByGender = testing_df_AvgHrsWages.loc[
    (testing_df_AvgHrsWages['Characteristics'] == 'Female employees') |
    (testing_df_AvgHrsWages['Characteristics'] == 'Male employees')
]
# print(testing_df_AvgHrsWages_ByGender.head(20))
grouped = testing_df_AvgHrsWages_ByGender.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(testing_df_AvgHrsWages_ByGender.index))

print("\nEducation group")
testing_df_AvgHrsWages_ByEducation = testing_df_AvgHrsWages.loc[
    (testing_df_AvgHrsWages['Characteristics'] == 'High school diploma and less') |
    (testing_df_AvgHrsWages['Characteristics'] == 'Trade certificate') |
    (testing_df_AvgHrsWages['Characteristics'] == 'University degree and higher')
]
# print(testing_df_AvgHrsWages_ByEducation.head(20))
grouped = testing_df_AvgHrsWages_ByEducation.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(testing_df_AvgHrsWages_ByEducation.index))

print("\nImmigrant group")
testing_df_AvgHrsWages_ByImmigrant = testing_df_AvgHrsWages.loc[
    (testing_df_AvgHrsWages['Characteristics'] == 'Immigrant employees') |
    (testing_df_AvgHrsWages['Characteristics'] == 'Non-immigrant employees')
]
# print(testing_df_AvgHrsWages_ByImmigrant.head(20))
grouped = testing_df_AvgHrsWages_ByImmigrant.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(testing_df_AvgHrsWages_ByImmigrant.index))

# print("\nIndigenous group in Alberta")
# testing_df_AvgHrsWages_ByIndigenous = testing_df_AvgHrsWages.loc[
#     (testing_df_AvgHrsWages['Characteristics'] == 'Indigenous identity employees') |
#     (testing_df_AvgHrsWages['Characteristics'] == 'Non-indigenous identity employees')
# ]
# # print(testing_df_AvgHrsWages_ByIndigenous.head(20))
# grouped = testing_df_AvgHrsWages_ByIndigenous.groupby(['Characteristics'])
# print(grouped['VALUE'].agg([np.sum, np.size]))
# print("The total number of this one is ",len(testing_df_AvgHrsWages_ByIndigenous.index))

# %% [markdown]
# Filtered for "Average weekly hours worked" by following: "Age group", "Gender level", "Education level", and "Immigration status".<br />
# "Aboriginal status" has been commented.

# %%
# Dataset by training dataset inside "Average weekly hours worked"

print("\nAge group")
training_df_AvgWeekHrsWrked_ByAge = training_df_AvgWeekHrsWrked.loc[
    (training_df_AvgWeekHrsWrked['Characteristics'] == '15 to 24 years') |
    (training_df_AvgWeekHrsWrked['Characteristics'] == '25 to 34 years') |
    (training_df_AvgWeekHrsWrked['Characteristics'] == '35 to 44 years') |
    (training_df_AvgWeekHrsWrked['Characteristics'] == '45 to 54 years') |
    (training_df_AvgWeekHrsWrked['Characteristics'] == '55 to 64 years') |
    (training_df_AvgWeekHrsWrked['Characteristics'] == '65 years old and over')]
# print(training_df_AvgWeekHrsWrked_ByAge.head(20))
grouped = training_df_AvgWeekHrsWrked_ByAge.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(training_df_AvgWeekHrsWrked_ByAge.index))

print("\nGender group")
training_df_AvgWeekHrsWrked_ByGender = training_df_AvgWeekHrsWrked.loc[
    (training_df_AvgWeekHrsWrked['Characteristics'] == 'Female employees') |
    (training_df_AvgWeekHrsWrked['Characteristics'] == 'Male employees')
]
# print(training_df_AvgWeekHrsWrked_ByGender.head(20))
grouped = training_df_AvgWeekHrsWrked_ByGender.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(training_df_AvgWeekHrsWrked_ByGender.index))

print("\nEducation group")
training_df_AvgWeekHrsWrked_ByEducation = training_df_AvgWeekHrsWrked.loc[
    (training_df_AvgWeekHrsWrked['Characteristics'] == 'High school diploma and less') |
    (training_df_AvgWeekHrsWrked['Characteristics'] == 'Trade certificate') |
    (training_df_AvgWeekHrsWrked['Characteristics'] == 'University degree and higher')
]
# print(training_df_AvgWeekHrsWrked_ByEducation.head(20))
grouped = training_df_AvgWeekHrsWrked_ByEducation.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(training_df_AvgWeekHrsWrked_ByEducation.index))

print("\nImmigrant group")
training_df_AvgWeekHrsWrked_ByImmigrant = training_df_AvgWeekHrsWrked.loc[
    (training_df_AvgWeekHrsWrked['Characteristics'] == 'Immigrant employees') |
    (training_df_AvgWeekHrsWrked['Characteristics'] == 'Non-immigrant employees')
]
# print(training_df_AvgWeekHrsWrked_ByImmigrant.head(20))
grouped = training_df_AvgWeekHrsWrked_ByImmigrant.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(training_df_AvgWeekHrsWrked_ByImmigrant.index))

# print("\nIndigenous group in Alberta")
# training_df_AvgWeekHrsWrked_ByIndigenous = training_df_AvgWeekHrsWrked.loc[
#     (training_df_AvgWeekHrsWrked['Characteristics'] == 'Indigenous identity employees') |
#     (training_df_AvgWeekHrsWrked['Characteristics'] == 'Non-indigenous identity employees')
# ]
# # print(training_df_AvgWeekHrsWrked_ByIndigenous.head(20))
# grouped = training_df_AvgWeekHrsWrked_ByIndigenous.groupby(['Characteristics'])
# print(grouped['VALUE'].agg([np.sum, np.size]))
# print("The total number of this one is ",len(training_df_AvgWeekHrsWrked_ByIndigenous.index))

# %%
# Dataset by testing dataset inside "Average weekly hours worked"

print("\nAge group")
testing_df_AvgWeekHrsWrked_ByAge = testing_df_AvgWeekHrsWrked.loc[
    (testing_df_AvgWeekHrsWrked['Characteristics'] == '15 to 24 years') |
    (testing_df_AvgWeekHrsWrked['Characteristics'] == '25 to 34 years') |
    (testing_df_AvgWeekHrsWrked['Characteristics'] == '35 to 44 years') |
    (testing_df_AvgWeekHrsWrked['Characteristics'] == '45 to 54 years') |
    (testing_df_AvgWeekHrsWrked['Characteristics'] == '55 to 64 years') |
    (testing_df_AvgWeekHrsWrked['Characteristics'] == '65 years old and over')]
# print(testing_df_AvgWeekHrsWrked_ByAge.head(20))
grouped = testing_df_AvgWeekHrsWrked_ByAge.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(testing_df_AvgWeekHrsWrked_ByAge.index))

print("\nGender group")
testing_df_AvgWeekHrsWrked_ByGender = testing_df_AvgWeekHrsWrked.loc[
    (testing_df_AvgWeekHrsWrked['Characteristics'] == 'Female employees') |
    (testing_df_AvgWeekHrsWrked['Characteristics'] == 'Male employees')
]
# print(testing_df_AvgWeekHrsWrked_ByGender.head(20))
grouped = testing_df_AvgWeekHrsWrked_ByGender.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(testing_df_AvgWeekHrsWrked_ByGender.index))

print("\nEducation group")
testing_df_AvgWeekHrsWrked_ByEducation = testing_df_AvgWeekHrsWrked.loc[
    (testing_df_AvgWeekHrsWrked['Characteristics'] == 'High school diploma and less') |
    (testing_df_AvgWeekHrsWrked['Characteristics'] == 'Trade certificate') |
    (testing_df_AvgWeekHrsWrked['Characteristics'] == 'University degree and higher')
]
# print(testing_df_AvgWeekHrsWrked_ByEducation.head(20))
grouped = testing_df_AvgWeekHrsWrked_ByEducation.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(testing_df_AvgWeekHrsWrked_ByEducation.index))

print("\nImmigrant group")
testing_df_AvgWeekHrsWrked_ByImmigrant = testing_df_AvgWeekHrsWrked.loc[
    (testing_df_AvgWeekHrsWrked['Characteristics'] == 'Immigrant employees') |
    (testing_df_AvgWeekHrsWrked['Characteristics'] == 'Non-immigrant employees')
]
# print(testing_df_AvgWeekHrsWrked_ByImmigrant.head(20))
grouped = testing_df_AvgWeekHrsWrked_ByImmigrant.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(testing_df_AvgWeekHrsWrked_ByImmigrant.index))

# print("\nIndigenous group in Alberta")
# testing_df_AvgWeekHrsWrked_ByIndigenous = testing_df_AvgWeekHrsWrked.loc[
#     (testing_df_AvgWeekHrsWrked['Characteristics'] == 'Indigenous identity employees') |
#     (testing_df_AvgWeekHrsWrked['Characteristics'] == 'Non-indigenous identity employees')
# ]
# # print(testing_df_AvgWeekHrsWrked_ByIndigenous.head(20))
# grouped = testing_df_AvgWeekHrsWrked_ByIndigenous.groupby(['Characteristics'])
# print(grouped['VALUE'].agg([np.sum, np.size]))
# print("The total number of this one is ",len(testing_df_AvgWeekHrsWrked_ByIndigenous.index))

# %% [markdown]
# Filtered for "Hours worked" by following: "Age group", "Gender level", "Education level", and "Immigration status".<br />
# "Aboriginal status" has been commented.
# 

# %%
# Dataset by training dataset inside "Hours Worked"

print("\nAge group in Alberta")
training_df_Hrs_Wrked_ByAge = training_df_Hrs_Wrked.loc[
    (training_df_Hrs_Wrked['Characteristics'] == '15 to 24 years') |
    (training_df_Hrs_Wrked['Characteristics'] == '25 to 34 years') |
    (training_df_Hrs_Wrked['Characteristics'] == '35 to 44 years') |
    (training_df_Hrs_Wrked['Characteristics'] == '45 to 54 years') |
    (training_df_Hrs_Wrked['Characteristics'] == '55 to 64 years') |
    (training_df_Hrs_Wrked['Characteristics'] == '65 years old and over')]
# print(training_df_Hrs_Wrked_ByAge.head(20))
grouped = training_df_Hrs_Wrked_ByAge.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(training_df_Hrs_Wrked_ByAge.index))

print("\nGender group in Alberta")
training_df_Hrs_Wrked_ByGender = training_df_Hrs_Wrked.loc[
    (training_df_Hrs_Wrked['Characteristics'] == 'Female employees') |
    (training_df_Hrs_Wrked['Characteristics'] == 'Male employees')
]
# print(training_df_Hrs_Wrked_ByGender.head(20))
grouped = training_df_Hrs_Wrked_ByGender.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(training_df_Hrs_Wrked_ByGender.index))

print("\nEducation group in Alberta")
training_df_Hrs_Wrked_ByEducation = training_df_Hrs_Wrked.loc[
    (training_df_Hrs_Wrked['Characteristics'] == 'High school diploma and less') |
    (training_df_Hrs_Wrked['Characteristics'] == 'Trade certificate') |
    (training_df_Hrs_Wrked['Characteristics'] == 'University degree and higher')
]
# print(training_df_Hrs_Wrked_ByEducation.head(20))
grouped = training_df_Hrs_Wrked_ByEducation.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(training_df_Hrs_Wrked_ByEducation.index))

print("\nImmigrant group in Alberta")
training_df_Hrs_Wrked_ByImmigrant = training_df_Hrs_Wrked.loc[
    (training_df_Hrs_Wrked['Characteristics'] == 'Immigrant employees') |
    (training_df_Hrs_Wrked['Characteristics'] == 'Non-immigrant employees')
]
# print(training_df_Hrs_Wrked_ByImmigrant.head(20))
grouped = training_df_Hrs_Wrked_ByImmigrant.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(training_df_Hrs_Wrked_ByImmigrant.index))

# print("\nIndigenous group in Alberta")
# training_df_Hrs_Wrked_ByIndigenous = training_df_Hrs_Wrked.loc[
#     (training_df_Hrs_Wrked['Characteristics'] == 'Indigenous identity employees') |
#     (training_df_Hrs_Wrked['Characteristics'] == 'Non-indigenous identity employees')
# ]
# # print(training_df_Hrs_Wrked_ByIndigenous.head(20))
# grouped = training_df_Hrs_Wrked_ByIndigenous.groupby(['Characteristics'])
# print(grouped['VALUE'].agg([np.sum, np.size]))
# print("The total number of this one is ",len(training_df_Hrs_Wrked_ByIndigenous.index))

# %%
# Dataset by testing dataset inside "Hours Worked"

print("\nAge group in Alberta")
testing_df_Hrs_Wrked_ByAge = testing_df_Hrs_Wrked.loc[
    (testing_df_Hrs_Wrked['Characteristics'] == '15 to 24 years') |
    (testing_df_Hrs_Wrked['Characteristics'] == '25 to 34 years') |
    (testing_df_Hrs_Wrked['Characteristics'] == '35 to 44 years') |
    (testing_df_Hrs_Wrked['Characteristics'] == '45 to 54 years') |
    (testing_df_Hrs_Wrked['Characteristics'] == '55 to 64 years') |
    (testing_df_Hrs_Wrked['Characteristics'] == '65 years old and over')]
# print(testing_df_Hrs_Wrked_ByAge.head(20))
grouped = testing_df_Hrs_Wrked_ByAge.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(testing_df_Hrs_Wrked_ByAge.index))

print("\nGender group in Alberta")
testing_df_Hrs_Wrked_ByGender = testing_df_Hrs_Wrked.loc[
    (testing_df_Hrs_Wrked['Characteristics'] == 'Female employees') |
    (testing_df_Hrs_Wrked['Characteristics'] == 'Male employees')
]
# print(testing_df_Hrs_Wrked_ByGender.head(20))
grouped = testing_df_Hrs_Wrked_ByGender.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(testing_df_Hrs_Wrked_ByGender.index))

print("\nEducation group in Alberta")
testing_df_Hrs_Wrked_ByEducation = testing_df_Hrs_Wrked.loc[
    (testing_df_Hrs_Wrked['Characteristics'] == 'High school diploma and less') |
    (testing_df_Hrs_Wrked['Characteristics'] == 'Trade certificate') |
    (testing_df_Hrs_Wrked['Characteristics'] == 'University degree and higher')
]
# print(testing_df_Hrs_Wrked_ByEducation.head(20))
grouped = testing_df_Hrs_Wrked_ByEducation.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(testing_df_Hrs_Wrked_ByEducation.index))

print("\nImmigrant group in Alberta")
testing_df_Hrs_Wrked_ByImmigrant = testing_df_Hrs_Wrked.loc[
    (testing_df_Hrs_Wrked['Characteristics'] == 'Immigrant employees') |
    (testing_df_Hrs_Wrked['Characteristics'] == 'Non-immigrant employees')
]
# print(testing_df_Hrs_Wrked_ByImmigrant.head(20))
grouped = testing_df_Hrs_Wrked_ByImmigrant.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(testing_df_Hrs_Wrked_ByImmigrant.index))

# print("\nIndigenous group in Alberta")
# testing_df_Hrs_Wrked_ByIndigenous = testing_df_Hrs_Wrked.loc[
#     (testing_df_Hrs_Wrked['Characteristics'] == 'Indigenous identity employees') |
#     (testing_df_Hrs_Wrked['Characteristics'] == 'Non-indigenous identity employees')
# ]
# # print(testing_df_Hrs_Wrked_ByIndigenous.head(20))
# grouped = testing_df_Hrs_Wrked_ByIndigenous.groupby(['Characteristics'])
# print(grouped['VALUE'].agg([np.sum, np.size]))
# print("The total number of this one is ",len(testing_df_Hrs_Wrked_ByIndigenous.index))

# %% [markdown]
# Filtered for "Number of jobs" by following: "Age group", "Gender level", "Education level", and "Immigration status".<br />
# "Aboriginal status" has been commented.

# %%
# Dataset by training dataset inside "Number of jobs"

print("\nAge group in Alberta")
training_df_NumOfJob_ByAge = training_df_NumOfJob.loc[
    (training_df_NumOfJob['Characteristics'] == '15 to 24 years') |
    (training_df_NumOfJob['Characteristics'] == '25 to 34 years') |
    (training_df_NumOfJob['Characteristics'] == '35 to 44 years') |
    (training_df_NumOfJob['Characteristics'] == '45 to 54 years') |
    (training_df_NumOfJob['Characteristics'] == '55 to 64 years') |
    (training_df_NumOfJob['Characteristics'] == '65 years old and over')]
# print(training_df_NumOfJob_ByAge.head(20))
grouped = training_df_NumOfJob_ByAge.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(training_df_NumOfJob_ByAge.index))

print("\nGender group in Alberta")
training_df_NumOfJob_ByGender = training_df_NumOfJob.loc[
    (training_df_NumOfJob['Characteristics'] == 'Female employees') |
    (training_df_NumOfJob['Characteristics'] == 'Male employees')
]
# print(training_df_NumOfJob_ByGender.head(20))
grouped = training_df_NumOfJob_ByGender.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(training_df_NumOfJob_ByGender.index))

print("\nEducation group in Alberta")
training_df_NumOfJob_ByEducation = training_df_NumOfJob.loc[
    (training_df_NumOfJob['Characteristics'] == 'High school diploma and less') |
    (training_df_NumOfJob['Characteristics'] == 'Trade certificate') |
    (training_df_NumOfJob['Characteristics'] == 'University degree and higher')
]
# print(training_df_NumOfJob_ByEducation.head(20))
grouped = training_df_NumOfJob_ByEducation.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(training_df_NumOfJob_ByEducation.index))

print("\nImmigrant group in Alberta")
training_df_NumOfJob_ByImmigrant = training_df_NumOfJob.loc[
    (training_df_NumOfJob['Characteristics'] == 'Immigrant employees') |
    (training_df_NumOfJob['Characteristics'] == 'Non-immigrant employees')
]
# print(training_df_NumOfJob_ByImmigrant.head(20))
grouped = training_df_NumOfJob_ByImmigrant.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(training_df_NumOfJob_ByImmigrant.index))

# print("\nIndigenous group in Alberta")
# training_df_NumOfJob_ByIndigenous = training_df_NumOfJob.loc[
#     (training_df_NumOfJob['Characteristics'] == 'Indigenous identity employees') |
#     (training_df_NumOfJob['Characteristics'] == 'Non-indigenous identity employees')
# ]
# # print(training_df_NumOfJob_ByIndigenous.head(20))
# grouped = training_df_NumOfJob_ByIndigenous.groupby(['Characteristics'])
# print(grouped['VALUE'].agg([np.sum, np.size]))
# print("The total number of this one is ",len(training_df_NumOfJob_ByIndigenous.index))

# %%
# Dataset by testing dataset inside "Number of jobs"

print("\nAge group in Alberta")
testing_df_NumOfJob_ByAge = testing_df_NumOfJob.loc[
    (testing_df_NumOfJob['Characteristics'] == '15 to 24 years') |
    (testing_df_NumOfJob['Characteristics'] == '25 to 34 years') |
    (testing_df_NumOfJob['Characteristics'] == '35 to 44 years') |
    (testing_df_NumOfJob['Characteristics'] == '45 to 54 years') |
    (testing_df_NumOfJob['Characteristics'] == '55 to 64 years') |
    (testing_df_NumOfJob['Characteristics'] == '65 years old and over')]
# print(testing_df_NumOfJob_ByAge.head(20))
grouped = testing_df_NumOfJob_ByAge.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(testing_df_NumOfJob_ByAge.index))

print("\nGender group in Alberta")
testing_df_NumOfJob_ByGender = testing_df_NumOfJob.loc[
    (testing_df_NumOfJob['Characteristics'] == 'Female employees') |
    (testing_df_NumOfJob['Characteristics'] == 'Male employees')
]
# print(testing_df_NumOfJob_ByGender.head(20))
grouped = testing_df_NumOfJob_ByGender.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(testing_df_NumOfJob_ByGender.index))

print("\nEducation group in Alberta")
testing_df_NumOfJob_ByEducation = testing_df_NumOfJob.loc[
    (testing_df_NumOfJob['Characteristics'] == 'High school diploma and less') |
    (testing_df_NumOfJob['Characteristics'] == 'Trade certificate') |
    (testing_df_NumOfJob['Characteristics'] == 'University degree and higher')
]
# print(testing_df_NumOfJob_ByEducation.head(20))
grouped = testing_df_NumOfJob_ByEducation.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(testing_df_NumOfJob_ByEducation.index))

print("\nImmigrant group in Alberta")
testing_df_NumOfJob_ByImmigrant = testing_df_NumOfJob.loc[
    (testing_df_NumOfJob['Characteristics'] == 'Immigrant employees') |
    (testing_df_NumOfJob['Characteristics'] == 'Non-immigrant employees')
]
# print(testing_df_NumOfJob_ByImmigrant.head(20))
grouped = testing_df_NumOfJob_ByImmigrant.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(testing_df_NumOfJob_ByImmigrant.index))

# print("\nIndigenous group in Alberta")
# testing_df_NumOfJob_ByIndigenous = testing_df_NumOfJob.loc[
#     (testing_df_NumOfJob['Characteristics'] == 'Indigenous identity employees') |
#     (testing_df_NumOfJob['Characteristics'] == 'Non-indigenous identity employees')
# ]
# # print(testing_df_NumOfJob_ByIndigenous.head(20))
# grouped = testing_df_NumOfJob_ByIndigenous.groupby(['Characteristics'])
# print(grouped['VALUE'].agg([np.sum, np.size]))
# print("The total number of this one is ",len(testing_df_NumOfJob_ByIndigenous.index))

# %% [markdown]
# Filtered for "Wages and Salaries" by following: "Age group", "Gender level", "Education level", and "Immigration status". <br />
# "Aboriginal status" has been commented.

# %%
# Dataset training set inside "Wages and Salaries"

print("\nAge group in Alberta")
training_df_WagesAndSalaries_ByAge = training_df_WagesAndSalaries.loc[
    (training_df_WagesAndSalaries['Characteristics'] == '15 to 24 years') |
    (training_df_WagesAndSalaries['Characteristics'] == '25 to 34 years') |
    (training_df_WagesAndSalaries['Characteristics'] == '35 to 44 years') |
    (training_df_WagesAndSalaries['Characteristics'] == '45 to 54 years') |
    (training_df_WagesAndSalaries['Characteristics'] == '55 to 64 years') |
    (training_df_WagesAndSalaries['Characteristics'] == '65 years old and over')]
# print(training_df_WagesAndSalaries_ByAge.head(20))
grouped = training_df_WagesAndSalaries_ByAge.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(training_df_WagesAndSalaries_ByAge.index))

print("\nGender group in Alberta")
training_df_WagesAndSalaries_ByGender = training_df_WagesAndSalaries.loc[
    (training_df_WagesAndSalaries['Characteristics'] == 'Female employees') |
    (training_df_WagesAndSalaries['Characteristics'] == 'Male employees')
]
# print(training_df_WagesAndSalaries_ByGender.head(20))
grouped = training_df_WagesAndSalaries_ByGender.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(training_df_WagesAndSalaries_ByGender.index))

print("\nEducation group in Alberta")
training_df_WagesAndSalaries_ByEducation = training_df_WagesAndSalaries.loc[
    (training_df_WagesAndSalaries['Characteristics'] == 'High school diploma and less') |
    (training_df_WagesAndSalaries['Characteristics'] == 'Trade certificate') |
    (training_df_WagesAndSalaries['Characteristics'] == 'University degree and higher')
]
# print(training_df_WagesAndSalaries_ByEducation.head(20))
grouped = training_df_WagesAndSalaries_ByEducation.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(training_df_WagesAndSalaries_ByEducation.index))

print("\nImmigrant group in Alberta")
training_df_WagesAndSalaries_ByImmigrant = training_df_WagesAndSalaries.loc[
    (training_df_WagesAndSalaries['Characteristics'] == 'Immigrant employees') |
    (training_df_WagesAndSalaries['Characteristics'] == 'Non-immigrant employees')
]
# print(training_df_WagesAndSalaries_ByImmigrant.head(20))
grouped = training_df_WagesAndSalaries_ByImmigrant.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(training_df_WagesAndSalaries_ByImmigrant.index))

# print("\nIndigenous group in Alberta")
# training_df_WagesAndSalaries_ByIndigenous = training_df_WagesAndSalaries.loc[
#     (training_df_WagesAndSalaries['Characteristics'] == 'Indigenous identity employees') |
#     (training_df_WagesAndSalaries['Characteristics'] == 'Non-indigenous identity employees')
# ]
# # print(training_df_WagesAndSalaries_ByIndigenous.head(20))
# grouped = training_df_WagesAndSalaries_ByIndigenous.groupby(['Characteristics'])
# print(grouped['VALUE'].agg([np.sum, np.size]))
# print("The total number of this one is ",len(training_df_WagesAndSalaries_ByIndigenous.index))

# %%
# Dataset testing dataset inside "Wages and Salaries"

print("\nAge group in Alberta")
testing_df_WagesAndSalaries_ByAge = testing_df_WagesAndSalaries.loc[
    (testing_df_WagesAndSalaries['Characteristics'] == '15 to 24 years') |
    (testing_df_WagesAndSalaries['Characteristics'] == '25 to 34 years') |
    (testing_df_WagesAndSalaries['Characteristics'] == '35 to 44 years') |
    (testing_df_WagesAndSalaries['Characteristics'] == '45 to 54 years') |
    (testing_df_WagesAndSalaries['Characteristics'] == '55 to 64 years') |
    (testing_df_WagesAndSalaries['Characteristics'] == '65 years old and over')]
# print(testing_df_WagesAndSalaries_ByAge.head(20))
grouped = testing_df_WagesAndSalaries_ByAge.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(testing_df_WagesAndSalaries_ByAge.index))

print("\nGender group in Alberta")
testing_df_WagesAndSalaries_ByGender = testing_df_WagesAndSalaries.loc[
    (testing_df_WagesAndSalaries['Characteristics'] == 'Female employees') |
    (testing_df_WagesAndSalaries['Characteristics'] == 'Male employees')
]
# print(testing_df_WagesAndSalaries_ByGender.head(20))
grouped = testing_df_WagesAndSalaries_ByGender.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(testing_df_WagesAndSalaries_ByGender.index))

print("\nEducation group in Alberta")
testing_df_WagesAndSalaries_ByEducation = testing_df_WagesAndSalaries.loc[
    (testing_df_WagesAndSalaries['Characteristics'] == 'High school diploma and less') |
    (testing_df_WagesAndSalaries['Characteristics'] == 'Trade certificate') |
    (testing_df_WagesAndSalaries['Characteristics'] == 'University degree and higher')
]
# print(testing_df_WagesAndSalaries_ByEducation.head(20))
grouped = testing_df_WagesAndSalaries_ByEducation.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(testing_df_WagesAndSalaries_ByEducation.index))

print("\nImmigrant group in Alberta")
testing_df_WagesAndSalaries_ByImmigrant = testing_df_WagesAndSalaries.loc[
    (testing_df_WagesAndSalaries['Characteristics'] == 'Immigrant employees') |
    (testing_df_WagesAndSalaries['Characteristics'] == 'Non-immigrant employees')
]
# print(testing_df_WagesAndSalaries_ByImmigrant.head(20))
grouped = testing_df_WagesAndSalaries_ByImmigrant.groupby(['Characteristics'])
print(grouped['VALUE'].agg([np.sum, np.size]))
print("The total number of this one is ",len(testing_df_WagesAndSalaries_ByImmigrant.index))

# print("\nIndigenous group in Alberta")
# testing_df_WagesAndSalaries_ByIndigenous = testing_df_WagesAndSalaries.loc[
#     (testing_df_WagesAndSalaries['Characteristics'] == 'Indigenous identity employees') |
#     (testing_df_WagesAndSalaries['Characteristics'] == 'Non-indigenous identity employees')
# ]
# # print(testing_df_WagesAndSalaries_ByIndigenous.head(20))
# grouped = testing_df_WagesAndSalaries_ByIndigenous.groupby(['Characteristics'])
# print(grouped['VALUE'].agg([np.sum, np.size]))
# print("The total number of this one is ",len(testing_df_WagesAndSalaries_ByIndigenous.index))

# %% [markdown]
# <h3> Part 8 - Other non-used indcators will be dropped </h3>

# %% [markdown]
# Next step, will be the final output.

# %% [markdown]
# Final Output for "Average annual hours worked"<br />
# First being Training dataset and second being Testing dataset.

# %%
dfa_Target_To_Analysis = [training_df_AvgAnnHrsWrk_ByAge, training_df_AvgAnnHrsWrk_ByGender,training_df_AvgAnnHrsWrk_ByEducation, training_df_AvgAnnHrsWrk_ByImmigrant]
dfa_Target_To_Analysis = Target_To_Analysis(dfa_Target_To_Analysis, pd, np, pp, sns, ['training set By Age',
                                                                                      'training set By Gender',
                                                                                      'training set By Education',
                                                                                      'training set By Immigrant'])
dfa_Target_To_Analysis.print_result()

# %%
print("Histogram for training dataset by age")
dfa_Target_To_Analysis.print_histogram(0)

print("Histgram for training dataset by gender")
dfa_Target_To_Analysis.print_histogram(1)

print("Histgram for training dataset by education")
dfa_Target_To_Analysis.print_histogram(2)

print("Histgram for training dataset by immigrant")
dfa_Target_To_Analysis.print_histogram(3)

# %%
dfa_Target_To_Analysis = [testing_df_AvgAnnHrsWrk_ByAge, testing_df_AvgAnnHrsWrk_ByGender, testing_df_AvgAnnHrsWrk_ByEducation, testing_df_AvgAnnHrsWrk_ByImmigrant]
dfa_Target_To_Analysis = Target_To_Analysis(dfa_Target_To_Analysis, pd, np, pp, sns, ['testing set By Age',
                                                                                      'testing set By Gender',
                                                                                      'testing set By Education',
                                                                                      'testing set By Immigrant'])
dfa_Target_To_Analysis.print_result()

# %%
print("Histogram for testing dataset by age")
dfa_Target_To_Analysis.print_histogram(0)

print("Histgram for testing dataset by gender")
dfa_Target_To_Analysis.print_histogram(1)

print("Histgram for testing dataset by education")
dfa_Target_To_Analysis.print_histogram(2)

print("Histgram for testing dataset by immigrant")
dfa_Target_To_Analysis.print_histogram(3)

# %% [markdown]
# Final Output for "Average annual wages and salaries"<br />
# First being Training dataset and second being Testing dataset.

# %%
dfa_Target_To_Analysis = [training_df_AvgAnnWages_ByAge, training_df_AvgAnnWages_ByGender,training_df_AvgAnnWages_ByEducation, training_df_AvgAnnWages_ByImmigrant]
dfa_Target_To_Analysis = Target_To_Analysis(dfa_Target_To_Analysis, pd, np, pp, sns, ['training set By Age',
                                                                                      'training set By Gender',
                                                                                      'training set By Education',
                                                                                      'training set By Immigrant'])
dfa_Target_To_Analysis.print_result()

# %%
print("Histogram for training dataset by age")
dfa_Target_To_Analysis.print_histogram(0)

print("Histgram for training dataset by gender")
dfa_Target_To_Analysis.print_histogram(1)

print("Histgram for training dataset by education")
dfa_Target_To_Analysis.print_histogram(2)

print("Histgram for training dataset by immigrant")
dfa_Target_To_Analysis.print_histogram(3)

# %%
dfa_Target_To_Analysis = [testing_df_AvgAnnWages_ByAge, testing_df_AvgAnnWages_ByGender, testing_df_AvgAnnWages_ByEducation, testing_df_AvgAnnWages_ByImmigrant]
dfa_Target_To_Analysis = Target_To_Analysis(dfa_Target_To_Analysis, pd, np, pp, sns, ['testing set By Age',
                                                                                      'testing set By Gender',
                                                                                      'testing set By Education',
                                                                                      'testing set By Immigrant'])
dfa_Target_To_Analysis.print_result()

# %%
print("Histogram for testing dataset by age")
dfa_Target_To_Analysis.print_histogram(0)

print("Histgram for testing dataset by gender")
dfa_Target_To_Analysis.print_histogram(1)

print("Histgram for testing dataset by education")
dfa_Target_To_Analysis.print_histogram(2)

print("Histgram for testing dataset by immigrant")
dfa_Target_To_Analysis.print_histogram(3)

# %% [markdown]
# Final Output for "Average hourly wage"<br />
# First being Training dataset and second being Testing dataset.

# %%
dfa_Target_To_Analysis = [training_df_AvgHrsWages_ByAge, training_df_AvgHrsWages_ByGender,training_df_AvgHrsWages_ByEducation, training_df_AvgHrsWages_ByImmigrant]
dfa_Target_To_Analysis = Target_To_Analysis(dfa_Target_To_Analysis, pd, np, pp, sns, ['training set By Age',
                                                                                      'training set By Gender',
                                                                                      'training set By Education',
                                                                                      'training set By Immigrant'])
dfa_Target_To_Analysis.print_result()

# %%
print("Histogram for training dataset by age")
dfa_Target_To_Analysis.print_histogram(0)

print("Histgram for training dataset by gender")
dfa_Target_To_Analysis.print_histogram(1)

print("Histgram for training dataset by education")
dfa_Target_To_Analysis.print_histogram(2)

print("Histgram for training dataset by immigrant")
dfa_Target_To_Analysis.print_histogram(3)

# %%
dfa_Target_To_Analysis = [testing_df_AvgHrsWages_ByAge, testing_df_AvgHrsWages_ByGender, testing_df_AvgHrsWages_ByEducation, testing_df_AvgHrsWages_ByImmigrant]
dfa_Target_To_Analysis = Target_To_Analysis(dfa_Target_To_Analysis, pd, np, pp, sns, ['testing set By Age',
                                                                                      'testing set By Gender',
                                                                                      'testing set By Education',
                                                                                      'testing set By Immigrant'])
dfa_Target_To_Analysis.print_result()

# %%
print("Histogram for testing dataset by age")
dfa_Target_To_Analysis.print_histogram(0)

print("Histgram for testing dataset by gender")
dfa_Target_To_Analysis.print_histogram(1)

print("Histgram for testing dataset by education")
dfa_Target_To_Analysis.print_histogram(2)

print("Histgram for testing dataset by immigrant")
dfa_Target_To_Analysis.print_histogram(3)

# %% [markdown]
# Final Output for "Average weekly hours worked"<br />
# First being Training dataset and second being Testing dataset.

# %%
dfa_Target_To_Analysis = [training_df_AvgWeekHrsWrked_ByAge, training_df_Hrs_Wrked_ByGender,training_df_Hrs_Wrked_ByEducation, training_df_Hrs_Wrked_ByImmigrant]
dfa_Target_To_Analysis = Target_To_Analysis(dfa_Target_To_Analysis, pd, np, pp, sns, ['training set By Age',
                                                                                      'training set By Gender',
                                                                                      'training set By Education',
                                                                                      'training set By Immigrant'])
dfa_Target_To_Analysis.print_result()

# %%
print("Histogram for training dataset by age")
dfa_Target_To_Analysis.print_histogram(0)

print("Histgram for training dataset by gender")
dfa_Target_To_Analysis.print_histogram(1)

print("Histgram for training dataset by education")
dfa_Target_To_Analysis.print_histogram(2)

print("Histgram for training dataset by immigrant")
dfa_Target_To_Analysis.print_histogram(3)

# %%
dfa_Target_To_Analysis = [testing_df_AvgWeekHrsWrked_ByAge, testing_df_AvgWeekHrsWrked_ByGender, testing_df_AvgWeekHrsWrked_ByEducation, testing_df_AvgWeekHrsWrked_ByImmigrant]
dfa_Target_To_Analysis = Target_To_Analysis(dfa_Target_To_Analysis, pd, np, pp, sns, ['testing set By Age',
                                                                                      'testing set By Gender',
                                                                                      'testing set By Education',
                                                                                      'testing set By Immigrant'])
dfa_Target_To_Analysis.print_result()

# %%
print("Histogram for testing dataset by age")
dfa_Target_To_Analysis.print_histogram(0)

print("Histgram for testing dataset by gender")
dfa_Target_To_Analysis.print_histogram(1)

print("Histgram for testing dataset by education")
dfa_Target_To_Analysis.print_histogram(2)

print("Histgram for testing dataset by immigrant")
dfa_Target_To_Analysis.print_histogram(3)

# %% [markdown]
# Final Output for "Hours Worked"<br />
# First being Training dataset and second being Testing dataset.

# %%
dfa_Target_To_Analysis = [training_df_Hrs_Wrked_ByAge, training_df_Hrs_Wrked_ByGender,training_df_Hrs_Wrked_ByEducation, training_df_Hrs_Wrked_ByImmigrant]
dfa_Target_To_Analysis = Target_To_Analysis(dfa_Target_To_Analysis, pd, np, pp, sns, ['training set By Age',
                                                                                      'training set By Gender',
                                                                                      'training set By Education',
                                                                                      'training set By Immigrant'])
dfa_Target_To_Analysis.print_result()

# %%
print("Histogram for training dataset by age")
dfa_Target_To_Analysis.print_histogram(0)

print("Histgram for training dataset by gender")
dfa_Target_To_Analysis.print_histogram(1)

print("Histgram for training dataset by education")
dfa_Target_To_Analysis.print_histogram(2)

print("Histgram for training dataset by immigrant")
dfa_Target_To_Analysis.print_histogram(3)

# %%
dfa_Target_To_Analysis = [testing_df_Hrs_Wrked_ByAge, testing_df_Hrs_Wrked_ByGender, testing_df_Hrs_Wrked_ByEducation, testing_df_Hrs_Wrked_ByImmigrant]
dfa_Target_To_Analysis = Target_To_Analysis(dfa_Target_To_Analysis, pd, np, pp, sns, ['testing set By Age',
                                                                                      'testing set By Gender',
                                                                                      'testing set By Education',
                                                                                      'testing set By Immigrant'])
dfa_Target_To_Analysis.print_result()

# %%
print("Histogram for testing dataset by age")
dfa_Target_To_Analysis.print_histogram(0)

print("Histgram for testing dataset by gender")
dfa_Target_To_Analysis.print_histogram(1)

print("Histgram for testing dataset by education")
dfa_Target_To_Analysis.print_histogram(2)

print("Histgram for testing dataset by immigrant")
dfa_Target_To_Analysis.print_histogram(3)

# %% [markdown]
# Final Output for "Number of jobs"<br />
# First being Training dataset and second being Testing dataset.

# %%
dfa_Target_To_Analysis = [training_df_NumOfJob_ByAge, training_df_NumOfJob_ByGender,training_df_NumOfJob_ByEducation, training_df_NumOfJob_ByImmigrant]
dfa_Target_To_Analysis = Target_To_Analysis(dfa_Target_To_Analysis, pd, np, pp, sns, ['training set By Age',
                                                                                      'training set By Gender',
                                                                                      'training set By Education',
                                                                                      'training set By Immigrant'])
dfa_Target_To_Analysis.print_result()

# %%
print("Histogram for training dataset by age")
dfa_Target_To_Analysis.print_histogram(0)

print("Histgram for training dataset by gender")
dfa_Target_To_Analysis.print_histogram(1)

print("Histgram for training dataset by education")
dfa_Target_To_Analysis.print_histogram(2)

print("Histgram for training dataset by immigrant")
dfa_Target_To_Analysis.print_histogram(3)

# %%
dfa_Target_To_Analysis = [testing_df_NumOfJob_ByAge, testing_df_NumOfJob_ByGender, testing_df_NumOfJob_ByEducation, testing_df_NumOfJob_ByImmigrant]
dfa_Target_To_Analysis = Target_To_Analysis(dfa_Target_To_Analysis, pd, np, pp, sns, ['testing set By Age',
                                                                                      'testing set By Gender',
                                                                                      'testing set By Education',
                                                                                      'testing set By Immigrant'])
dfa_Target_To_Analysis.print_result()

# %%
print("Histogram for testing dataset by age")
dfa_Target_To_Analysis.print_histogram(0)

print("Histgram for testing dataset by gender")
dfa_Target_To_Analysis.print_histogram(1)

print("Histgram for testing dataset by education")
dfa_Target_To_Analysis.print_histogram(2)

print("Histgram for testing dataset by immigrant")
dfa_Target_To_Analysis.print_histogram(3)

# %% [markdown]
# Final Output for "Wages and Salaries"<br />
# First being Training dataset and second being Testing dataset.

# %%
dfa_Target_To_Analysis = [training_df_WagesAndSalaries_ByAge, training_df_WagesAndSalaries_ByGender,training_df_WagesAndSalaries_ByEducation, training_df_WagesAndSalaries_ByImmigrant]
dfa_Target_To_Analysis = Target_To_Analysis(dfa_Target_To_Analysis, pd, np, pp, sns, ['training set By Age',
                                                                                      'training set By Gender',
                                                                                      'training set By Education',
                                                                                      'training set By Immigrant'])
dfa_Target_To_Analysis.print_result()

# %%
print("Histogram for training dataset by age")
dfa_Target_To_Analysis.print_histogram(0)

print("Histgram for training dataset by gender")
dfa_Target_To_Analysis.print_histogram(1)

print("Histgram for training dataset by education")
dfa_Target_To_Analysis.print_histogram(2)

print("Histgram for training dataset by immigrant")
dfa_Target_To_Analysis.print_histogram(3)

# %%
dfa_Target_To_Analysis = [testing_df_WagesAndSalaries_ByAge, testing_df_WagesAndSalaries_ByGender, testing_df_WagesAndSalaries_ByEducation, testing_df_WagesAndSalaries_ByImmigrant]
dfa_Target_To_Analysis = Target_To_Analysis(dfa_Target_To_Analysis, pd, np, pp, sns, ['testing set By Age',
                                                                                      'testing set By Gender',
                                                                                      'testing set By Education',
                                                                                      'testing set By Immigrant'])
dfa_Target_To_Analysis.print_result()

# %%
print("Histogram for testing dataset by age")
dfa_Target_To_Analysis.print_histogram(0)

print("Histgram for testing dataset by gender")
dfa_Target_To_Analysis.print_histogram(1)

print("Histgram for testing dataset by education")
dfa_Target_To_Analysis.print_histogram(2)

print("Histgram for testing dataset by immigrant")
dfa_Target_To_Analysis.print_histogram(3)

# %% [markdown]
# Back up previous result to the CSV file.

# %%
# Save the dataframe to a CSV file

training_df_AvgAnnHrsWrk.to_csv('Result_By_Characteristics/training_df_AvgAnnHrsWrk.csv', index=False) # Average annual hours worked
testing_df_AvgAnnHrsWrk.to_csv('Result_By_Characteristics/testing_df_AvgAnnHrsWrk.csv', index=False)

training_df_AvgAnnWages.to_csv('Result_By_Characteristics/training_df_AvgAnnWages.csv', index=False) # Average annual wages and salaries
testing_df_AvgAnnWages.to_csv('Result_By_Characteristics/testing_df_AvgAnnWages.csv', index=False)

training_df_AvgHrsWages.to_csv('Result_By_Characteristics/training_df_AvgHrsWages.csv', index=False) # Average hourly wage
testing_df_AvgHrsWages.to_csv('Result_By_Characteristics/testing_df_AvgHrsWages.csv', index=False)

training_df_AvgWeekHrsWrked.to_csv('Result_By_Characteristics/training_df_AvgWeekHrsWrked.csv', index=False) # Average weekly hours worked
testing_df_AvgWeekHrsWrked.to_csv('Result_By_Characteristics/testing_df_AvgWeekHrsWrked.csv', index=False)

training_df_Hrs_Wrked.to_csv('Result_By_Characteristics/training_df_Hrs_Wrked.csv', index=False) # Hours Worked
testing_df_Hrs_Wrked.to_csv('Result_By_Characteristics/testing_df_Hrs_Wrked.csv', index=False)

training_df_NumOfJob.to_csv('Result_By_Characteristics/training_df_NumOfJob.csv', index=False) # Number of jobs
testing_df_NumOfJob.to_csv('Result_By_Characteristics/testing_df_NumOfJob.csv', index=False)

training_df_WagesAndSalaries.to_csv('Result_By_Characteristics/training_df_WagesAndSalaries.csv', index=False) # Wages and Salaries
testing_df_WagesAndSalaries.to_csv('Result_By_Characteristics/testing_df_WagesAndSalaries.csv', index=False)

# %% [markdown]
# <h3> Part 9 - Divide dataset by provinces but use only five provinces </h3>

# %% [markdown]
# <b>First part, divide by provinces</b>

# %%
# https://www.educative.io/blog/one-hot-encoding
# https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
# https://www.geeksforgeeks.org/ml-one-hot-encoding-of-datasets-in-python/
# https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/
# https://www.baeldung.com/cs/train-test-datasets-ratio


# %%
print("Final steps, by sorting out by provinces.")

# -- sum          mean           std  size
# -- GEO                                                                    
# -- Alberta                     2193966.0   2031.450000   2695.836034  1080
# -- British Columbia            2401296.0   2223.422222   2804.925187  1080
# -- Canada                     18252439.0  16900.406481  22232.852533  1080
# -- Manitoba                     767802.0    710.927778    915.637659  1080
# -- New Brunswick                359320.0    332.703704    530.962762  1080
# -- Newfoundland and Labrador    315895.0    306.099806    482.634908  1032
# -- Northwest Territories         42804.0     41.476744     51.817046  1032
# -- Nova Scotia                  531805.0    492.412037    757.119411  1080
# -- Nunavut                       14235.0     15.208333     14.752372   936
# -- Ontario                     6601634.0   6112.624074   7594.433779  1080
# -- Prince Edward Island          77931.0     75.514535    121.297367  1032
# -- Quebec                      4271657.0   3955.237963   5580.294544  1080
# -- Saskatchewan                 650781.0    602.575000    876.896377  1080
# -- Yukon                         16914.0     18.070513     20.188135   936
# -- The total number of this one is  14688


# %% [markdown]
# Created directory called "Provinces" to managed files for split related to provinces dataset.

# %%
CreatedTheFile = toOrganizedOutputFiles('Result_By_Provinces')

# %% [markdown]
# As final step, I am classifying the data by province. This is the final step, and this is where I will get the final result with.<br />
# For this step, I will use class methods to avoid duplicated and repeatitive steps to do programming.<br />
# For the complex of the analysis, only training (2013-2018) and testing set (2019-2021) are being used. <br />
# However, 2010-2012 one will get commented.

# %% [markdown]
# Main class for Province Analysis:

# %%
# https://www.w3schools.com/python/python_classes.asp
# https://www.w3schools.com/python/python_for_loops.asp
# https://www.educba.com/multidimensional-array-in-python/

class ProvinceAnalysis:

    # Province :
    # -- ['Alberta',  'British Columbia',    'Canada' , 'Manitoba' , 'New Brunswick' 
    # 'Newfoundland and Labrador', 'Northwest Territories' , 'Nova Scotia' , 'Nunavut'
    # 'Ontario' , 'Prince Edward Island', 'Quebec', 'Saskatchewan', 'Yukon']

    def __init__(self, df, pd, np, pp):
        self.df = df
        self.province = ['Alberta',  'British Columbia', 'Canada', 'Manitoba', 
                        'New Brunswick', 'Newfoundland and Labrador', 
                        'Northwest Territories' , 'Nova Scotia' , 'Nunavut',
                        'Ontario' , 'Prince Edward Island', 'Quebec', 
                        'Saskatchewan', 'Yukon'
                        ]
        self.indicator = ["Average annual hours worked",
                        "Average annual wages and salaries",
                        "Average hourly wage",
                        "Average weekly hours worked",
                        "Hours Worked",
                        "Number of jobs",
                        "Wages and Salaries"]
        self.characteristic = ["Age group", "Gender", "Education Level", "Immigrant status", "Aboriginal status"]
        self.year = ["2010",
                    "below 2015",
                    "above 2016",
                    "2013",
                    "2016",
                    "2019"]
        self.pd = pd
        self.np = np
        self.pp = pp
        self.df_ByProvince = []
        for x in self.province:
            df_sorted = df.loc[df['GEO'] == x]
            self.df_ByProvince.append(df_sorted)

    def outputProvince(self, province_id):
        print(self.province[province_id])

    def outputIndicator(self, indicator_id):
        print(self.province[indicator_id])

    def outputCharacteristic(self, cha_id):
        print(self.province[cha_id])

    def outputYear(self, year_id):
        print(self.province[year_id])

    def outputAnalysis(self, province_id):
        print("\nGrab the dataset only in " + str(self.province[province_id]))
        grouped = self.df_ByProvince[province_id].groupby(['Characteristics'])
        print(grouped['VALUE'].agg([np.sum, np.mean, np.min, np.median, np.max, np.size]))
        print("")
        print("Overall,")
        print("Sum : ",np.sum(self.df_ByProvince[province_id]['VALUE']))
        print("Mean : ",np.mean(self.df_ByProvince[province_id]['VALUE']))
        print("Min/median/max :",np.min(self.df_ByProvince[province_id]['VALUE']),"/",
            np.median(self.df_ByProvince[province_id]['VALUE']),"/",
            np.max(self.df_ByProvince[province_id]['VALUE']))
        print("Skewnewss : ",self.df_ByProvince[province_id]['VALUE'].skew())
        print("Total size : ",len(self.df_ByProvince[province_id].index))

    def outputAnalysisSimple(self, province_id):
        print("\nGrab the dataset only in " + str(self.province[province_id]))
        grouped = self.df_ByProvince[province_id].groupby(['Characteristics'])
        print(grouped['VALUE'].agg([self.np.sum, self.np.mean, self.np.size]))

    def outputList(self, province_id, num):
        print("\nGrab the dataset only in " + str(self.province[province_id]))
        print(self.df_ByProvince[province_id].head(num))
        print(self.df_ByProvince[province_id].info())

    def outputPandaProfiling(self, province_id, indicator_id, type_id):

        fileName = str(self.indicator[indicator_id]) + " " + str(self.year[type_id])+" in " + str(self.province[province_id]) + ".html"
        
        pp = ProfileReport(self.df_ByProvince[province_id])
        pp_df = pp.to_html()

        print("File name will be saved under "+str(fileName))
        f = open(fileName, "a")  # Expert into html file without modifying any columns in dataset.
        f.write(pp_df)
        f.close()
    
    def print_histogram(self, province_id):
        sns.displot(data=self.province[province_id], x="VALUE", kind="hist", bins = 100, aspect = 1.5)
        plt.show()

    def outputFiveProvinces(self, pro1, pro2, pro3, pro4, pro5):
        frames = [self.df_ByProvince[pro1], self.df_ByProvince[pro2], self.df_ByProvince[pro3], self.df_ByProvince[pro4], self.df_ByProvince[pro5]]
        result = pd.concat(frames)
        return result

# %% [markdown]
# Filtered by provinces by "Average annual hours worked"

# %%
# By Average annual hours worked categories by provinces.

training_df_AvgAnnHrsWrk_ByAge_Provinces = ProvinceAnalysis(training_df_AvgAnnHrsWrk_ByAge, pd, np, pp)
testing_df_AvgAnnHrsWrk_ByAge_Provinces = ProvinceAnalysis(testing_df_AvgAnnHrsWrk_ByAge, pd, np, pp)

training_df_AvgAnnHrsWrk_ByGender_Provinces = ProvinceAnalysis(training_df_AvgAnnHrsWrk_ByGender, pd, np, pp)
testing_df_AvgAnnHrsWrk_ByGender_Provinces = ProvinceAnalysis(testing_df_AvgAnnHrsWrk_ByGender, pd, np, pp)

training_df_AvgAnnHrsWrk_ByEducation_Provinces = ProvinceAnalysis(training_df_AvgAnnHrsWrk_ByEducation, pd, np, pp)
testing_df_AvgAnnHrsWrk_ByEducation_Provinces = ProvinceAnalysis(testing_df_AvgAnnHrsWrk_ByEducation, pd, np, pp)

training_df_AvgAnnHrsWrk_ByImmigrant_Provinces = ProvinceAnalysis(training_df_AvgAnnHrsWrk_ByImmigrant, pd, np, pp)
testing_df_AvgAnnHrsWrk_ByImmigrant_Provinces = ProvinceAnalysis(testing_df_AvgAnnHrsWrk_ByImmigrant, pd, np, pp)

# training_df_AvgAnnHrsWrk_ByIndigenous_Provinces = ProvinceAnalysis(training_df_AvgAnnHrsWrk_ByIndigenous, pd, np, pp)
# testing_df_AvgAnnHrsWrk_ByIndigenous_Provinces = ProvinceAnalysis(testing_df_AvgAnnHrsWrk_ByIndigenous, pd, np, pp)

# %% [markdown]
# Filtered by provinces by "Average wages and salaries"

# %%
# By Average annual wages and salaries worked categories by provinces.

training_df_AvgAnnWages_ByAge_Provinces = ProvinceAnalysis(training_df_AvgAnnWages_ByAge, pd, np, pp)
testing_df_AvgAnnWages_ByAge_Provinces = ProvinceAnalysis(testing_df_AvgAnnWages_ByAge, pd, np, pp)

training_df_AvgAnnWages_ByGender_Provinces = ProvinceAnalysis(training_df_AvgAnnWages_ByGender, pd, np, pp)
testing_df_AvgAnnWages_ByGender_Provinces = ProvinceAnalysis(testing_df_AvgAnnWages_ByGender, pd, np, pp)

training_df_AvgAnnWages_ByEducation_Provinces = ProvinceAnalysis(training_df_AvgAnnWages_ByEducation, pd, np, pp)
testing_df_AvgAnnWages_ByEducation_Provinces = ProvinceAnalysis(testing_df_AvgAnnWages_ByEducation, pd, np, pp)

training_df_AvgAnnWages_ByImmigrant_Provinces = ProvinceAnalysis(training_df_AvgAnnWages_ByImmigrant, pd, np, pp)
testing_df_AvgAnnWages_ByImmigrant_Provinces = ProvinceAnalysis(testing_df_AvgAnnWages_ByImmigrant, pd, np, pp)

# training_df_AvgAnnWages_ByIndigenous_Provinces = ProvinceAnalysis(training_df_AvgAnnWages_ByIndigenous, pd, np, pp)
# testing_df_AvgAnnWages_ByIndigenous_Provinces = ProvinceAnalysis(testing_df_AvgAnnWages_ByIndigenous, pd, np, pp)

# %% [markdown]
# Filtered by provinces by "Average hourly wage"

# %%
# By Average hourly wages and salaries worked categories by provinces.

training_df_AvgHrsWages_ByAge_Provinces = ProvinceAnalysis(training_df_AvgHrsWages_ByAge, pd, np, pp)
testing_df_AvgHrsWages_ByAge_Provinces = ProvinceAnalysis(testing_df_AvgHrsWages_ByAge, pd, np, pp)

training_df_AvgHrsWages_ByGender_Provinces = ProvinceAnalysis(training_df_AvgHrsWages_ByGender, pd, np, pp)
testing_df_AvgHrsWages_ByGender_Provinces = ProvinceAnalysis(testing_df_AvgHrsWages_ByGender, pd, np, pp)

training_df_AvgHrsWages_ByEducation_Provinces = ProvinceAnalysis(training_df_AvgHrsWages_ByEducation, pd, np, pp)
testing_df_AvgHrsWages_ByEducation_Provinces = ProvinceAnalysis(testing_df_AvgHrsWages_ByEducation, pd, np, pp)

training_df_AvgHrsWages_ByImmigrant_Provinces = ProvinceAnalysis(training_df_AvgHrsWages_ByImmigrant, pd, np, pp)
testing_df_AvgHrsWages_ByImmigrant_Provinces = ProvinceAnalysis(testing_df_AvgHrsWages_ByImmigrant, pd, np, pp)

# training_df_AvgHrsWages_ByIndigenous_Provinces = ProvinceAnalysis(training_df_AvgHrsWages_ByIndigenous, pd, np, pp)
# testing_df_AvgHrsWages_ByIndigenous_Provinces = ProvinceAnalysis(testing_df_AvgHrsWages_ByIndigenous, pd, np, pp)

# %% [markdown]
# Filtered by provinces by "Average weekly hours worked"

# %%
# By Average annual wages and salaries worked categories by provinces.

training_df_AvgWeekHrsWrked_ByAge_Provinces = ProvinceAnalysis(training_df_AvgWeekHrsWrked_ByAge, pd, np, pp)
testing_df_AvgWeekHrsWrked_ByAge_Provinces = ProvinceAnalysis(testing_df_AvgWeekHrsWrked_ByAge, pd, np, pp)

training_df_AvgWeekHrsWrked_ByGender_Provinces = ProvinceAnalysis(training_df_AvgWeekHrsWrked_ByGender, pd, np, pp)
testing_df_AvgWeekHrsWrked_ByGender_Provinces = ProvinceAnalysis(testing_df_AvgWeekHrsWrked_ByGender, pd, np, pp)

training_df_AvgWeekHrsWrked_ByEducation_Provinces = ProvinceAnalysis(training_df_AvgWeekHrsWrked_ByEducation, pd, np, pp)
testing_df_AvgWeekHrsWrked_ByEducation_Provinces = ProvinceAnalysis(testing_df_AvgWeekHrsWrked_ByEducation, pd, np, pp)

training_df_AvgWeekHrsWrked_ByImmigrant_Provinces = ProvinceAnalysis(training_df_AvgWeekHrsWrked_ByImmigrant, pd, np, pp)
testing_df_AvgWeekHrsWrked_ByImmigrant_Provinces = ProvinceAnalysis(testing_df_AvgWeekHrsWrked_ByImmigrant, pd, np, pp)

# training_df_AvgWeekHrsWrked_ByIndigenous_Provinces = ProvinceAnalysis(training_df_AvgWeekHrsWrked_ByIndigenous, pd, np, pp)
# testing_df_AvgWeekHrsWrked_ByIndigenous_Provinces = ProvinceAnalysis(testing_df_AvgWeekHrsWrked_ByIndigenous, pd, np, pp)

# %% [markdown]
# Filtered by provinces by "Hours Worked"

# %%
# By Hours worked and salaries worked categories by provinces.

training_df_Hrs_Wrked_ByAge_Provinces = ProvinceAnalysis(training_df_Hrs_Wrked_ByAge, pd, np, pp)
testing_df_Hrs_Wrked_ByAge_Provinces = ProvinceAnalysis(testing_df_Hrs_Wrked_ByAge, pd, np, pp)

training_df_Hrs_Wrked_ByGender_Provinces = ProvinceAnalysis(training_df_Hrs_Wrked_ByGender, pd, np, pp)
testing_df_Hrs_Wrked_ByGender_Provinces = ProvinceAnalysis(testing_df_Hrs_Wrked_ByGender, pd, np, pp)

training_df_Hrs_Wrked_ByEducation_Provinces = ProvinceAnalysis(training_df_Hrs_Wrked_ByEducation, pd, np, pp)
testing_df_Hrs_Wrked_ByEducation_Provinces = ProvinceAnalysis(testing_df_Hrs_Wrked_ByEducation, pd, np, pp)

training_df_Hrs_Wrked_ByImmigrant_Provinces = ProvinceAnalysis(training_df_Hrs_Wrked_ByImmigrant, pd, np, pp)
testing_df_Hrs_Wrked_ByImmigrant_Provinces = ProvinceAnalysis(testing_df_Hrs_Wrked_ByImmigrant, pd, np, pp)

# training_df_Hrs_Wrked_ByIndigenous_Provinces = ProvinceAnalysis(training_df_Hrs_Wrked_ByIndigenous, pd, np, pp)
# testing_df_Hrs_Wrked_ByIndigenous_Provinces = ProvinceAnalysis(testing_df_Hrs_Wrked_ByIndigenous, pd, np, pp)


# %% [markdown]
# Filtered by provinces by "Number of jobs"

# %%
# By Number of jobs and salaries worked categories by provinces.

training_df_NumOfJob_ByAge_Provinces = ProvinceAnalysis(training_df_NumOfJob_ByAge, pd, np, pp)
testing_df_NumOfJob_ByAge_Provinces = ProvinceAnalysis(testing_df_NumOfJob_ByAge, pd, np, pp)

training_df_NumOfJob_ByGender_Provinces = ProvinceAnalysis(training_df_NumOfJob_ByGender, pd, np, pp)
testing_df_NumOfJob_ByGender_Provinces = ProvinceAnalysis(testing_df_NumOfJob_ByGender, pd, np, pp)

training_df_NumOfJob_ByEducation_Provinces = ProvinceAnalysis(training_df_NumOfJob_ByEducation, pd, np, pp)
testing_df_NumOfJob_ByEducation_Provinces = ProvinceAnalysis(testing_df_NumOfJob_ByEducation, pd, np, pp)

training_df_NumOfJob_ByImmigrant_Provinces = ProvinceAnalysis(training_df_NumOfJob_ByImmigrant, pd, np, pp)
testing_df_NumOfJob_ByImmigrant_Provinces = ProvinceAnalysis(testing_df_NumOfJob_ByImmigrant, pd, np, pp)

# training_df_NumOfJob_ByIndigenous_Provinces = ProvinceAnalysis(training_df_NumOfJob_ByIndigenous, pd, np, pp)
# testing_df_NumOfJob_ByIndigenous_Provinces = ProvinceAnalysis(testing_df_NumOfJob_ByIndigenous, pd, np, pp)

# %% [markdown]
# Filted by provinces by "Wages and Salaries"

# %%
# By Wages and Salaries worked categories by provinces.

training_df_WagesAndSalaries_ByAge_Provinces = ProvinceAnalysis(training_df_WagesAndSalaries_ByAge, pd, np, pp)
testing_df_WagesAndSalaries_ByAge_Provinces = ProvinceAnalysis(testing_df_WagesAndSalaries_ByAge, pd, np, pp)

training_df_WagesAndSalaries_ByGender_Provinces = ProvinceAnalysis(training_df_WagesAndSalaries_ByGender, pd, np, pp)
testing_df_WagesAndSalaries_ByGender_Provinces = ProvinceAnalysis(testing_df_WagesAndSalaries_ByGender, pd, np, pp)

training_df_WagesAndSalaries_ByEducation_Provinces = ProvinceAnalysis(training_df_WagesAndSalaries_ByEducation, pd, np, pp)
testing_df_WagesAndSalaries_ByEducation_Provinces = ProvinceAnalysis(testing_df_WagesAndSalaries_ByEducation, pd, np, pp)

training_df_WagesAndSalaries_ByImmigrant_Provinces = ProvinceAnalysis(training_df_WagesAndSalaries_ByImmigrant, pd, np, pp)
testing_df_WagesAndSalaries_ByImmigrant_Provinces = ProvinceAnalysis(testing_df_WagesAndSalaries_ByImmigrant, pd, np, pp)

# training_df_WagesAndSalaries_ByIndigenous_Provinces = ProvinceAnalysis(training_df_WagesAndSalaries_ByIndigenous, pd, np, pp)
# testing_df_WagesAndSalaries_ByIndigenous_Provinces = ProvinceAnalysis(testing_df_WagesAndSalaries_ByIndigenous, pd, np, pp)

# %% [markdown]
# <b>Next part, select five provinces and merge with previous divided dataset.</b>

# %% [markdown]
# Since there will be too many datasets to deal with if I were to working with many provinces. <br />
# Instead I will be selected only five provinces to work on and will be put it in one database. <br />

# %% [markdown]
# Class methods to deal with five provinces,

# %%
# Import label encoder 
from sklearn import preprocessing 

class FiveProvinceAnalysis:

    # Province :
    # -- ['Alberta',  'British Columbia',    'Canada' , 'Manitoba' , 'New Brunswick' 
    # 'Newfoundland and Labrador', 'Northwest Territories' , 'Nova Scotia' , 'Nunavut'
    # 'Ontario' , 'Prince Edward Island', 'Quebec', 'Saskatchewan', 'Yukon']

    def __init__(self, df, pd, np, pp):

        # Based on this result, https://www.linkedin.com/pulse/5-best-provinces-canada-look-jobs-2023-/
        # Five popular province for employments, ['Alberta', 'BC', 'Nova Scotia', 'Ontario', 'Quebec']

        temp_df = df.outputFiveProvinces(0,1,7,9,11)
        
        self.df_FiveProvinces = temp_df.copy()
        self.one_hot_encoded_data = pd.get_dummies(temp_df, columns = ['GEO'])# , 'Characteristics']) 
        self.pd = pd
        self.np = np

    def convertCategoricalToNumericValue(self, ct):
        # ct = ['age', 'gender', 'education', 'immigrant', 'aboriginal']

        if (ct == 0):
            # https://www.statology.org/pandas-create-duplicate-column/
            # https://saturncloud.io/blog/how-to-replace-values-on-specific-columns-in-pandas/
            # Alternative, https://www.statology.org/data-binning-in-python/
            # Using Binning numerical variables from https://www.datacamp.com/tutorial/categorical-data

            # ['15 to 24 years' '25 to 34 years' '35 to 44 years' '45 to 54 years' '55 to 64 years' '65 years old and over']
            
            self.one_hot_encoded_data['Age_group'] = self.one_hot_encoded_data.loc[:,'Characteristics'] # pd.qcut(df['Characteristics'], q=3)
            # print(self.one_hot_encoded_data['Age_group'].unique())

            age_mapping = {
                '15 to 24 years': 20,
                '25 to 34 years': 30,
                '35 to 44 years': 40,
                '45 to 54 years': 50,
                '55 to 64 years': 60,
                '65 years old and over': 70
            }

            # Define a custom function
            def replace_agestr_with_number(age_group):
                return age_mapping.get(age_group, age_group)

            # Apply the custom function to the 'Age_Group' column
            # self.one_hot_encoded_data = df.copy()
            self.one_hot_encoded_data['Age_group'] = self.one_hot_encoded_data['Age_group'].apply(replace_agestr_with_number)
            # print(self.one_hot_encoded_data['Age_Group'].unique())
            # print(self.one_hot_encoded_data.head(20))
        elif (ct == 2):
            self.one_hot_encoded_data['Education_group'] = self.one_hot_encoded_data.loc[:,'Characteristics']

            education_mapping = {
                'High school diploma and less': 1,
                'Trade certificate': 2,
                'University degree and higher': 3,
            }

            # Define a custom function
            def replace_agestr_with_number(education_group):
                return education_mapping.get(education_group, education_group)

            # Apply the custom function to the 'Age_Group' column
            self.one_hot_encoded_data['Education_group'] = self.one_hot_encoded_data['Education_group'].apply(replace_agestr_with_number)
        elif (ct == 1) :
            # https://www.geeksforgeeks.org/ml-label-encoding-of-datasets-in-python/

            self.one_hot_encoded_data['Gender_group'] = self.one_hot_encoded_data.loc[:,'Characteristics']
            
            # label_encoder object knows  
            # how to understand word labels. 
            label_encoder = preprocessing.LabelEncoder() 
            
            # Encode labels in column 'species'. 
            self.one_hot_encoded_data['Gender_group'] = label_encoder.fit_transform(self.one_hot_encoded_data['Gender_group'] ) 
        elif (ct == 3):
            # Repeat from ct = 1
            self.one_hot_encoded_data['Immigrant_status'] = self.one_hot_encoded_data.loc[:,'Characteristics']
            label_encoder = preprocessing.LabelEncoder() 
            self.one_hot_encoded_data['Immigrant_status'] = label_encoder.fit_transform(self.one_hot_encoded_data['Immigrant_status'] ) 
        elif (ct == 4):
            # Repeat from ct = 1
            self.one_hot_encoded_data['Aboriginal_status'] = self.one_hot_encoded_data.loc[:,'Characteristics']
            label_encoder = preprocessing.LabelEncoder() 
            self.one_hot_encoded_data['Aboriginal_status'] = label_encoder.fit_transform(self.one_hot_encoded_data['Aboriginal_status'] ) 
        else:
            print("Error! Not Egliable to convert.")

    def print_Unique(self, to_id):
        print(self.one_hot_encoded_data[to_id].unique())
    
    def print_pre_Unique(self, to_id):
        print(self.df_FiveProvinces[to_id].unique())

    def print_Province_Unique(self):
        self.print_Unique('GEO')

    def print_Value_Counts(self, to_id):
        print(self.df_FiveProvinces[to_id].value_counts())

    def print_Province_Value_Counts(self):
        self.print_Value_Count('GEO')

    def print_One_Hot_Encoded_Data(self):
        print(self.one_hot_encoded_data.head(20))

    def print_One_Hot_Encoded_Data_Info(self):
        print(self.one_hot_encoded_data.info())

    def output_One_Hot_Encoded_Data(self):
        return self.one_hot_encoded_data
    
    def output_Original_Data(self):
        return self.df_FiveProvinces
    
    def print_Analysis_Provinces(self):
        print("Five popular province for employments,")
        print(self.print_Province_Unique())
        print("Sources: https://www.linkedin.com/pulse/5-best-provinces-canada-look-jobs-2023-/")

    def print_AnalysisResult_ByCharacteristics(self):
        grouped = self.df_FiveProvinces.groupby(['Characteristics'])
        print(grouped['VALUE'].agg([np.sum, np.mean, np.min, np.median, np.max, np.std, np.size]))
        print("")
        print("Overall,")
        print("Sum : ",np.sum(self.df_FiveProvinces['VALUE']))
        print("Mean : ",np.mean(self.df_FiveProvinces['VALUE']))
        print("Min/median/max :",np.min(self.df_FiveProvinces['VALUE']),"/",
            np.median(self.df_FiveProvinces['VALUE']),"/",
            np.max(self.df_FiveProvinces['VALUE']))
        print("Standard Deviation: ",np.std(self.df_FiveProvinces['VALUE']))
        print("Skewnewss : ",self.df_FiveProvinces['VALUE'].skew())
        print("Total size : ",len(self.df_FiveProvinces.index))

    def print_AnalysisResult_ByProvinces(self):
        grouped = self.df_FiveProvinces.groupby(['GEO'])
        print(grouped['VALUE'].agg([np.sum, np.mean, np.min, np.median, np.max, np.std, np.size]))
        print("")
        print("Overall,")
        print("Sum : ",np.sum(self.df_FiveProvinces['VALUE']))
        print("Mean : ",np.mean(self.df_FiveProvinces['VALUE']))
        print("Min/median/max :",np.min(self.df_FiveProvinces['VALUE']),"/",
            np.median(self.df_FiveProvinces['VALUE']),"/",
            np.max(self.df_FiveProvinces['VALUE']))
        print("Standard Deviation: ",np.std(self.df_FiveProvinces['VALUE']))
        print("Skewnewss : ",self.df_FiveProvinces['VALUE'].skew())
        print("Total size : ",len(self.df_FiveProvinces.index))

    def print_histogram(self):
        sns.displot(data=self.df_FiveProvinces, x="VALUE", kind="hist", bins = 100, aspect = 1.5)
        plt.show()

# %%
training_df_AvgAnnHrsWrk_ByAge_FiveProvinces = FiveProvinceAnalysis(training_df_AvgAnnHrsWrk_ByAge_Provinces, pd, np, pp)
training_df_AvgAnnHrsWrk_ByAge_FiveProvinces.convertCategoricalToNumericValue(0)
testing_df_AvgAnnHrsWrk_ByAge_FiveProvinces = FiveProvinceAnalysis(testing_df_AvgAnnHrsWrk_ByAge_Provinces, pd, np, pp)
testing_df_AvgAnnHrsWrk_ByAge_FiveProvinces.convertCategoricalToNumericValue(0)
testing_df_AvgAnnHrsWrk_ByAge_FiveProvinces.print_One_Hot_Encoded_Data_Info()
testing_df_AvgAnnHrsWrk_ByAge_FiveProvinces.print_One_Hot_Encoded_Data()
testing_df_AvgAnnHrsWrk_ByAge_FiveProvinces.print_Unique('Age_group')


training_df_AvgAnnHrsWrk_ByGender_FiveProvinces = FiveProvinceAnalysis(training_df_AvgAnnHrsWrk_ByGender_Provinces, pd, np, pp)
training_df_AvgAnnHrsWrk_ByGender_FiveProvinces.convertCategoricalToNumericValue(1)
testing_df_AvgAnnHrsWrk_ByGender_FiveProvinces = FiveProvinceAnalysis(testing_df_AvgAnnHrsWrk_ByGender_Provinces, pd, np, pp)
testing_df_AvgAnnHrsWrk_ByGender_FiveProvinces.convertCategoricalToNumericValue(1)
testing_df_AvgAnnHrsWrk_ByGender_FiveProvinces.print_One_Hot_Encoded_Data_Info()
testing_df_AvgAnnHrsWrk_ByGender_FiveProvinces.print_Unique('Gender_group')

training_df_AvgAnnHrsWrk_ByEducation_FiveProvinces = FiveProvinceAnalysis(training_df_AvgAnnHrsWrk_ByEducation_Provinces, pd, np, pp)
training_df_AvgAnnHrsWrk_ByEducation_FiveProvinces.convertCategoricalToNumericValue(2)
testing_df_AvgAnnHrsWrk_ByEducation_FiveProvinces = FiveProvinceAnalysis(testing_df_AvgAnnHrsWrk_ByEducation_Provinces, pd, np, pp)
testing_df_AvgAnnHrsWrk_ByEducation_FiveProvinces.convertCategoricalToNumericValue(2)
testing_df_AvgAnnHrsWrk_ByEducation_FiveProvinces.print_One_Hot_Encoded_Data_Info()
testing_df_AvgAnnHrsWrk_ByEducation_FiveProvinces.print_Unique('Education_group')

training_df_AvgAnnHrsWrk_ByImmigrant_FiveProvinces = FiveProvinceAnalysis(training_df_AvgAnnHrsWrk_ByImmigrant_Provinces, pd, np, pp)
training_df_AvgAnnHrsWrk_ByImmigrant_FiveProvinces.convertCategoricalToNumericValue(3)
testing_df_AvgAnnHrsWrk_ByImmigrant_FiveProvinces = FiveProvinceAnalysis(testing_df_AvgAnnHrsWrk_ByImmigrant_Provinces, pd, np, pp)
testing_df_AvgAnnHrsWrk_ByImmigrant_FiveProvinces.convertCategoricalToNumericValue(3)
testing_df_AvgAnnHrsWrk_ByImmigrant_FiveProvinces.print_One_Hot_Encoded_Data_Info()
testing_df_AvgAnnHrsWrk_ByImmigrant_FiveProvinces.print_Unique('Immigrant_status')

# training_df_AvgAnnHrsWrk_ByIndigenous_FiveProvinces = FiveProvinceAnalysis(training_df_AvgAnnHrsWrk_ByIndigenous_Provinces, pd, np, pp)
# training_df_AvgAnnHrsWrk_ByIndigenous_FiveProvinces.convertCategoricalToNumericValue(4)
# testing_df_AvgAnnHrsWrk_ByIndigenous_FiveProvinces = FiveProvinceAnalysis(training_df_AvgAnnHrsWrk_ByIndigenous_Provinces, pd, np, pp)
# testing_df_AvgAnnHrsWrk_ByIndigenous_FiveProvinces.convertCategoricalToNumericValue(4)
# testing_df_AvgAnnHrsWrk_ByIndigenous_FiveProvinces.print_One_Hot_Encoded_Data_Info()
# testing_df_AvgAnnHrsWrk_ByIndigenous_FiveProvinces.print_Unique('Aboriginal_status')

# %%
training_df_AvgAnnWages_ByAge_FiveProvinces = FiveProvinceAnalysis(training_df_AvgAnnWages_ByAge_Provinces, pd, np, pp)
training_df_AvgAnnWages_ByAge_FiveProvinces.convertCategoricalToNumericValue(0)
testing_df_AvgAnnWages_ByAge_FiveProvinces = FiveProvinceAnalysis(testing_df_AvgAnnWages_ByAge_Provinces, pd, np, pp)
testing_df_AvgAnnWages_ByAge_FiveProvinces.convertCategoricalToNumericValue(0)
testing_df_AvgAnnWages_ByAge_FiveProvinces.print_One_Hot_Encoded_Data_Info()
# testing_df_AvgAnnWages_ByAge_FiveProvinces.print_Unique('Age_Group')

training_df_AvgAnnWages_ByGender_FiveProvinces = FiveProvinceAnalysis(training_df_AvgAnnWages_ByGender_Provinces, pd, np, pp)
training_df_AvgAnnWages_ByGender_FiveProvinces.convertCategoricalToNumericValue(1)
testing_df_AvgAnnWages_ByGender_FiveProvinces = FiveProvinceAnalysis(testing_df_AvgAnnWages_ByGender_Provinces, pd, np, pp)
testing_df_AvgAnnWages_ByGender_FiveProvinces.convertCategoricalToNumericValue(1)
testing_df_AvgAnnWages_ByGender_FiveProvinces.print_One_Hot_Encoded_Data_Info()
testing_df_AvgAnnWages_ByGender_FiveProvinces.print_Unique('Gender_group')

training_df_AvgAnnWages_ByEducation_FiveProvinces = FiveProvinceAnalysis(training_df_AvgAnnWages_ByEducation_Provinces, pd, np, pp)
training_df_AvgAnnWages_ByEducation_FiveProvinces.convertCategoricalToNumericValue(2)
testing_df_AvgAnnWages_ByEducation_FiveProvinces = FiveProvinceAnalysis(testing_df_AvgAnnWages_ByEducation_Provinces, pd, np, pp)
testing_df_AvgAnnWages_ByEducation_FiveProvinces.convertCategoricalToNumericValue(2)
testing_df_AvgAnnWages_ByEducation_FiveProvinces.print_One_Hot_Encoded_Data_Info()
testing_df_AvgAnnWages_ByEducation_FiveProvinces.print_Unique('Education_group')

training_df_AvgAnnWages_ByImmigrant_FiveProvinces = FiveProvinceAnalysis(training_df_AvgAnnWages_ByImmigrant_Provinces, pd, np, pp)
training_df_AvgAnnWages_ByImmigrant_FiveProvinces.convertCategoricalToNumericValue(3)
testing_df_AvgAnnWages_ByImmigrant_FiveProvinces = FiveProvinceAnalysis(testing_df_AvgAnnWages_ByImmigrant_Provinces, pd, np, pp)
testing_df_AvgAnnWages_ByImmigrant_FiveProvinces.convertCategoricalToNumericValue(3)
testing_df_AvgAnnWages_ByImmigrant_FiveProvinces.print_One_Hot_Encoded_Data_Info()
testing_df_AvgAnnWages_ByImmigrant_FiveProvinces.print_Unique('Immigrant_status')

# training_df_AvgAnnWages_ByIndigenous_FiveProvinces = FiveProvinceAnalysis(training_df_AvgAnnWages_ByIndigenous_Provinces, pd, np, pp)
# training_df_AvgAnnWages_ByIndigenous_FiveProvinces.convertCategoricalToNumericValue(4)
# testing_df_AvgAnnWages_ByIndigenous_FiveProvinces = FiveProvinceAnalysis(training_df_AvgAnnWages_ByIndigenous_Provinces, pd, np, pp)
# testing_df_AvgAnnWages_ByIndigenous_FiveProvinces.convertCategoricalToNumericValue(4)
# testing_df_AvgAnnWages_ByIndigenous_FiveProvinces.print_One_Hot_Encoded_Data_Info()
# testing_df_AvgAnnWages_ByIndigenous_FiveProvinces.print_Unique('Immigrant_status')

# %%
training_df_AvgHrsWages_ByAge_FiveProvinces = FiveProvinceAnalysis(training_df_AvgHrsWages_ByAge_Provinces, pd, np, pp)
training_df_AvgHrsWages_ByAge_FiveProvinces.convertCategoricalToNumericValue(0)
testing_df_AvgHrsWages_ByAge_FiveProvinces = FiveProvinceAnalysis(testing_df_AvgHrsWages_ByAge_Provinces, pd, np, pp)
testing_df_AvgHrsWages_ByAge_FiveProvinces.convertCategoricalToNumericValue(0)
testing_df_AvgHrsWages_ByAge_FiveProvinces.print_One_Hot_Encoded_Data_Info()
testing_df_AvgHrsWages_ByAge_FiveProvinces.print_Unique('Age_group')

training_df_AvgHrsWages_ByGender_FiveProvinces = FiveProvinceAnalysis(training_df_AvgHrsWages_ByGender_Provinces, pd, np, pp)
training_df_AvgHrsWages_ByGender_FiveProvinces.convertCategoricalToNumericValue(1)
testing_df_AvgHrsWages_ByGender_FiveProvinces = FiveProvinceAnalysis(testing_df_AvgHrsWages_ByGender_Provinces, pd, np, pp)
testing_df_AvgHrsWages_ByGender_FiveProvinces.convertCategoricalToNumericValue(1)
testing_df_AvgHrsWages_ByGender_FiveProvinces.print_One_Hot_Encoded_Data_Info()
testing_df_AvgHrsWages_ByGender_FiveProvinces.print_Unique('Gender_group')

training_df_AvgHrsWages_ByEducation_FiveProvinces = FiveProvinceAnalysis(training_df_AvgHrsWages_ByEducation_Provinces, pd, np, pp)
training_df_AvgHrsWages_ByEducation_FiveProvinces.convertCategoricalToNumericValue(2)
testing_df_AvgHrsWages_ByEducation_FiveProvinces = FiveProvinceAnalysis(testing_df_AvgHrsWages_ByEducation_Provinces, pd, np, pp)
testing_df_AvgHrsWages_ByEducation_FiveProvinces.convertCategoricalToNumericValue(2)
testing_df_AvgHrsWages_ByEducation_FiveProvinces.print_One_Hot_Encoded_Data_Info()
testing_df_AvgHrsWages_ByEducation_FiveProvinces.print_Unique('Education_group')

training_df_AvgHrsWages_ByImmigrant_FiveProvinces = FiveProvinceAnalysis(training_df_AvgHrsWages_ByImmigrant_Provinces, pd, np, pp)
training_df_AvgHrsWages_ByImmigrant_FiveProvinces.convertCategoricalToNumericValue(3)
testing_df_AvgHrsWages_ByImmigrant_FiveProvinces = FiveProvinceAnalysis(testing_df_AvgHrsWages_ByImmigrant_Provinces, pd, np, pp)
testing_df_AvgHrsWages_ByImmigrant_FiveProvinces.convertCategoricalToNumericValue(3)
testing_df_AvgHrsWages_ByImmigrant_FiveProvinces.print_One_Hot_Encoded_Data_Info()
testing_df_AvgHrsWages_ByImmigrant_FiveProvinces.print_Unique('Immigrant_status')

# training_df_AvgHrsWages_ByIndigenous_FiveProvince = FiveProvinceAnalysis(training_df_AvgHrsWages_ByIndigenous_Provinces, pd, np, pp)
# training_df_AvgHrsWages_ByImmigrant_FiveProvince.print_One_Hot_Encoded_Data_Info()
# testing_df_AvgHrsWages_ByIndigenous_FiveProvince = FiveProvinceAnalysis(training_df_AvgHrsWages_ByIndigenous_Provinces, pd, np, pp)
# testing_df_AvgHrsWages_ByImmigrant_FiveProvince.print_One_Hot_Encoded_Data_Info()

# %%
training_df_AvgWeekHrsWrked_ByAge_FiveProvinces = FiveProvinceAnalysis(training_df_AvgWeekHrsWrked_ByAge_Provinces, pd, np, pp)
training_df_AvgWeekHrsWrked_ByAge_FiveProvinces.convertCategoricalToNumericValue(0)
testing_df_AvgWeekHrsWrked_ByAge_FiveProvinces = FiveProvinceAnalysis(testing_df_AvgWeekHrsWrked_ByAge_Provinces, pd, np, pp)
testing_df_AvgWeekHrsWrked_ByAge_FiveProvinces.convertCategoricalToNumericValue(0)
testing_df_AvgWeekHrsWrked_ByAge_FiveProvinces.print_One_Hot_Encoded_Data_Info()
# testing_df_AvgWeekHrsWrked_ByAge_FiveProvinces.print_Unique('Age_Group')

training_df_AvgWeekHrsWrked_ByGender_FiveProvinces = FiveProvinceAnalysis(training_df_AvgWeekHrsWrked_ByGender_Provinces, pd, np, pp)
training_df_AvgWeekHrsWrked_ByGender_FiveProvinces.convertCategoricalToNumericValue(1)
testing_df_AvgWeekHrsWrked_ByGender_FiveProvinces = FiveProvinceAnalysis(testing_df_AvgWeekHrsWrked_ByGender_Provinces, pd, np, pp)
testing_df_AvgWeekHrsWrked_ByGender_FiveProvinces.convertCategoricalToNumericValue(1)
testing_df_AvgWeekHrsWrked_ByGender_FiveProvinces.print_One_Hot_Encoded_Data_Info()
testing_df_AvgWeekHrsWrked_ByGender_FiveProvinces.print_Unique('Gender_group')

training_df_AvgWeekHrsWrked_ByEducation_FiveProvinces = FiveProvinceAnalysis(training_df_AvgWeekHrsWrked_ByEducation_Provinces, pd, np, pp)
training_df_AvgWeekHrsWrked_ByEducation_FiveProvinces .convertCategoricalToNumericValue(2)
testing_df_AvgWeekHrsWrked_ByEducation_FiveProvinces = FiveProvinceAnalysis(testing_df_AvgWeekHrsWrked_ByEducation_Provinces, pd, np, pp)
testing_df_AvgWeekHrsWrked_ByEducation_FiveProvinces.convertCategoricalToNumericValue(2)
testing_df_AvgWeekHrsWrked_ByEducation_FiveProvinces.print_Unique('Education_group')
testing_df_AvgWeekHrsWrked_ByEducation_FiveProvinces.print_One_Hot_Encoded_Data_Info()

training_df_AvgWeekHrsWrked_ByImmigrant_FiveProvinces = FiveProvinceAnalysis(training_df_AvgWeekHrsWrked_ByImmigrant_Provinces, pd, np, pp)
training_df_AvgWeekHrsWrked_ByImmigrant_FiveProvinces.convertCategoricalToNumericValue(3)
testing_df_AvgWeekHrsWrked_ByImmigrant_FiveProvinces = FiveProvinceAnalysis(testing_df_AvgWeekHrsWrked_ByImmigrant_Provinces, pd, np, pp)
testing_df_AvgWeekHrsWrked_ByImmigrant_FiveProvinces.convertCategoricalToNumericValue(3)
testing_df_AvgWeekHrsWrked_ByImmigrant_FiveProvinces.print_One_Hot_Encoded_Data_Info()
testing_df_AvgWeekHrsWrked_ByImmigrant_FiveProvinces.print_Unique('Immigrant_status')

# training_df_AvgWeekHrsWrked_ByIndigenous_FiveProvince = FiveProvinceAnalysis(training_df_AvgWeekHrsWrkeds_ByIndigenous_Provinces, pd, np, pp)
# training_df_AvgWeekHrsWrked_ByImmigrant_FiveProvince.print_One_Hot_Encoded_Data_Info()
# testing_df_AvgWeekHrsWrked_ByIndigenous_FiveProvince = FiveProvinceAnalysis(training_df_AvgWeekHrsWrkeds_ByIndigenous_Provinces, pd, np, pp)

# %%
training_df_Hrs_Wrked_ByAge_FiveProvinces = FiveProvinceAnalysis(training_df_Hrs_Wrked_ByAge_Provinces, pd, np, pp)
training_df_Hrs_Wrked_ByAge_FiveProvinces.convertCategoricalToNumericValue(0)
testing_df_Hrs_Wrked_ByAge_FiveProvinces = FiveProvinceAnalysis(testing_df_Hrs_Wrked_ByAge_Provinces, pd, np, pp)
testing_df_Hrs_Wrked_ByAge_FiveProvinces.convertCategoricalToNumericValue(0)
testing_df_Hrs_Wrked_ByAge_FiveProvinces.print_One_Hot_Encoded_Data_Info()
testing_df_Hrs_Wrked_ByAge_FiveProvinces.print_Unique('Age_group')

training_df_Hrs_Wrked_ByGender_FiveProvinces = FiveProvinceAnalysis(training_df_Hrs_Wrked_ByGender_Provinces, pd, np, pp)
training_df_Hrs_Wrked_ByGender_FiveProvinces.convertCategoricalToNumericValue(1)
testing_df_Hrs_Wrked_ByGender_FiveProvinces = FiveProvinceAnalysis(testing_df_Hrs_Wrked_ByGender_Provinces, pd, np, pp)
testing_df_Hrs_Wrked_ByGender_FiveProvinces.convertCategoricalToNumericValue(1)
testing_df_Hrs_Wrked_ByGender_FiveProvinces.print_One_Hot_Encoded_Data_Info()
testing_df_Hrs_Wrked_ByGender_FiveProvinces.print_Unique('Gender_group')

training_df_Hrs_Wrked_ByEducation_FiveProvinces = FiveProvinceAnalysis(training_df_Hrs_Wrked_ByEducation_Provinces, pd, np, pp)
training_df_Hrs_Wrked_ByEducation_FiveProvinces.convertCategoricalToNumericValue(2)
testing_df_Hrs_Wrked_ByEducation_FiveProvinces = FiveProvinceAnalysis(testing_df_Hrs_Wrked_ByEducation_Provinces, pd, np, pp)
testing_df_Hrs_Wrked_ByEducation_FiveProvinces.convertCategoricalToNumericValue(2)
testing_df_Hrs_Wrked_ByEducation_FiveProvinces.print_One_Hot_Encoded_Data_Info()
testing_df_Hrs_Wrked_ByEducation_FiveProvinces.print_Unique('Education_group')

training_df_Hrs_Wrked_ByImmigrant_FiveProvinces = FiveProvinceAnalysis(training_df_Hrs_Wrked_ByImmigrant_Provinces, pd, np, pp)
training_df_Hrs_Wrked_ByImmigrant_FiveProvinces.convertCategoricalToNumericValue(3)
testing_df_Hrs_Wrked_ByImmigrant_FiveProvinces = FiveProvinceAnalysis(testing_df_Hrs_Wrked_ByImmigrant_Provinces, pd, np, pp)
testing_df_Hrs_Wrked_ByImmigrant_FiveProvinces.convertCategoricalToNumericValue(3)
testing_df_Hrs_Wrked_ByImmigrant_FiveProvinces.print_One_Hot_Encoded_Data_Info()
testing_df_Hrs_Wrked_ByImmigrant_FiveProvinces.print_Unique('Immigrant_status')

# training_df_Hrs_Wrkeds_ByIndigenous_FiveProvince = FiveProvinceAnalysis(training_df_Hrs_Wrkeds_ByIndigenous_Provinces, pd, np, pp)
# training_df_Hrs_Wrkeds_ByImmigrant_FiveProvince.print_One_Hot_Encoded_Data_Info()
# testing_df_Hrs_Wrkeds_ByIndigenous_FiveProvince = FiveProvinceAnalysis(training_df_Hrs_Wrkeds_ByIndigenous_Provinces, pd, np, pp)
# testing_df_Hrs_Wrkeds_ByImmigrant_FiveProvince.print_One_Hot_Encoded_Data_Info()

# %%
training_df_NumOfJob_ByAge_FiveProvinces = FiveProvinceAnalysis(training_df_NumOfJob_ByAge_Provinces, pd, np, pp)
training_df_NumOfJob_ByAge_FiveProvinces.convertCategoricalToNumericValue(0)
testing_df_NumOfJob_ByAge_FiveProvinces = FiveProvinceAnalysis(testing_df_NumOfJob_ByAge_Provinces, pd, np, pp)
testing_df_NumOfJob_ByAge_FiveProvinces.convertCategoricalToNumericValue(0)
testing_df_NumOfJob_ByAge_FiveProvinces.print_One_Hot_Encoded_Data_Info()
testing_df_NumOfJob_ByAge_FiveProvinces.print_Unique('Age_group')

training_df_NumOfJob_ByGender_FiveProvinces = FiveProvinceAnalysis(training_df_NumOfJob_ByGender_Provinces, pd, np, pp)
training_df_NumOfJob_ByGender_FiveProvinces.convertCategoricalToNumericValue(1)
testing_df_NumOfJob_ByGender_FiveProvinces = FiveProvinceAnalysis(testing_df_NumOfJob_ByGender_Provinces, pd, np, pp)
testing_df_NumOfJob_ByGender_FiveProvinces.convertCategoricalToNumericValue(1)
testing_df_NumOfJob_ByGender_FiveProvinces.print_One_Hot_Encoded_Data_Info()
testing_df_NumOfJob_ByGender_FiveProvinces.print_Unique('Gender_group')

training_df_NumOfJob_ByEducation_FiveProvinces = FiveProvinceAnalysis(training_df_NumOfJob_ByEducation_Provinces, pd, np, pp)
training_df_NumOfJob_ByEducation_FiveProvinces.convertCategoricalToNumericValue(2)
testing_df_NumOfJob_ByEducation_FiveProvinces = FiveProvinceAnalysis(testing_df_NumOfJob_ByEducation_Provinces, pd, np, pp)
testing_df_NumOfJob_ByEducation_FiveProvinces.convertCategoricalToNumericValue(2)
testing_df_NumOfJob_ByEducation_FiveProvinces.print_One_Hot_Encoded_Data_Info()
testing_df_NumOfJob_ByEducation_FiveProvinces.print_Unique('Education_group')

training_df_NumOfJob_ByImmigrant_FiveProvinces = FiveProvinceAnalysis(training_df_NumOfJob_ByImmigrant_Provinces, pd, np, pp)
training_df_NumOfJob_ByImmigrant_FiveProvinces.convertCategoricalToNumericValue(3)
testing_df_NumOfJob_ByImmigrant_FiveProvinces = FiveProvinceAnalysis(testing_df_NumOfJob_ByImmigrant_Provinces, pd, np, pp)
testing_df_NumOfJob_ByImmigrant_FiveProvinces.convertCategoricalToNumericValue(3)
testing_df_NumOfJob_ByImmigrant_FiveProvinces.print_One_Hot_Encoded_Data_Info()
testing_df_NumOfJob_ByImmigrant_FiveProvinces.print_Unique('Immigrant_status')

# training_df_NumOfJobs_ByIndigenous_FiveProvinces = FiveProvinceAnalysis(training_df_NumOfJobs_ByIndigenous_Provinces, pd, np, pp)
# training_df_NumOfJobs_ByImmigrant_FiveProvinces.print_One_Hot_Encoded_Data_Info()
# testing_df_NumOfJobs_ByIndigenous_FiveProvinces = FiveProvinceAnalysis(training_df_NumOfJobs_ByIndigenous_Provinces, pd, np, pp)
# testing_df_NumOfJobs_ByImmigrant_FiveProvinces.print_One_Hot_Encoded_Data_Info()

# %%
training_df_WagesAndSalaries_ByAge_FiveProvinces = FiveProvinceAnalysis(training_df_WagesAndSalaries_ByAge_Provinces, pd, np, pp)
training_df_WagesAndSalaries_ByAge_FiveProvinces.convertCategoricalToNumericValue(0)
testing_df_WagesAndSalaries_ByAge_FiveProvinces = FiveProvinceAnalysis(testing_df_WagesAndSalaries_ByAge_Provinces, pd, np, pp)
testing_df_WagesAndSalaries_ByAge_FiveProvinces.convertCategoricalToNumericValue(0)
testing_df_WagesAndSalaries_ByAge_FiveProvinces.print_One_Hot_Encoded_Data_Info()
testing_df_WagesAndSalaries_ByAge_FiveProvinces.print_Unique('Age_group')

training_df_WagesAndSalaries_ByGender_FiveProvinces = FiveProvinceAnalysis(training_df_WagesAndSalaries_ByGender_Provinces, pd, np, pp)
training_df_WagesAndSalaries_ByGender_FiveProvinces.convertCategoricalToNumericValue(1)
testing_df_WagesAndSalaries_ByGender_FiveProvinces = FiveProvinceAnalysis(testing_df_WagesAndSalaries_ByGender_Provinces, pd, np, pp)
testing_df_WagesAndSalaries_ByGender_FiveProvinces.convertCategoricalToNumericValue(1)
testing_df_WagesAndSalaries_ByGender_FiveProvinces.print_One_Hot_Encoded_Data_Info()
testing_df_WagesAndSalaries_ByGender_FiveProvinces.print_Unique('Gender_group')

training_df_WagesAndSalaries_ByEducation_FiveProvinces = FiveProvinceAnalysis(training_df_WagesAndSalaries_ByEducation_Provinces, pd, np, pp)
training_df_WagesAndSalaries_ByEducation_FiveProvinces.convertCategoricalToNumericValue(2)
testing_df_WagesAndSalaries_ByEducation_FiveProvinces = FiveProvinceAnalysis(testing_df_WagesAndSalaries_ByEducation_Provinces, pd, np, pp)
testing_df_WagesAndSalaries_ByEducation_FiveProvinces.convertCategoricalToNumericValue(2)
testing_df_WagesAndSalaries_ByEducation_FiveProvinces.print_One_Hot_Encoded_Data_Info()
testing_df_WagesAndSalaries_ByEducation_FiveProvinces.print_Unique('Education_group')

training_df_WagesAndSalaries_ByImmigrant_FiveProvinces = FiveProvinceAnalysis(training_df_WagesAndSalaries_ByImmigrant_Provinces, pd, np, pp)
training_df_WagesAndSalaries_ByImmigrant_FiveProvinces.convertCategoricalToNumericValue(3)
testing_df_WagesAndSalaries_ByImmigrant_FiveProvinces = FiveProvinceAnalysis(testing_df_WagesAndSalaries_ByImmigrant_Provinces, pd, np, pp)
testing_df_WagesAndSalaries_ByImmigrant_FiveProvinces.convertCategoricalToNumericValue(3)
testing_df_WagesAndSalaries_ByImmigrant_FiveProvinces.print_One_Hot_Encoded_Data_Info()
testing_df_WagesAndSalaries_ByImmigrant_FiveProvinces.print_Unique('Immigrant_status')

# training_df_WagesAndSalaries_ByIndigenous_FiveProvince = FiveProvinceAnalysis(training_df_WagesAndSalaries_ByIndigenous_Provinces, pd, np, pp)
# training_df_WagesAndSalaries_ByImmigrant_FiveProvince.print_One_Hot_Encoded_Data_Info()
# testing_df_WagesAndSalaries_ByIndigenous_FiveProvince = FiveProvinceAnalysis(training_df_WagesAndSalaries_ByIndigenous_Provinces, pd, np, pp)

# %% [markdown]
# <h3> Final part - will be performed in other notebook files </h3>

# %% [markdown]
# Saving into CSV files to replay this result in case needed.

# %%
# Save the dataframe to a CSV file

training_df_AvgAnnHrsWrk_ByAge.to_csv('Result_By_Provinces/training_df_AvgAnnHrsWrk_ByAge.csv', index=False) # Average annual hours worked
testing_df_AvgAnnHrsWrk_ByAge.to_csv('Result_By_Provinces/testing_df_AvgAnnHrsWrk_ByAge.csv', index=False)

training_df_AvgAnnHrsWrk_ByGender.to_csv('Result_By_Provinces/training_df_AvgAnnHrsWrk_ByGender.csv', index=False) # Average annual hours worked
testing_df_AvgAnnHrsWrk_ByGender.to_csv('Result_By_Provinces/testing_df_AvgAnnHrsWrk_ByGender.csv', index=False)

training_df_AvgAnnHrsWrk_ByEducation.to_csv('Result_By_Provinces/training_df_AvgAnnHrsWrk_ByEducation.csv', index=False) # Average annual hours worked
testing_df_AvgAnnHrsWrk_ByEducation.to_csv('Result_By_Provinces/testing_df_AvgAnnHrsWrk_ByEducation.csv', index=False)

training_df_AvgAnnHrsWrk_ByImmigrant.to_csv('Result_By_Provinces/training_df_AvgAnnHrsWrk_ByImmigrant.csv', index=False) # Average annual hours worked
testing_df_AvgAnnHrsWrk_ByImmigrant.to_csv('Result_By_Provinces/testing_df_AvgAnnHrsWrk_ByImmigrant.csv', index=False)
# End of Loop and will start next one or end here.

# %%
training_df_AvgAnnWages_ByAge.to_csv('Result_By_Provinces/training_df_AvgAnnWages_ByAge.csv', index=False) # Average annual hours worked
testing_df_AvgAnnWages_ByAge.to_csv('Result_By_Provinces/testing_df_AvgAnnWages_ByAge.csv', index=False)

training_df_AvgAnnWages_ByGender.to_csv('Result_By_Provinces/training_df_AvgAnnWages_ByGender.csv', index=False) # Average annual hours worked
testing_df_AvgAnnWages_ByGender.to_csv('Result_By_Provinces/testing_df_AvgAnnWages_ByGender.csv', index=False)

training_df_AvgAnnWages_ByEducation.to_csv('Result_By_Provinces/training_df_AvgAnnWages_ByEducation.csv', index=False) # Average annual hours worked
testing_df_AvgAnnWages_ByEducation.to_csv('Result_By_Provinces/testing_df_AvgAnnWages_ByEducation.csv', index=False)

training_df_AvgAnnWages_ByImmigrant.to_csv('Result_By_Provinces/training_df_AvgAnnWages_ByImmigrant.csv', index=False) # Average annual hours worked
testing_df_AvgAnnWages_ByImmigrant.to_csv('Result_By_Provinces/testing_df_AvgAnnWages_ByImmigrant.csv', index=False)
# End of Loop and will start next one or end here.

# %%
training_df_AvgHrsWages_ByAge.to_csv('Result_By_Provinces/training_df_AvgHrsWages_ByAge.csv', index=False) # Average annual hours worked
testing_df_AvgHrsWages_ByAge.to_csv('Result_By_Provinces/testing_df_AvgHrsWages_ByAge.csv', index=False)

training_df_AvgHrsWages_ByGender.to_csv('Result_By_Provinces/training_df_AvgHrsWages_ByGender.csv', index=False) # Average annual hours worked
testing_df_AvgHrsWages_ByGender.to_csv('Result_By_Provinces/testing_df_AvgHrsWages_ByGender.csv', index=False)

training_df_AvgHrsWages_ByEducation.to_csv('Result_By_Provinces/training_df_AvgHrsWages_ByEducation.csv', index=False) # Average annual hours worked
testing_df_AvgHrsWages_ByEducation.to_csv('Result_By_Provinces/testing_df_AvgHrsWages_ByEducation.csv', index=False)

training_df_AvgHrsWages_ByImmigrant.to_csv('Result_By_Provinces/training_df_AvgHrsWages_ByImmigrant.csv', index=False) # Average annual hours worked
testing_df_AvgHrsWages_ByImmigrant.to_csv('Result_By_Provinces/testing_df_AvgHrsWages_ByImmigrant.csv', index=False)
# End of Loop and will start next one or end here.

# %%
training_df_AvgWeekHrsWrked_ByAge.to_csv('Result_By_Provinces/training_df_AvgWeekHrsWrked_ByAge.csv', index=False) # Average annual hours worked
testing_df_AvgWeekHrsWrked_ByAge.to_csv('Result_By_Provinces/testing_df_AvgWeekHrsWrked_ByAge.csv', index=False)

training_df_AvgWeekHrsWrked_ByGender.to_csv('Result_By_Provinces/training_df_AvgWeekHrsWrked_ByGender.csv', index=False) # Average annual hours worked
testing_df_AvgWeekHrsWrked_ByGender.to_csv('Result_By_Provinces/testing_df_AvgWeekHrsWrked_ByGender.csv', index=False)

training_df_AvgWeekHrsWrked_ByEducation.to_csv('Result_By_Provinces/training_df_AvgWeekHrsWrked_ByEducation.csv', index=False) # Average annual hours worked
testing_df_AvgWeekHrsWrked_ByEducation.to_csv('Result_By_Provinces/testing_df_AvgWeekHrsWrked_ByEducation.csv', index=False)

training_df_AvgWeekHrsWrked_ByImmigrant.to_csv('Result_By_Provinces/training_df_AvgWeekHrsWrked_ByImmigrant.csv', index=False) # Average annual hours worked
testing_df_AvgWeekHrsWrked_ByImmigrant.to_csv('Result_By_Provinces/testing_df_AvgWeekHrsWrked_ByImmigrant.csv', index=False)
# End of Loop and will start next one or end here.

# %%
training_df_Hrs_Wrked_ByAge.to_csv('Result_By_Provinces/training_df_Hrs_Wrked_ByAge.csv', index=False) # Average annual hours worked
testing_df_Hrs_Wrked_ByAge.to_csv('Result_By_Provinces/testing_df_Hrs_Wrked_ByAge.csv', index=False)

training_df_Hrs_Wrked_ByGender.to_csv('Result_By_Provinces/training_df_Hrs_Wrked_ByGender.csv', index=False) # Average annual hours worked
testing_df_Hrs_Wrked_ByGender.to_csv('Result_By_Provinces/testing_df_Hrs_Wrked_ByGender.csv', index=False)

training_df_Hrs_Wrked_ByEducation.to_csv('Result_By_Provinces/training_df_Hrs_Wrked_ByEducation.csv', index=False) # Average annual hours worked
testing_df_Hrs_Wrked_ByEducation.to_csv('Result_By_Provinces/testing_df_Hrs_Wrked_ByEducation.csv', index=False)

training_df_Hrs_Wrked_ByImmigrant.to_csv('Result_By_Provinces/training_df_Hrs_Wrked_ByImmigrant.csv', index=False) # Average annual hours worked
testing_df_Hrs_Wrked_ByImmigrant.to_csv('Result_By_Provinces/testing_df_Hrs_Wrked_ByImmigrant.csv', index=False)
# End of Loop and will start next one or end here.

# %%
training_df_NumOfJob_ByAge.to_csv('Result_By_Provinces/training_df_NumOfJob_ByAge.csv', index=False) # Average annual hours worked
testing_df_NumOfJob_ByAge.to_csv('Result_By_Provinces/testing_df_NumOfJob_ByAge.csv', index=False)

training_df_NumOfJob_ByGender.to_csv('Result_By_Provinces/training_df_NumOfJob_ByGender.csv', index=False) # Average annual hours worked
testing_df_NumOfJob_ByGender.to_csv('Result_By_Provinces/testing_df_NumOfJob_ByGender.csv', index=False)

training_df_NumOfJob_ByEducation.to_csv('Result_By_Provinces/training_df_NumOfJob_ByEducation.csv', index=False) # Average annual hours worked
testing_df_NumOfJob_ByEducation.to_csv('Result_By_Provinces/testing_df_NumOfJob_ByEducation.csv', index=False)

training_df_NumOfJob_ByImmigrant.to_csv('Result_By_Provinces/training_df_NumOfJob_ByImmigrant.csv', index=False) # Average annual hours worked
testing_df_NumOfJob_ByImmigrant.to_csv('Result_By_Provinces/testing_df_NumOfJob_ByImmigrant.csv', index=False)
# End of Loop and will start next one or end here.

# %%
training_df_WagesAndSalaries_ByAge.to_csv('Result_By_Provinces/training_df_WagesAndSalaries_ByAge.csv', index=False) # Average annual hours worked
testing_df_WagesAndSalaries_ByAge.to_csv('Result_By_Provinces/testing_df_WagesAndSalaries_ByAge.csv', index=False)

training_df_WagesAndSalaries_ByGender.to_csv('Result_By_Provinces/training_df_WagesAndSalaries_ByGender.csv', index=False) # Average annual hours worked
testing_df_WagesAndSalaries_ByGender.to_csv('Result_By_Provinces/testing_df_WagesAndSalaries_ByGender.csv', index=False)

training_df_WagesAndSalaries_ByEducation.to_csv('Result_By_Provinces/training_df_WagesAndSalaries_ByEducation.csv', index=False) # Average annual hours worked
testing_df_WagesAndSalaries_ByEducation.to_csv('Result_By_Provinces/testing_df_WagesAndSalaries_ByEducation.csv', index=False)

training_df_WagesAndSalaries_ByImmigrant.to_csv('Result_By_Provinces/training_df_WagesAndSalaries_ByImmigrant.csv', index=False) # Average annual hours worked
testing_df_WagesAndSalaries_ByImmigrant.to_csv('Result_By_Provinces/testing_df_WagesAndSalaries_ByImmigrant.csv', index=False)
# End of Loop and will start next one or end here.

# %% [markdown]
# Directory that deal with the final result.

# %%
CreatedTheFile = toOrganizedOutputFiles('Final_Result')

# %%
# Result of my final analysis.
# Commented because it is not needed. Further analysis contain inside Final_Result/'Portion_Technical_Report_Final_Select.ipynb'
# Also this code required to run inside 'Final_Result' directory

# testing_df_AvgAnnHrsWrk_ByAge_FiveProvinces.print_AnalysisResult_ByCharacteristics()
# testing_df_AvgAnnHrsWrk_ByGender_FiveProvinces.print_AnalysisResult_ByCharacteristics()
# testing_df_AvgAnnHrsWrk_ByEducation_FiveProvinces.print_AnalysisResult_ByCharacteristics()
# testing_df_AvgAnnHrsWrk_ByImmigrant_FiveProvinces.print_AnalysisResult_ByCharacteristics()

# testing_df_AvgAnnWages_ByAge_FiveProvinces.print_AnalysisResult_ByCharacteristics()
# testing_df_AvgAnnWages_ByGender_FiveProvinces.print_AnalysisResult_ByCharacteristics()
# testing_df_AvgAnnWages_ByEducation_FiveProvinces.print_AnalysisResult_ByCharacteristics()
# testing_df_AvgAnnWages_ByImmigrant_FiveProvinces.print_AnalysisResult_ByCharacteristics()

# testing_df_AvgWeekHrsWrked_ByAge_FiveProvinces.print_AnalysisResult_ByCharacteristics()
# testing_df_AvgWeekHrsWrked_ByGender_FiveProvinces.print_AnalysisResult_ByCharacteristics()
# testing_df_AvgWeekHrsWrked_ByEducation_FiveProvinces.print_AnalysisResult_ByCharacteristics()
# testing_df_AvgWeekHrsWrked_ByImmigrant_FiveProvinces.print_AnalysisResult_ByCharacteristics()

# testing_df_Hrs_Wrked_ByAge_FiveProvinces.print_AnalysisResult_ByCharacteristics()
# testing_df_Hrs_Wrked_ByGender_FiveProvinces.print_AnalysisResult_ByCharacteristics()
# testing_df_Hrs_Wrked_ByEducation_FiveProvinces.print_AnalysisResult_ByCharacteristics()
# testing_df_Hrs_Wrked_ByImmigrant_FiveProvinces.print_AnalysisResult_ByCharacteristics()

# testing_df_NumOfJob_ByAge_FiveProvinces.print_AnalysisResult_ByCharacteristics()
# testing_df_NumOfJob_ByGender_FiveProvinces.print_AnalysisResult_ByCharacteristics()
# testing_df_NumOfJob_ByEducation_FiveProvinces.print_AnalysisResult_ByCharacteristics()
# testing_df_NumOfJob_ByImmigrant_FiveProvinces.print_AnalysisResult_ByCharacteristics()


# testing_df_WagesAndSalaries_ByAge_FiveProvinces.print_AnalysisResult_ByCharacteristics()
# testing_df_WagesAndSalaries_ByGender_FiveProvinces.print_AnalysisResult_ByCharacteristics()
# testing_df_WagesAndSalaries_ByEducation_FiveProvinces.print_AnalysisResult_ByCharacteristics()
# testing_df_WagesAndSalaries_ByImmigrant_FiveProvinces.print_AnalysisResult_ByCharacteristics()

# testing_df_WagesAndSalaries_ByAge_FiveProvinces.print_AnalysisResult_ByCharacteristics()
# testing_df_WagesAndSalaries_ByGender_FiveProvinces.print_AnalysisResult_ByCharacteristics()
# testing_df_WagesAndSalaries_ByEducation_FiveProvinces.print_AnalysisResult_ByCharacteristics()
# testing_df_WagesAndSalaries_ByImmigrant_FiveProvinces.print_AnalysisResult_ByCharacteristics()



# %%
print("Further analysis contain inside Final_Result/\'Portion_Technical_Report_Final_Select.ipynb\'")
print("Also this code required to run inside 'Final_Result' directory")

# %% [markdown]
# Saving Final Result dataset into csv files. Starting from training dataset.

# %%
# Backing up modified training dataset with numeric orders with five Provinces

file_training_df_output_df_AvgAnnHrsWrk_ByAge = training_df_AvgAnnHrsWrk_ByAge_FiveProvinces.output_One_Hot_Encoded_Data()
file_training_df_output_df_AvgAnnHrsWrk_ByEducation = training_df_AvgAnnHrsWrk_ByEducation_FiveProvinces.output_One_Hot_Encoded_Data()
file_training_df_output_df_AvgAnnHrsWrk_ByGender = training_df_AvgAnnHrsWrk_ByGender_FiveProvinces.output_One_Hot_Encoded_Data()
file_training_df_output_df_AvgAnnHrsWrk_ByImmigrant = training_df_AvgAnnHrsWrk_ByImmigrant_FiveProvinces.output_One_Hot_Encoded_Data()

file_training_df_output_df_AvgAnnWages_ByAge = training_df_AvgAnnWages_ByAge_FiveProvinces.output_One_Hot_Encoded_Data()
file_training_df_output_df_AvgAnnWages_ByEducation = training_df_AvgAnnWages_ByEducation_FiveProvinces.output_One_Hot_Encoded_Data()
file_training_df_output_df_AvgAnnWages_ByGender = training_df_AvgAnnWages_ByGender_FiveProvinces.output_One_Hot_Encoded_Data()
file_training_df_output_df_AvgAnnWages_ByImmigrant = training_df_AvgAnnWages_ByImmigrant_FiveProvinces.output_One_Hot_Encoded_Data()

file_training_df_output_df_AvgHrsWages_ByAge = training_df_AvgHrsWages_ByAge_FiveProvinces.output_One_Hot_Encoded_Data()
file_training_df_output_df_AvgHrsWages_ByGender = training_df_AvgHrsWages_ByGender_FiveProvinces.output_One_Hot_Encoded_Data()
file_training_df_output_df_AvgHrsWages_ByEducation = training_df_AvgHrsWages_ByEducation_FiveProvinces.output_One_Hot_Encoded_Data()
file_training_df_output_df_AvgHrsWages_ByImmigrant = training_df_AvgHrsWages_ByImmigrant_FiveProvinces.output_One_Hot_Encoded_Data()

file_training_df_output_df_AvgWeekHrsWrked_ByAge = training_df_AvgWeekHrsWrked_ByAge_FiveProvinces.output_One_Hot_Encoded_Data()
file_training_df_output_df_AvgWeekHrsWrked_ByEducation = training_df_AvgWeekHrsWrked_ByEducation_FiveProvinces.output_One_Hot_Encoded_Data()
file_training_df_output_df_AvgWeekHrsWrked_ByGender = training_df_AvgWeekHrsWrked_ByGender_FiveProvinces.output_One_Hot_Encoded_Data()
file_training_df_output_df_AvgWeekHrsWrked_ByImmigrant = training_df_AvgWeekHrsWrked_ByImmigrant_FiveProvinces.output_One_Hot_Encoded_Data()

file_training_df_output_df_Hrs_Wrked_ByAge = training_df_Hrs_Wrked_ByAge_FiveProvinces.output_One_Hot_Encoded_Data()
file_training_df_output_df_Hrs_Wrked_ByEducation = training_df_Hrs_Wrked_ByEducation_FiveProvinces.output_One_Hot_Encoded_Data()
file_training_df_output_df_Hrs_Wrked_ByGender = training_df_Hrs_Wrked_ByGender_FiveProvinces.output_One_Hot_Encoded_Data()
file_training_df_output_df_Hrs_Wrked_ByImmigrant = training_df_Hrs_Wrked_ByImmigrant_FiveProvinces.output_One_Hot_Encoded_Data()

file_training_df_output_df_NumOfJob_ByAge = training_df_NumOfJob_ByAge_FiveProvinces.output_One_Hot_Encoded_Data()
file_training_df_output_df_NumOfJob_ByEducation = training_df_NumOfJob_ByEducation_FiveProvinces.output_One_Hot_Encoded_Data()
file_training_df_output_df_NumOfJob_ByGender = training_df_NumOfJob_ByGender_FiveProvinces.output_One_Hot_Encoded_Data()
file_training_df_output_df_NumOfJob_ByImmigrant = training_df_NumOfJob_ByImmigrant_FiveProvinces.output_One_Hot_Encoded_Data()

file_training_df_output_df_WagesAndSalaries_ByAge = training_df_WagesAndSalaries_ByAge_FiveProvinces.output_One_Hot_Encoded_Data()
file_training_df_output_df_WagesAndSalaries_ByEducation = training_df_WagesAndSalaries_ByEducation_FiveProvinces.output_One_Hot_Encoded_Data()
file_training_df_output_df_WagesAndSalaries_ByGender = training_df_WagesAndSalaries_ByGender_FiveProvinces.output_One_Hot_Encoded_Data()
file_training_df_output_df_WagesAndSalaries_ByImmigrant = training_df_WagesAndSalaries_ByImmigrant_FiveProvinces.output_One_Hot_Encoded_Data()

# %%
# Backing up original training dataset with five Provinces
# Skipping saving into csv files for this.

# file_training_df_output_df_AvgAnnHrsWrk_ByAge_Original = training_df_AvgAnnHrsWrk_ByAge_FiveProvinces.output_Original_Data()
# file_training_df_output_df_AvgAnnHrsWrk_ByEducation_Original = training_df_AvgAnnHrsWrk_ByEducation_FiveProvinces.output_Original_Data()
# file_training_df_output_df_AvgAnnHrsWrk_ByGender_Original = training_df_AvgAnnHrsWrk_ByGender_FiveProvinces.output_Original_Data()
# file_training_df_output_df_AvgAnnHrsWrk_ByImmigrant_Original = training_df_AvgAnnHrsWrk_ByImmigrant_FiveProvinces.output_Original_Data()

# file_training_df_output_df_AvgAnnWages_ByAge_Original = training_df_AvgAnnWages_ByAge_FiveProvinces.output_Original_Data()
# file_training_df_output_df_AvgAnnWages_ByEducation_Original = training_df_AvgAnnWages_ByEducation_FiveProvinces.output_Original_Data()
# file_training_df_output_df_AvgAnnWages_ByGender_Original = training_df_AvgAnnWages_ByGender_FiveProvinces.output_Original_Data()
# file_training_df_output_df_AvgAnnWages_ByImmigrant_Original = training_df_AvgAnnWages_ByImmigrant_FiveProvinces.output_Original_Data()

# file_training_df_output_df_AvgHrsWages_ByAge_Original = training_df_AvgHrsWages_ByAge_FiveProvinces.output_Original_Data()
# file_training_df_output_df_AvgHrsWages_ByGender_Original = training_df_AvgHrsWages_ByGender_FiveProvinces.output_Original_Data()
# file_training_df_output_df_AvgHrsWages_ByEducation_Original = training_df_AvgHrsWages_ByEducation_FiveProvinces.output_Original_Data()
# file_training_df_output_df_AvgHrsWages_ByImmigrant_Original = training_df_AvgHrsWages_ByImmigrant_FiveProvinces.output_Original_Data()

# file_training_df_output_df_AvgWeekHrsWrked_ByAge_Original = training_df_AvgWeekHrsWrked_ByAge_FiveProvinces.output_Original_Data()
# file_training_df_output_df_AvgWeekHrsWrked_ByEducation_Original = training_df_AvgWeekHrsWrked_ByEducation_FiveProvinces.output_Original_Data()
# file_training_df_output_df_AvgWeekHrsWrked_ByGender_Original = training_df_AvgWeekHrsWrked_ByGender_FiveProvinces.output_Original_Data()
# file_training_df_output_df_AvgWeekHrsWrked_ByImmigrant_Original = training_df_AvgWeekHrsWrked_ByImmigrant_FiveProvinces.output_Original_Data()

# file_training_df_output_df_Hrs_Wrked_ByAge_Original = training_df_Hrs_Wrked_ByAge_FiveProvinces.output_Original_Data()
# file_training_df_output_df_Hrs_Wrked_ByEducation_Original = training_df_Hrs_Wrked_ByEducation_FiveProvinces.output_Original_Data()
# file_training_df_output_df_Hrs_Wrked_ByGender_Original = training_df_Hrs_Wrked_ByGender_FiveProvinces.output_Original_Data()
# file_training_df_output_df_Hrs_Wrked_ByImmigrant_Original = training_df_Hrs_Wrked_ByImmigrant_FiveProvinces.output_Original_Data()

# file_training_df_output_df_NumOfJob_ByAge_Original = training_df_NumOfJob_ByAge_FiveProvinces.output_Original_Data()
# file_training_df_output_df_NumOfJob_ByEducation_Original = training_df_NumOfJob_ByEducation_FiveProvinces.output_Original_Data()
# file_training_df_output_df_NumOfJob_ByGender_Original = training_df_NumOfJob_ByGender_FiveProvinces.output_Original_Data()
# file_training_df_output_df_NumOfJob_ByImmigrant_Original = training_df_NumOfJob_ByImmigrant_FiveProvinces.output_Original_Data()

# file_training_df_output_df_WagesAndSalaries_ByAge_Original = training_df_WagesAndSalaries_ByAge_FiveProvinces.output_Original_Data()
# file_training_df_output_df_WagesAndSalaries_ByEducation_Original = training_df_WagesAndSalaries_ByEducation_FiveProvinces.output_Original_Data()
# file_training_df_output_df_WagesAndSalaries_ByGender_Original = training_df_WagesAndSalaries_ByGender_FiveProvinces.output_Original_Data()
# file_training_df_output_df_WagesAndSalaries_ByImmigrant_Original = training_df_WagesAndSalaries_ByImmigrant_FiveProvinces.output_Original_Data()

# %%
# Save the dataframe to a CSV file

file_training_df_output_df_AvgAnnHrsWrk_ByAge.to_csv('Final_Result/final_training_df_output_df_AvgAnnHrsWrk_ByAge.csv', index=False)
file_training_df_output_df_AvgAnnHrsWrk_ByEducation.to_csv('Final_Result/final_training_df_output_df_AvgAnnHrsWrk_ByEducation.csv', index=False)
file_training_df_output_df_AvgAnnHrsWrk_ByGender.to_csv('Final_Result/final_training_df_output_df_AvgAnnHrsWrk_ByGender.csv', index=False)
file_training_df_output_df_AvgAnnHrsWrk_ByImmigrant.to_csv('Final_Result/final_training_df_output_df_AvgAnnHrsWrk_ByImmigrant.csv', index=False)

file_training_df_output_df_AvgAnnWages_ByAge.to_csv('Final_Result/final_training_df_output_df_AvgAnnWages_ByAge.csv', index=False)
file_training_df_output_df_AvgAnnWages_ByEducation.to_csv('Final_Result/final_training_df_output_df_AvgAnnWages_ByEducation.csv', index=False)
file_training_df_output_df_AvgAnnWages_ByGender.to_csv('Final_Result/final_training_df_output_df_AvgAnnWages_ByGender.csv', index=False)
file_training_df_output_df_AvgAnnWages_ByImmigrant.to_csv('Final_Result/final_training_df_output_df_AvgAnnWages_ByImmigrant.csv', index=False)

file_training_df_output_df_AvgHrsWages_ByAge.to_csv('Final_Result/final_training_df_output_df_AvgHrsWages_ByAge.csv', index=False)
file_training_df_output_df_AvgHrsWages_ByGender.to_csv('Final_Result/final_training_df_output_df_AvgHrsWages_ByGender.csv', index=False)
file_training_df_output_df_AvgHrsWages_ByEducation.to_csv('Final_Result/final_training_df_output_df_AvgHrsWages_ByEducation.csv', index=False)
file_training_df_output_df_AvgHrsWages_ByImmigrant.to_csv('Final_Result/final_training_df_output_df_AvgHrsWages_ByImmigrant.csv', index=False)

file_training_df_output_df_AvgWeekHrsWrked_ByAge.to_csv('Final_Result/final_training_df_output_df_AvgWeekHrsWrked_ByAge.csv', index=False)
file_training_df_output_df_AvgWeekHrsWrked_ByEducation.to_csv('Final_Result/final_training_df_output_df_AvgWeekHrsWrked_ByEducation.csv', index=False)
file_training_df_output_df_AvgWeekHrsWrked_ByGender.to_csv('Final_Result/final_training_df_output_df_AvgWeekHrsWrked_ByGender.csv', index=False)
file_training_df_output_df_AvgWeekHrsWrked_ByImmigrant.to_csv('Final_Result/final_training_df_output_df_AvgWeekHrsWrked_ByImmigrant.csv', index=False)

file_training_df_output_df_Hrs_Wrked_ByAge.to_csv('Final_Result/final_training_df_output_df_Hrs_Wrked_ByAge.csv', index=False)
file_training_df_output_df_Hrs_Wrked_ByEducation.to_csv('Final_Result/final_training_df_output_df_Hrs_Wrked_ByEducation.csv', index=False)
file_training_df_output_df_Hrs_Wrked_ByGender.to_csv('Final_Result/final_training_df_output_df_Hrs_Wrked_ByGender.csv', index=False)
file_training_df_output_df_Hrs_Wrked_ByImmigrant.to_csv('Final_Result/final_training_df_output_df_Hrs_Wrked_ByImmigrant.csv', index=False)

file_training_df_output_df_NumOfJob_ByAge.to_csv('Final_Result/final_training_df_output_df_NumOfJob_ByAge.csv', index=False)
file_training_df_output_df_NumOfJob_ByEducation.to_csv('Final_Result/final_training_df_output_df_NumOfJob_ByEducation.csv', index=False)
file_training_df_output_df_NumOfJob_ByGender.to_csv('Final_Result/final_training_df_output_df_NumOfJob_ByGender.csv', index=False)
file_training_df_output_df_NumOfJob_ByImmigrant.to_csv('Final_Result/final_training_df_output_df_NumOfJob_ByImmigrant.csv', index=False)

file_training_df_output_df_WagesAndSalaries_ByAge.to_csv('Final_Result/final_training_df_output_df_WagesAndSalaries_ByAge.csv', index=False)
file_training_df_output_df_WagesAndSalaries_ByEducation.to_csv('Final_Result/final_training_df_output_df_WagesAndSalaries_ByEducation.csv', index=False)
file_training_df_output_df_WagesAndSalaries_ByGender.to_csv('Final_Result/final_training_df_output_df_WagesAndSalaries_ByGender.csv', index=False)
file_training_df_output_df_WagesAndSalaries_ByImmigrant.to_csv('Final_Result/final_training_df_output_df_WagesAndSalaries_ByImmigrant.csv', index=False)

# %% [markdown]
# Saving Final Result dataset into csv files. Next testing dataset.

# %%
# Backing up modified testing dataset dataset with numeric orders with five Provinces

file_testing_df_output_df_AvgAnnHrsWrk_ByAge = testing_df_AvgAnnHrsWrk_ByAge_FiveProvinces.output_One_Hot_Encoded_Data()
file_testing_df_output_df_AvgAnnHrsWrk_ByEducation = testing_df_AvgAnnHrsWrk_ByEducation_FiveProvinces.output_One_Hot_Encoded_Data()
file_testing_df_output_df_AvgAnnHrsWrk_ByGender = testing_df_AvgAnnHrsWrk_ByGender_FiveProvinces.output_One_Hot_Encoded_Data()
file_testing_df_output_df_AvgAnnHrsWrk_ByImmigrant = testing_df_AvgAnnHrsWrk_ByImmigrant_FiveProvinces.output_One_Hot_Encoded_Data()

file_testing_df_output_df_AvgAnnWages_ByAge = testing_df_AvgAnnWages_ByAge_FiveProvinces.output_One_Hot_Encoded_Data()
file_testing_df_output_df_AvgAnnWages_ByEducation = testing_df_AvgAnnWages_ByEducation_FiveProvinces.output_One_Hot_Encoded_Data()
file_testing_df_output_df_AvgAnnWages_ByGender = testing_df_AvgAnnWages_ByGender_FiveProvinces.output_One_Hot_Encoded_Data()
file_testing_df_output_df_AvgAnnWages_ByImmigrant = testing_df_AvgAnnWages_ByImmigrant_FiveProvinces.output_One_Hot_Encoded_Data()

file_testing_df_output_df_AvgHrsWages_ByAge = testing_df_AvgHrsWages_ByAge_FiveProvinces.output_One_Hot_Encoded_Data()
file_testing_df_output_df_AvgHrsWages_ByGender = testing_df_AvgHrsWages_ByGender_FiveProvinces.output_One_Hot_Encoded_Data()
file_testing_df_output_df_AvgHrsWages_ByEducation = testing_df_AvgHrsWages_ByEducation_FiveProvinces.output_One_Hot_Encoded_Data()
file_testing_df_output_df_AvgHrsWages_ByImmigrant = testing_df_AvgHrsWages_ByImmigrant_FiveProvinces.output_One_Hot_Encoded_Data()

file_testing_df_output_df_AvgWeekHrsWrked_ByAge = testing_df_AvgWeekHrsWrked_ByAge_FiveProvinces.output_One_Hot_Encoded_Data()
file_testing_df_output_df_AvgWeekHrsWrked_ByEducation = testing_df_AvgWeekHrsWrked_ByEducation_FiveProvinces.output_One_Hot_Encoded_Data()
file_testing_df_output_df_AvgWeekHrsWrked_ByGender = testing_df_AvgWeekHrsWrked_ByGender_FiveProvinces.output_One_Hot_Encoded_Data()
file_testing_df_output_df_AvgWeekHrsWrked_ByImmigrant = testing_df_AvgWeekHrsWrked_ByImmigrant_FiveProvinces.output_One_Hot_Encoded_Data()

file_testing_df_output_df_Hrs_Wrked_ByAge = testing_df_Hrs_Wrked_ByAge_FiveProvinces.output_One_Hot_Encoded_Data()
file_testing_df_output_df_Hrs_Wrked_ByEducation = testing_df_Hrs_Wrked_ByEducation_FiveProvinces.output_One_Hot_Encoded_Data()
file_testing_df_output_df_Hrs_Wrked_ByGender = testing_df_Hrs_Wrked_ByGender_FiveProvinces.output_One_Hot_Encoded_Data()
file_testing_df_output_df_Hrs_Wrked_ByImmigrant = testing_df_Hrs_Wrked_ByImmigrant_FiveProvinces.output_One_Hot_Encoded_Data()

file_testing_df_output_df_NumOfJob_ByAge = testing_df_NumOfJob_ByAge_FiveProvinces.output_One_Hot_Encoded_Data()
file_testing_df_output_df_NumOfJob_ByEducation = testing_df_NumOfJob_ByEducation_FiveProvinces.output_One_Hot_Encoded_Data()
file_testing_df_output_df_NumOfJob_ByGender = testing_df_NumOfJob_ByGender_FiveProvinces.output_One_Hot_Encoded_Data()
file_testing_df_output_df_NumOfJob_ByImmigrant = testing_df_NumOfJob_ByImmigrant_FiveProvinces.output_One_Hot_Encoded_Data()

file_testing_df_output_df_WagesAndSalaries_ByAge = testing_df_WagesAndSalaries_ByAge_FiveProvinces.output_One_Hot_Encoded_Data()
file_testing_df_output_df_WagesAndSalaries_ByEducation = testing_df_WagesAndSalaries_ByEducation_FiveProvinces.output_One_Hot_Encoded_Data()
file_testing_df_output_df_WagesAndSalaries_ByGender = testing_df_WagesAndSalaries_ByGender_FiveProvinces.output_One_Hot_Encoded_Data()
file_testing_df_output_df_WagesAndSalaries_ByImmigrant = testing_df_WagesAndSalaries_ByImmigrant_FiveProvinces.output_One_Hot_Encoded_Data()

# %%
# Backing up original testing dataset with five Provinces
# Skipping saving into csv files for this.

# file_testing_df_output_df_AvgAnnHrsWrk_ByAge_Original = testing_df_AvgAnnHrsWrk_ByAge_FiveProvinces.output_Original_Data()
# file_testing_df_output_df_AvgAnnHrsWrk_ByEducation_Original = testing_df_AvgAnnHrsWrk_ByEducation_FiveProvinces.output_Original_Data()
# file_testing_df_output_df_AvgAnnHrsWrk_ByGender_Original = testing_df_AvgAnnHrsWrk_ByGender_FiveProvinces.output_Original_Data()
# file_testing_df_output_df_AvgAnnHrsWrk_ByImmigrant_Original = testing_df_AvgAnnHrsWrk_ByImmigrant_FiveProvinces.output_Original_Data()

# file_testing_df_output_df_AvgAnnWages_ByAge_Original = testing_df_AvgAnnWages_ByAge_FiveProvinces.output_Original_Data()
# file_testing_df_output_df_AvgAnnWages_ByEducation_Original = testing_df_AvgAnnWages_ByEducation_FiveProvinces.output_Original_Data()
# file_testing_df_output_df_AvgAnnWages_ByGender_Original = testing_df_AvgAnnWages_ByGender_FiveProvinces.output_Original_Data()
# file_testing_df_output_df_AvgAnnWages_ByImmigrant_Original = testing_df_AvgAnnWages_ByImmigrant_FiveProvinces.output_Original_Data()

# file_testing_df_output_df_AvgHrsWages_ByAge_Original = testing_df_AvgHrsWages_ByAge_FiveProvinces.output_Original_Data()
# file_testing_df_output_df_AvgHrsWages_ByGender_Original = testing_df_AvgHrsWages_ByGender_FiveProvinces.output_Original_Data()
# file_testing_df_output_df_AvgHrsWages_ByEducation_Original = testing_df_AvgHrsWages_ByEducation_FiveProvinces.output_Original_Data()
# file_testing_df_output_df_AvgHrsWages_ByImmigrant_Original = testing_df_AvgHrsWages_ByImmigrant_FiveProvinces.output_Original_Data()

# file_testing_df_output_df_AvgWeekHrsWrked_ByAge_Original = testing_df_AvgWeekHrsWrked_ByAge_FiveProvinces.output_Original_Data()
# file_testing_df_output_df_AvgWeekHrsWrked_ByEducation_Original = testing_df_AvgWeekHrsWrked_ByEducation_FiveProvinces.output_Original_Data()
# file_testing_df_output_df_AvgWeekHrsWrked_ByGender_Original = testing_df_AvgWeekHrsWrked_ByGender_FiveProvinces.output_Original_Data()
# file_testing_df_output_df_AvgWeekHrsWrked_ByImmigrant_Original = testing_df_AvgWeekHrsWrked_ByImmigrant_FiveProvinces.output_Original_Data()

# file_testing_df_output_df_Hrs_Wrked_ByAge_Original = testing_df_Hrs_Wrked_ByAge_FiveProvinces.output_Original_Data()
# file_testing_df_output_df_Hrs_Wrked_ByEducation_Original = testing_df_Hrs_Wrked_ByEducation_FiveProvinces.output_Original_Data()
# file_testing_df_output_df_Hrs_Wrked_ByGender_Original = testing_df_Hrs_Wrked_ByGender_FiveProvinces.output_Original_Data()
# file_testing_df_output_df_Hrs_Wrked_ByImmigrant_Original = testing_df_Hrs_Wrked_ByImmigrant_FiveProvinces.output_Original_Data()

# file_testing_df_output_df_NumOfJob_ByAge_Original = testing_df_NumOfJob_ByAge_FiveProvinces.output_Original_Data()
# file_testing_df_output_df_NumOfJob_ByEducation_Original = testing_df_NumOfJob_ByEducation_FiveProvinces.output_Original_Data()
# file_testing_df_output_df_NumOfJob_ByGender_Original = testing_df_NumOfJob_ByGender_FiveProvinces.output_Original_Data()
# file_testing_df_output_df_NumOfJob_ByImmigrant_Original = testing_df_NumOfJob_ByImmigrant_FiveProvinces.output_Original_Data()

# file_testing_df_output_df_WagesAndSalaries_ByAge_Original = testing_df_WagesAndSalaries_ByAge_FiveProvinces.output_Original_Data()
# file_testing_df_output_df_WagesAndSalaries_ByEducation_Original = testing_df_WagesAndSalaries_ByEducation_FiveProvinces.output_Original_Data()
# file_testing_df_output_df_WagesAndSalaries_ByGender_Original = testing_df_WagesAndSalaries_ByGender_FiveProvinces.output_Original_Data()
# file_testing_df_output_df_WagesAndSalaries_ByImmigrant_Original = testing_df_WagesAndSalaries_ByImmigrant_FiveProvinces.output_Original_Data()

# %%
# Save the dataframe to a CSV file

file_testing_df_output_df_AvgAnnHrsWrk_ByAge.to_csv('Final_Result/final_testing_df_output_df_AvgAnnHrsWrk_ByAge.csv', index=False)
file_testing_df_output_df_AvgAnnHrsWrk_ByEducation.to_csv('Final_Result/final_testing_df_output_df_AvgAnnHrsWrk_ByEducation.csv', index=False)
file_testing_df_output_df_AvgAnnHrsWrk_ByGender.to_csv('Final_Result/final_testing_df_output_df_AvgAnnHrsWrk_ByGender.csv', index=False)
file_testing_df_output_df_AvgAnnHrsWrk_ByImmigrant.to_csv('Final_Result/final_testing_df_output_df_AvgAnnHrsWrk_ByImmigrant.csv', index=False)

file_testing_df_output_df_AvgAnnWages_ByAge.to_csv('Final_Result/final_testing_df_output_df_AvgAnnWages_ByAge.csv', index=False)
file_testing_df_output_df_AvgAnnWages_ByEducation.to_csv('Final_Result/final_testing_df_output_df_AvgAnnWages_ByEducation.csv', index=False)
file_testing_df_output_df_AvgAnnWages_ByGender.to_csv('Final_Result/final_testing_df_output_df_AvgAnnWages_ByGender.csv', index=False)
file_testing_df_output_df_AvgAnnWages_ByImmigrant.to_csv('Final_Result/final_testing_df_output_df_AvgAnnWages_ByImmigrant.csv', index=False)

file_testing_df_output_df_AvgHrsWages_ByAge.to_csv('Final_Result/final_testing_df_output_df_AvgHrsWages_ByAge.csv', index=False)
file_testing_df_output_df_AvgHrsWages_ByGender.to_csv('Final_Result/final_testing_df_output_df_AvgHrsWages_ByGender.csv', index=False)
file_testing_df_output_df_AvgHrsWages_ByEducation.to_csv('Final_Result/final_testing_df_output_df_AvgHrsWages_ByEducation.csv', index=False)
file_testing_df_output_df_AvgHrsWages_ByImmigrant.to_csv('Final_Result/final_testing_df_output_df_AvgHrsWages_ByImmigrant.csv', index=False)

file_testing_df_output_df_AvgWeekHrsWrked_ByAge.to_csv('Final_Result/final_testing_df_output_df_AvgWeekHrsWrked_ByAge.csv', index=False)
file_testing_df_output_df_AvgWeekHrsWrked_ByEducation.to_csv('Final_Result/final_testing_df_output_df_AvgWeekHrsWrked_ByEducation.csv', index=False)
file_testing_df_output_df_AvgWeekHrsWrked_ByGender.to_csv('Final_Result/final_testing_df_output_df_AvgWeekHrsWrked_ByGender.csv', index=False)
file_testing_df_output_df_AvgWeekHrsWrked_ByImmigrant.to_csv('Final_Result/final_testing_df_output_df_AvgWeekHrsWrked_ByImmigrant.csv', index=False)

file_testing_df_output_df_Hrs_Wrked_ByAge.to_csv('Final_Result/final_testing_df_output_df_Hrs_Wrked_ByAge.csv', index=False)
file_testing_df_output_df_Hrs_Wrked_ByEducation.to_csv('Final_Result/final_testing_df_output_df_Hrs_Wrked_ByEducation.csv', index=False)
file_testing_df_output_df_Hrs_Wrked_ByGender.to_csv('Final_Result/final_testing_df_output_df_Hrs_Wrked_ByGender.csv', index=False)
file_testing_df_output_df_Hrs_Wrked_ByImmigrant.to_csv('Final_Result/final_testing_df_output_df_Hrs_Wrked_ByImmigrant.csv', index=False)

file_testing_df_output_df_NumOfJob_ByAge.to_csv('Final_Result/final_testing_df_output_df_NumOfJob_ByAge.csv', index=False)
file_testing_df_output_df_NumOfJob_ByEducation.to_csv('Final_Result/final_testing_df_output_df_NumOfJob_ByEducation.csv', index=False)
file_testing_df_output_df_NumOfJob_ByGender.to_csv('Final_Result/final_testing_df_output_df_NumOfJob_ByGender.csv', index=False)
file_testing_df_output_df_NumOfJob_ByImmigrant.to_csv('Final_Result/final_testing_df_output_df_NumOfJob_ByImmigrant.csv', index=False)

file_testing_df_output_df_WagesAndSalaries_ByAge.to_csv('Final_Result/final_testing_df_output_df_WagesAndSalaries_ByAge.csv', index=False)
file_testing_df_output_df_WagesAndSalaries_ByEducation.to_csv('Final_Result/final_testing_df_output_df_WagesAndSalaries_ByEducation.csv', index=False)
file_testing_df_output_df_WagesAndSalaries_ByGender.to_csv('Final_Result/final_testing_df_output_df_WagesAndSalaries_ByGender.csv', index=False)
file_testing_df_output_df_WagesAndSalaries_ByImmigrant.to_csv('Final_Result/final_testing_df_output_df_WagesAndSalaries_ByImmigrant.csv', index=False)


### End of this section ###

# %% [markdown]
# End of the analysis, final result will be performed in different notebook.



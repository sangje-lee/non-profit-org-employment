# %% [markdown]
# <h3> Third part of the code </h3>

# %% [markdown]
# Running the main code required.

# %% [markdown]
# Portion of the code involved, including,<br />
# - Portion of finishing code that are splited based on year.<br />
# - Spliting by training and testing set.<br />
# (2010-2013 --> Not being used) (2014-2018 --> Training) (2019-2021 --> Testing)<br />
# - Histogram of each set.<br />
# - Chi Square analysis of each set.

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from fitter import Fitter, get_common_distributions, get_distributions
import ydata_profiling as pp  
from ydata_profiling import ProfileReport 
from scipy.stats import chi2_contingency

import warnings
import os

warnings.filterwarnings('ignore')

# %%
# If the dataset are missing for this components, it will give a error.

# df_AvgAnnHrsWrk # Average annual hours worked

df_indicators = ['df_AvgAnnHrsWrk_2013.csv', 'df_AvgAnnHrsWrk_2016.csv', 'df_AvgAnnHrsWrk_2019.csv',
                 'df_AvgAnnWages_2013.csv', 'df_AvgAnnWages_2016.csv', 'df_AvgAnnWages_2019.csv',
                 'df_AvgHrsWages_2013.csv', 'df_AvgHrsWages_2016.csv', 'df_AvgHrsWages_2019.csv',
                 'df_AvgWeekHrsWrked_2013.csv', 'df_AvgWeekHrsWrked_2016.csv', 'df_AvgWeekHrsWrked_2019.csv',
                 'df_Hrs_Wrked_2013.csv', 'df_Hrs_Wrked_2016.csv', 'df_Hrs_Wrked_2019.csv',
                 'df_NumOfJob_2013.csv', 'df_NumOfJob_2016.csv', 'df_NumOfJob_2019.csv',
                 'df_WagesAndSalaries_2013.csv', 'df_WagesAndSalaries_2016.csv', 'df_WagesAndSalaries_2019.csv'
                 ]

df_list = []

for x in df_indicators:
    if os.path.isfile(x):
        df_sorted_na = pd.read_csv(x)

        print(df_sorted_na.info())
        print(df_sorted_na.head(10))

        df_list.append(df_sorted_na)
    else:
        print("Run main code first before running this.")
        break

# %%
# If the code cannot run, it will give a error.

# df_AvgAnnHrsWrk_2013
df_AvgAnnHrsWrk_2013 = df_list[0]
df_AvgAnnHrsWrk_2016 = df_list[1]
df_AvgAnnHrsWrk_2019 = df_list[2]

# df_AvgAnnWages # Average annual wages and salaries
df_AvgAnnWages_2013 = df_list[3]
df_AvgAnnWages_2016 = df_list[4]
df_AvgAnnWages_2019 = df_list[5]

# df_AvgHrsWages # Average hourly wage
df_AvgHrsWages_2013 = df_list[6]
df_AvgHrsWages_2016 = df_list[7]
df_AvgHrsWages_2019 = df_list[8]

# df_AvgWeekHrsWrked # Average weekly hours worked
df_AvgWeekHrsWrked_2013 = df_list[9]
df_AvgWeekHrsWrked_2016 = df_list[10]
df_AvgWeekHrsWrked_2019 = df_list[11]

# df_Hrs_Wrked # Hours Worked
df_Hrs_Wrked_2013 = df_list[12]
df_Hrs_Wrked_2016 = df_list[13]
df_Hrs_Wrked_2019 = df_list[14]

# df_NumOfJob # Number of jobs
df_NumOfJob_2013 = df_list[15]
df_NumOfJob_2016 = df_list[16]
df_NumOfJob_2019 = df_list[17]

# df_WagesAndSalaries # Wages and Salaries
df_WagesAndSalaries_2013 = df_list[18]
df_WagesAndSalaries_2016 = df_list[19]
df_WagesAndSalaries_2019 = df_list[20]

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
# Average annual wages and salaries

frames = [df_AvgAnnWages_2013, df_AvgAnnWages_2016]
training_df_AvgAnnWages = pd.concat(frames)
testing_df_AvgAnnWages = df_AvgAnnHrsWrk_2019.copy()

grouped = training_df_AvgAnnWages.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

grouped = testing_df_AvgAnnWages.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

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
# Average weekly hours worked

frames = [df_AvgWeekHrsWrked_2013, df_AvgWeekHrsWrked_2016]
training_df_AvgWeekHrsWrked = pd.concat(frames)
testing_df_AvgWeekHrsWrked = df_AvgWeekHrsWrked_2019.copy()

grouped = training_df_AvgWeekHrsWrked.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

grouped = testing_df_AvgWeekHrsWrked.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

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
# Number of jobs

frames = [df_NumOfJob_2013, df_NumOfJob_2016]
training_df_NumOfJob = pd.concat(frames)
testing_df_NumOfJob = df_NumOfJob_2019.copy()

grouped = training_df_NumOfJob.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

grouped = testing_df_NumOfJob.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

# %%
# Wages and Salaries

frames = [df_WagesAndSalaries_2013, df_WagesAndSalaries_2016]
training_df_WagesAndSalaries = pd.concat(frames)
testing_df_WagesAndSalaries = df_WagesAndSalaries_2019.copy()

grouped = training_df_WagesAndSalaries.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

grouped = testing_df_WagesAndSalaries.groupby(['REF_DATE'])
print(grouped['VALUE'].agg([np.sum, np.size]))

# %% [markdown]
# The next step is to display the output

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



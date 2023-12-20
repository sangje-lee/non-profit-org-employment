# %% [markdown]
# <h3> Fourth part of the code </h3>

# %% [markdown]
# Running the main code required.

# %% [markdown]
# Portion of the code involved, including,<br />
# - Portion of finishing code that are splited based on training and testing.<br />
# - Spliting dataset by characteristics.<br />
# - Histogram of each set.

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from fitter import Fitter, get_common_distributions, get_distributions
import ydata_profiling as pp  
from ydata_profiling import ProfileReport 

import warnings
import os

warnings.filterwarnings('ignore')

# %%
# If the dataset are missing for this components, it will give a error.

# df_AvgAnnHrsWrk # Average annual hours worked

df_indicators = ['training_df_AvgAnnHrsWrk.csv', 'testing_df_AvgAnnHrsWrk.csv',
                 'training_df_AvgAnnWages.csv', 'testing_df_AvgAnnWages.csv',
                 'training_df_AvgHrsWages.csv', 'testing_df_AvgHrsWages.csv',
                 'training_df_AvgWeekHrsWrked.csv', 'testing_df_AvgWeekHrsWrked.csv',
                 'training_df_Hrs_Wrked.csv', 'testing_df_Hrs_Wrked.csv',
                 'training_df_NumOfJob.csv', 'testing_df_NumOfJob.csv',
                 'training_df_WagesAndSalaries.csv', 'testing_df_WagesAndSalaries.csv'
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

training_df_AvgAnnHrsWrk = df_list[0] # Average annual hours worked
testing_df_AvgAnnHrsWrk = df_list[1]

training_df_AvgAnnWages = df_list[2] # Average annual wages and salaries
testing_df_AvgAnnWages = df_list[3]

training_df_AvgHrsWages = df_list[4]# Average hourly wage
testing_df_AvgHrsWages = df_list[5]

training_df_AvgWeekHrsWrked = df_list[6] # Average weekly hours worked
testing_df_AvgWeekHrsWrked = df_list[7]

training_df_Hrs_Wrked = df_list[8] # Hours Worked
testing_df_Hrs_Wrked = df_list[9]

training_df_NumOfJob = df_list[10] # Number of jobs
testing_df_NumOfJob = df_list[11]

training_df_WagesAndSalaries = df_list[12] # Wages and Salaries
testing_df_WagesAndSalaries = df_list[13]

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
# Next step, will be the final output.

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



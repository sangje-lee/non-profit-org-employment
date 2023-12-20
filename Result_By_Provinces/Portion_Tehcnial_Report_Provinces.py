# %% [markdown]
# <h3> Fifth part of the code </h3>

# %% [markdown]
# Running the main code required.

# %% [markdown]
# Portion of the code involved, including,<br />
# - Portion of finishing code that are splited based on indicators.<br />
# - Spliting dataset by Provinces.<br />
# - Picking five provinces to analysis and make them into one hand encoding, binary for each provinces <br />
# - Converting or Adding columns that represent each characteristics.<br />
# --> Those can repsent as low to high is set to integer based on low to high<br />
# --> Those that have two answer and unable to order (ex. immigrant and non-immigrant) is done with 0 and 1.

# %%
import pandas as pd
import numpy as np
import ydata_profiling as pp  
from ydata_profiling import ProfileReport 
import seaborn as sns
import matplotlib.pyplot as plt
from fitter import Fitter, get_common_distributions, get_distributions

import warnings
import os

warnings.filterwarnings('ignore')

# %%
def generateIndicators():
    theIndicators = ['df_AvgAnnHrsWrk', 'df_AvgAnnWages', 'df_AvgHrsWages', 'df_AvgWeekHrsWrked', 'df_Hrs_Wrked', 'df_NumOfJob', 'df_WagesAndSalaries']
    theSplits = ['training_', 'testing_']
    y = []
    for x in theIndicators:
        for z in theSplits:
            newSplits = z+x
            y.append(newSplits)

    theIndicators = y.copy()
    # print(theIndicators)

    theCharacteristics = ['_ByAge.csv', '_ByGender.csv', '_ByEducation.csv', '_ByImmigrant.csv']
    y = []
    for x in theIndicators:
        for z in theCharacteristics:
            newSplits = x+z
            y.append(newSplits)

    theIndicators = y.copy()
    # print(theIndicators)
    return theIndicators

# generateIndicators()

# %%
# If the dataset are missing for this components, it will give a error.

# df_AvgAnnHrsWrk # Average annual hours worked

df_indicators = generateIndicators()

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

training_df_AvgAnnHrsWrk_ByAge = df_list[0]
training_df_AvgAnnHrsWrk_ByGender = df_list[1]
training_df_AvgAnnHrsWrk_ByEducation = df_list[2]
training_df_AvgAnnHrsWrk_ByImmigrant = df_list[3]
testing_df_AvgAnnHrsWrk_ByAge = df_list[4]
testing_df_AvgAnnHrsWrk_ByGender = df_list[5]
testing_df_AvgAnnHrsWrk_ByEducation = df_list[6]
testing_df_AvgAnnHrsWrk_ByImmigrant = df_list[7]
training_df_AvgAnnWages_ByAge = df_list[8]
training_df_AvgAnnWages_ByGender = df_list[9]
training_df_AvgAnnWages_ByEducation = df_list[10]
training_df_AvgAnnWages_ByImmigrant = df_list[11]
testing_df_AvgAnnWages_ByAge = df_list[12]
testing_df_AvgAnnWages_ByGender = df_list[13]
testing_df_AvgAnnWages_ByEducation = df_list[14]
testing_df_AvgAnnWages_ByImmigrant = df_list[15]
training_df_AvgHrsWages_ByAge = df_list[16]
training_df_AvgHrsWages_ByGender = df_list[17]
training_df_AvgHrsWages_ByEducation = df_list[18]
training_df_AvgHrsWages_ByImmigrant = df_list[19]
testing_df_AvgHrsWages_ByAge = df_list[20]
testing_df_AvgHrsWages_ByGender = df_list[21]
testing_df_AvgHrsWages_ByEducation = df_list[22]
testing_df_AvgHrsWages_ByImmigrant = df_list[23]
training_df_AvgWeekHrsWrked_ByAge = df_list[24]
training_df_AvgWeekHrsWrked_ByGender = df_list[25]
training_df_AvgWeekHrsWrked_ByEducation = df_list[26]
training_df_AvgWeekHrsWrked_ByImmigrant = df_list[27]
testing_df_AvgWeekHrsWrked_ByAge = df_list[28]
testing_df_AvgWeekHrsWrked_ByGender = df_list[29]
testing_df_AvgWeekHrsWrked_ByEducation = df_list[30]
testing_df_AvgWeekHrsWrked_ByImmigrant = df_list[31]
training_df_Hrs_Wrked_ByAge = df_list[32]
training_df_Hrs_Wrked_ByGender = df_list[33]
training_df_Hrs_Wrked_ByEducation = df_list[34]
training_df_Hrs_Wrked_ByImmigrant = df_list[35]
testing_df_Hrs_Wrked_ByAge = df_list[36]
testing_df_Hrs_Wrked_ByGender = df_list[37]
testing_df_Hrs_Wrked_ByEducation = df_list[38]
testing_df_Hrs_Wrked_ByImmigrant = df_list[39]
training_df_NumOfJob_ByAge = df_list[40]
training_df_NumOfJob_ByGender = df_list[41]
training_df_NumOfJob_ByEducation = df_list[42]
training_df_NumOfJob_ByImmigrant = df_list[43]
testing_df_NumOfJob_ByAge = df_list[44]
testing_df_NumOfJob_ByGender = df_list[45]
testing_df_NumOfJob_ByEducation = df_list[46]
testing_df_NumOfJob_ByImmigrant = df_list[47]
training_df_WagesAndSalaries_ByAge = df_list[48]
training_df_WagesAndSalaries_ByGender = df_list[49]
training_df_WagesAndSalaries_ByEducation = df_list[50]
training_df_WagesAndSalaries_ByImmigrant = df_list[51]
testing_df_WagesAndSalaries_ByAge = df_list[52]
testing_df_WagesAndSalaries_ByGender = df_list[53]
testing_df_WagesAndSalaries_ByEducation = df_list[54]
testing_df_WagesAndSalaries_ByImmigrant = df_list[55]


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

    # def print_groupby_provinces(self):
        

    # def outputPandaProfiling(self, province_id, indicator_id, type_id):

    #     fileName = str(self.indicator[indicator_id]) + " " + str(self.year[type_id])+" in " + str(self.province[province_id]) + ".html"
        
    #     pp = ProfileReport(self.df_ByProvince[province_id])
    #     pp_df = pp.to_html()

    #     print("File name will be saved under "+str(fileName))
    #     f = open(fileName, "a")  # Expert into html file without modifying any columns in dataset.
    #     f.write(pp_df)
    #     f.close()

# %%
training_df_AvgAnnHrsWrk_ByAge_FiveProvinces = FiveProvinceAnalysis(training_df_AvgAnnHrsWrk_ByAge_Provinces, pd, np, pp)
training_df_AvgAnnHrsWrk_ByAge_FiveProvinces.convertCategoricalToNumericValue(0)
testing_df_AvgAnnHrsWrk_ByAge_FiveProvinces = FiveProvinceAnalysis(testing_df_AvgAnnHrsWrk_ByAge_Provinces, pd, np, pp)
testing_df_AvgAnnHrsWrk_ByAge_FiveProvinces.convertCategoricalToNumericValue(0)
testing_df_AvgAnnHrsWrk_ByAge_FiveProvinces.print_One_Hot_Encoded_Data_Info()
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
training_df_AvgHrsWages_ByEducation_FiveProvinces.convertCategoricalToNumericValue(2)
training_df_AvgHrsWages_ByEducation_FiveProvinces.print_One_Hot_Encoded_Data_Info()
training_df_AvgHrsWages_ByEducation_FiveProvinces.print_Unique('Education_group')

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
# training_df_Hrs_Wrked_ByImmigrant_FiveProvinces.convertCategoricalToNumericValue(3)
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



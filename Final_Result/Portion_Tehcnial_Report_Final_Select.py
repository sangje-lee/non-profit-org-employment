# %% [markdown]
# <h3> Last part of the code </h3>

# %% [markdown]
# Contain only training data set

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

import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings('ignore')

# %% [markdown]
# Obtain series of the CSV files,

# %%
# class get DatasetCSV: obtain CSV files from the main directory and split into its training and testig dataset.

class getDatasetCSV:

    def __init__(self):

        self.df_ListOfCSV_Training = []
        self.df_ListOfCSV_Testing = []
        self.theIndicator = []


        def getListFiles():

            import os
            import glob

            item = glob.glob("*.csv")

            return item

        df_List_Files = getListFiles()

        df_List_CSV_Files = []

        for x in df_List_Files:
            txt = x
            
            x = txt.split(".")

            if (x[1] == "csv"):
                df_List_CSV_Files.append(x[0])

        df_List_CSV_Splited = []
        for x in df_List_CSV_Files:
            txt = x
            
            x = txt.split("_")

            df_List_CSV_Splited.append(x)

        df_List_Organized_Training = []
        df_ListOfCSV_Files_Training = []
        count = 0
        for x in df_List_CSV_Splited:
            if x[1] == 'training' :
                df_List_Organized_Training.append(x)
                df_ListOfCSV_Files_Training.append(df_List_CSV_Files[count])
            count = count + 1

        df_List_Organized_Testing = []
        df_ListOfCSV_Files_Testing = []
        count = 0
        for x in df_List_CSV_Splited:
            if x[1] == 'testing' :
                df_List_Organized_Testing.append(x)
                df_ListOfCSV_Files_Testing.append(df_List_CSV_Files[count])
            count = count + 1

        # self.df_ListOfCSV_Training = []
        for x in df_ListOfCSV_Files_Training:
            newItem = str(x+".csv")
            self.df_ListOfCSV_Training.append(newItem)

        # self.df_ListOfCSV_Testing = []
        for x in df_ListOfCSV_Files_Testing:
            newItem = str(x+".csv")
            self.df_ListOfCSV_Testing.append(newItem)

        # self.theIndicator = []
        for x in df_List_Organized_Testing:
            if x[5] == 'Hrs':
                self.theIndicator.append(str(x[4]+"_"+x[5]+"_"+x[6]))
            else:
                self.theIndicator.append(str(x[4]+"_"+x[5]))

        theUnique = []
        for x in self.theIndicator:
            if x not in theUnique: 
                theUnique.append(x) 

        self.theIndicator = theUnique.copy()

        # for x in self.theIndicator:
        #     print(x)
    
    def getTrainingSet(self):
        return self.df_ListOfCSV_Training
    
    def getTestingSet(self):
        return self.df_ListOfCSV_Testing
    
    def printTrainingSet(self):
        for x in self.df_ListOfCSV_Training:
            print(x)

    def printTestingSet(self):
        for x in self.df_ListOfCSV_Testing:
            print(x)

# %%
# class generateDataset(): obtain name of the CSV files from the above class.

class generateDataset():
    def __init__(self):
        p1 = getDatasetCSV()
        self.df_Training = p1.getTrainingSet()
        self.df_Testing = p1.getTestingSet()

    def generateTrainingDataset(self):
        return self.df_Training
    
    def generateTestingDataset(self):
        return self.df_Testing

df_Get_Dataset = generateDataset()
df_Training_Dataset = df_Get_Dataset.generateTrainingDataset()
df_Testing_Dataset = df_Get_Dataset.generateTestingDataset()


# for x in df_Testing_Dataset:
#     print(x)

# %%
# class exportCSV_finalDataset(): convert into actual dataset by opening csv files from above code.

# If the dataset are missing for this components, it will give a error.

# df_AvgAnnHrsWrk # Average annual hours worked

class exportCSV_finalDataset():

    def __init__(self, df):

        self.df_indicators = list(df)

        self.df_list = []

        for x in self.df_indicators:
            if os.path.isfile(x):
                df_sorted_na = pd.read_csv(x)

                # print(df_sorted_na.info())
                # print(df_sorted_na.head(10))

                self.df_list.append(df_sorted_na)
            else:
                print("Run main code first before running this.")
                break
    
    def getSpecificIndicator(self, indicator):
        df_List_CSV_Splited = []
        for x in self.df_indicators:
            txt = x
            
            x = txt.split("_")

            df_List_CSV_Splited.append(x)

        theUnique = []
        for x in df_List_CSV_Splited:
            y = x[4]+"_"+x[5]
            if y not in theUnique:
                theUnique.append(y)

        count = 0
        df_List_to_Update = []
        df_List_to_Display = []
        for x in df_List_CSV_Splited:
            txt = x[4]+"_"+x[5]
            if txt == theUnique[indicator]:
                df_List_to_Update.append(self.df_indicators[count])
                df_List_to_Display.append(self.df_list[count])
            count = count + 1
        print(df_List_to_Update)
        return df_List_to_Display
    
    def getSpecificIndicatorName(self, indicator):
        df_List_CSV_Splited = []
        for x in self.df_indicators:
            txt = x
            
            x = txt.split("_")

            df_List_CSV_Splited.append(x)

        theUnique = []
        for x in df_List_CSV_Splited:
            y = x[4]+"_"+x[5]
            if y not in theUnique:
                theUnique.append(y)

        count = 0
        df_List_to_Update = []
        for x in df_List_CSV_Splited:
            txt = x[4]+"_"+x[5]
            if txt == theUnique[indicator]:
                df_List_to_Update.append(self.df_indicators[count])
            count = count + 1
        return df_List_to_Update

# %% [markdown]
# <h1> Testing set</h1>

# %% [markdown]
# Start run above code to return all of the testing dataset. 

# %%
df_Final_TestingDataset = exportCSV_finalDataset(df_Testing_Dataset)

# %% [markdown]
# Split the dataset based on 'Seven Indicators'.

# %%
# If the code cannot run, it will give a error.

# df_AvgAnnHrsWrk_2013
df_Final_TestingDataset_AvgAnnHrsWrk = df_Final_TestingDataset.getSpecificIndicator(0)
df_Final_TestingDataset_AvgAnnHrsWrk_Name = df_Final_TestingDataset.getSpecificIndicatorName(0)

# df_AvgAnnWages # Average annual wages and salaries
df_Final_TestingDataset_AvgAnnWages = df_Final_TestingDataset.getSpecificIndicator(1)
df_Final_TestingDataset_AvgAnnWages_Name = df_Final_TestingDataset.getSpecificIndicatorName(1)

# df_AvgHrsWages # Average hourly wage
df_Final_TestingDataset_AvgHrsWages = df_Final_TestingDataset.getSpecificIndicator(2)
df_Final_TestingDataset_AvgHrsWages_Name = df_Final_TestingDataset.getSpecificIndicatorName(2)

# df_AvgWeekHrsWrked # Average weekly hours worked
df_Final_TestingDataset_AvgWeekHrsWrked = df_Final_TestingDataset.getSpecificIndicator(3)
df_Final_TestingDataset_AvgWeekHrsWrked_Name = df_Final_TestingDataset.getSpecificIndicatorName(3)

# df_Hrs_Wrked # Hours Worked
df_Final_TestingDataset_Hrs_Wrked = df_Final_TestingDataset.getSpecificIndicator(4)
df_Final_TestingDataset_Hrs_Wrked_Name = df_Final_TestingDataset.getSpecificIndicatorName(4)

# df_NumOfJob # Number of jobs
df_Final_TestingDataset_NumOfJob = df_Final_TestingDataset.getSpecificIndicator(5)
df_Final_TestingDataset_NumOfJob_Name = df_Final_TestingDataset.getSpecificIndicatorName(5)

# df_WagesAndSalaries # Wages and Salaries
df_Final_TestingDataset_WagesAndSalaries = df_Final_TestingDataset.getSpecificIndicator(6)
df_Final_TestingDataset_WagesAndSalaries_Name = df_Final_TestingDataset.getSpecificIndicatorName(6)

# %% [markdown]
# Analysis the data inside the dataset.

# %%
# class Final_Target_To_Analysis: Class that contain all the Analysis using above code.

class Final_Target_To_Analysis:

    def __init__(self, df, pd, np, pp, sns, year):
        self.dfa_Target_To_Analysis = df
        self.year = year
        self.pd = pd
        self.np = np
        self.pp = pp
        self.sns = sns
 

    def get_wholeDataSet(self):
        return self.dfa_Target_To_Analysis
    
    def get_selectDataset(self, order):
        return self.dfa_Target_To_Analysis[order]
        
    def print_Year(self):
        for df_Target_To_Analysis in self.year:
          print(df_Target_To_Analysis)

    def print_unique(self):
        n = 0
        for df_Target_To_Analysis in self.dfa_Target_To_Analysis:
            print(self.year[n])
            print(df_Target_To_Analysis.unique())
            n = n + 1

    def print_info(self):
        n = 0
        for df_Target_To_Analysis in self.dfa_Target_To_Analysis:
            print(self.year[n])
            print(df_Target_To_Analysis.info())
            n = n + 1

    def print_content(self):
        n = 0
        for df_Target_To_Analysis in self.dfa_Target_To_Analysis:
            print(self.year[n])
            print(df_Target_To_Analysis.head(5))
            n = n + 1

    def print_info(self):
        n = 0
        for df_Target_To_Analysis in self.dfa_Target_To_Analysis:
            print(self.year[n])
            print(df_Target_To_Analysis.info())
            n = n + 1

    # create a function
    def print_result_all(self):
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
    
    def print_histogram_all(self):
        n = 0
        for df_Target_To_Analysis in self.dfa_Target_To_Analysis:
            print(self.year[n])
            sns.displot(data=df_Target_To_Analysis, x="VALUE", kind="hist", bins = 100, aspect = 1.5)
            plt.show()
            n = n + 1

    

    def print_result(self, n):
        df_Target_To_Analysis = self.dfa_Target_To_Analysis[n]
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
        df_Target_To_Analysis = self.dfa_Target_To_Analysis[n]
        print(self.year[n])
        sns.displot(data=df_Target_To_Analysis, x="VALUE", kind="hist", bins = 100, aspect = 1.5)
        plt.show()
        n = n + 1
    
    def print_classifier(self, n):
        # n = 0

        if (n == 0 or n == 1):
            # import matplotlib.pyplot as plt
            # from scipy import stats

            # https://realpython.com/linear-regression-in-python/
            # https://www.w3schools.com/python/python_ml_linear_regression.asp

            print(self.year[n])

            xy = self.dfa_Target_To_Analysis[n].copy()
            if (n == 0):
                x = xy['Age_group']
            else:
                x = xy['Education_group']
            y = xy['VALUE'] # [99,86,87,88,111,86,103,87,94,78,77,85,86]

            slope, intercept, r, p, std_err = stats.linregress(x, y)

            def myfunc(x):
                return slope * x + intercept

            mymodel = list(map(myfunc, x))

            plt.scatter(x, y)
            plt.plot(x, mymodel)
            plt.show()

            if (n == 1):
                print("Done by Linear Regression")
                print("Higher the number, higher the education\n")
            else:
                print("Done by Linear Regression\n")
        else:
            # import numpy as np
            # https://www.geeksforgeeks.org/seaborn-categorical-plots/

            print(self.year[n])

            xy = self.dfa_Target_To_Analysis[n].copy()
            if (n == 2):
                x = xy['Gender_group']
            else:
                x = xy['Immigrant_status']
            y = xy['VALUE'] # [99,86,87,88,111,86,103,87,94,78,77,85,86]
            z = 'Characteristics'
 
            # change the estimator from mean to standard deviation
            # sns.barplot(x =x, y ='VALUE', data = xy, 
            #             palette ='plasma', estimator = np.std)
            sns.stripplot(x = x, y ='VALUE', data = xy, 
                        jitter = True, dodge = True) # hue =z, dodge = True)
            plt.show()
            print("Done using Stripplot")

            xyz = []
            if (n == 2):
                print("[1, 0] = "+str(xy['Characteristics'].unique())+"\n")
            else:
                print("[0, 1] = "+str(xy['Characteristics'].unique())+"\n")

    def print_byYearlyGraph(self, x, n):
        print(self.year[n])
        df = self.dfa_Target_To_Analysis[n].copy()

        # x = self.x
        y = 'VALUE'

        sns.stripplot(x = x, y = y, data = df, 
                        jitter = True, dodge = True) # hue =z, dodge = True)
        plt.show()
        print("Done using Stripplot\n")

    def print_Panda_Profiling(self, folder_name):
        n = 0
        for df_Target_To_Analysis in self.dfa_Target_To_Analysis:
            
            old_title = self.year[n]
                
            new_title = old_title.split(".")

            print(new_title)

            new_panda_title = "Panda Profiling for "+new_title[0]

            pp = ProfileReport(df_Target_To_Analysis, title=new_panda_title)
            pp_df_sorted = pp.to_html()

            new_file = folder_name+"/"+new_title[0]+".html"
            f = open(new_file, "a") # Expert modifying data into html file.
            f.write(pp_df_sorted)
            f.close()
            n = n + 1

# %% [markdown]
# Result for testing set for'Average annual hours worked'

# %%
# Export dataset to analysis

dfa_Testing_AvgAnnHrsWrk_To_Analysis = df_Final_TestingDataset_AvgAnnHrsWrk
dfa_Testing_AvgAnnHrsWrk_To_Analysis = Final_Target_To_Analysis(dfa_Testing_AvgAnnHrsWrk_To_Analysis, pd, np, pp, sns, df_Final_TestingDataset_AvgAnnHrsWrk_Name)

# %% [markdown]
# Result for testing set for'Average annual hours worked' by Age group

# %%
dfa_to_cover = 0

dfa_Testing_AvgAnnHrsWrk_To_Analysis.print_result(dfa_to_cover) # Print the result for each Characteristics.
dfa_Testing_AvgAnnHrsWrk_To_Analysis.print_histogram(dfa_to_cover) # Print the histogram for each Characteristics.
dfa_Testing_AvgAnnHrsWrk_To_Analysis.print_classifier(dfa_to_cover) # Print the possible classifier for each Characteristics.

# %% [markdown]
# Result for testing set for'Average annual hours worked' by Education level

# %%
dfa_to_cover = 1

dfa_Testing_AvgAnnHrsWrk_To_Analysis.print_result(dfa_to_cover) # Print the result for each Characteristics.
dfa_Testing_AvgAnnHrsWrk_To_Analysis.print_histogram(dfa_to_cover) # Print the histogram for each Characteristics.
dfa_Testing_AvgAnnHrsWrk_To_Analysis.print_classifier(dfa_to_cover) # Print the possible classifier for each Characteristics.

# %% [markdown]
# Result for testing set for'Average annual hours worked' by Gender group
# 

# %%
dfa_to_cover = 2

dfa_Testing_AvgAnnHrsWrk_To_Analysis.print_result(dfa_to_cover) # Print the result for each Characteristics.
dfa_Testing_AvgAnnHrsWrk_To_Analysis.print_histogram(dfa_to_cover) # Print the histogram for each Characteristics.
dfa_Testing_AvgAnnHrsWrk_To_Analysis.print_classifier(dfa_to_cover) # Print the possible classifier for each Characteristics.

# %% [markdown]
# Result for testing set for'Average annual hours worked' by Immigrant status

# %%
dfa_to_cover = 3

dfa_Testing_AvgAnnHrsWrk_To_Analysis.print_result(dfa_to_cover) # Print the result for each Characteristics. 
dfa_Testing_AvgAnnHrsWrk_To_Analysis.print_histogram(dfa_to_cover) # Print the histogram for each Characteristics.
dfa_Testing_AvgAnnHrsWrk_To_Analysis.print_classifier(dfa_to_cover) # Print the possible classifier for each Characteristics.

# %% [markdown]
# Result for testing set for'Average annual hours worked' by yearly

# %%
dfa_to_cover = 0

dfa_Testing_AvgAnnHrsWrk_To_Analysis.print_byYearlyGraph('REF_DATE',dfa_to_cover) # Print the possible classifier based on Yearly only on Age group.

# %% [markdown]
# Result for testing set for'Average annual wages and salaries'

# %%
# Export dataset to analysis

dfa_Testing_AvgAnnWages_To_Analysis = df_Final_TestingDataset_AvgAnnWages
dfa_Testing_AvgAnnWages_To_Analysis = Final_Target_To_Analysis(dfa_Testing_AvgAnnWages_To_Analysis, pd, np, pp, sns, df_Final_TestingDataset_AvgAnnWages_Name)

# %% [markdown]
# Result for testing set for'Average annual wages and salaries' by Age group

# %%
dfa_to_cover = 0

dfa_Testing_AvgAnnWages_To_Analysis.print_result(dfa_to_cover) # Print the result for each Characteristics.
dfa_Testing_AvgAnnWages_To_Analysis.print_histogram(dfa_to_cover) # Print the histogram for each Characteristics.
dfa_Testing_AvgAnnWages_To_Analysis.print_classifier(dfa_to_cover) # Print the possible classifier for each Characteristics.

# %% [markdown]
# Result for testing set for'Average annual wages and salaries' by Education level

# %%
dfa_to_cover = 1

dfa_Testing_AvgAnnWages_To_Analysis.print_result(dfa_to_cover) # Print the result for each Characteristics.
dfa_Testing_AvgAnnWages_To_Analysis.print_histogram(dfa_to_cover) # Print the histogram for each Characteristics.
dfa_Testing_AvgAnnWages_To_Analysis.print_classifier(dfa_to_cover) # Print the possible classifier for each Characteristics.

# %% [markdown]
# Result for testing set for'Average annual wages and salaries' by Gender group

# %%
dfa_to_cover = 2

dfa_Testing_AvgAnnWages_To_Analysis.print_result(dfa_to_cover) # Print the result for each Characteristics.
dfa_Testing_AvgAnnWages_To_Analysis.print_histogram(dfa_to_cover) # Print the histogram for each Characteristics.
dfa_Testing_AvgAnnWages_To_Analysis.print_classifier(dfa_to_cover) # Print the possible classifier for each Characteristics.

# %% [markdown]
# Result for testing set for'Average annual wages and salaries' by Immigrant status

# %%
dfa_to_cover = 3

dfa_Testing_AvgAnnWages_To_Analysis.print_result(dfa_to_cover) # Print the result for each Characteristics. 
dfa_Testing_AvgAnnWages_To_Analysis.print_histogram(dfa_to_cover) # Print the histogram for each Characteristics.
dfa_Testing_AvgAnnWages_To_Analysis.print_classifier(dfa_to_cover) # Print the possible classifier for each Characteristics.

# %% [markdown]
# Result for testing set for'Average annual wages and salaries' by yearly

# %%
dfa_to_cover = 0

dfa_Testing_AvgAnnWages_To_Analysis.print_byYearlyGraph('REF_DATE',dfa_to_cover) # Print the possible classifier based on Yearly only on Age group.

# %% [markdown]
# Result for testing set for'Average hourly wage'

# %%
# Export dataset to analysis

dfa_Testing_AvgHrsWages_To_Analysis = df_Final_TestingDataset_AvgHrsWages
dfa_Testing_AvgHrsWages_To_Analysis = Final_Target_To_Analysis(dfa_Testing_AvgHrsWages_To_Analysis, pd, np, pp, sns, df_Final_TestingDataset_AvgHrsWages_Name)

# %% [markdown]
# Result for testing set for'Average hourly wage' by Age group

# %%
dfa_to_cover = 0

dfa_Testing_AvgHrsWages_To_Analysis.print_result(dfa_to_cover) # Print the result for each Characteristics.
dfa_Testing_AvgHrsWages_To_Analysis.print_histogram(dfa_to_cover) # Print the histogram for each Characteristics.
dfa_Testing_AvgHrsWages_To_Analysis.print_classifier(dfa_to_cover) # Print the possible classifier for each Characteristics.

# %% [markdown]
# Result for testing set for'Average hourly wage' by Education level

# %%
dfa_to_cover = 1

dfa_Testing_AvgHrsWages_To_Analysis.print_result(dfa_to_cover) # Print the result for each Characteristics.
dfa_Testing_AvgHrsWages_To_Analysis.print_histogram(dfa_to_cover) # Print the histogram for each Characteristics.
# dfa_Testing_AvgHrsWages_To_Analysis.print_result(dfa_to_cover) # Print the possible classifier for each Characteristics.

# %% [markdown]
# Result for testing set for'Average hourly wage' by Gender group

# %%
dfa_to_cover = 2

dfa_Testing_AvgHrsWages_To_Analysis.print_result(dfa_to_cover) # Print the result for each Characteristics.
dfa_Testing_AvgHrsWages_To_Analysis.print_histogram(dfa_to_cover) # Print the histogram for each Characteristics.
dfa_Testing_AvgHrsWages_To_Analysis.print_classifier(dfa_to_cover) # Print the possible classifier for each Characteristics.

# %% [markdown]
# Result for testing set for'Average hourly wage' by Immigrant status

# %%
dfa_to_cover = 3

dfa_Testing_AvgHrsWages_To_Analysis.print_result(dfa_to_cover) # Print the result for each Characteristics. 
dfa_Testing_AvgHrsWages_To_Analysis.print_histogram(dfa_to_cover) # Print the histogram for each Characteristics.
dfa_Testing_AvgHrsWages_To_Analysis.print_classifier(dfa_to_cover) # Print the possible classifier for each Characteristics.

# %% [markdown]
# Result for testing set for'Average hourly wage' by yearly

# %%
dfa_to_cover = 0

dfa_Testing_AvgHrsWages_To_Analysis.print_byYearlyGraph('REF_DATE',dfa_to_cover) # Print the possible classifier based on Yearly only on Age group.

# %% [markdown]
# Result for testing set for'Average weekly hours worked'

# %%
# Export dataset to analysis

dfa_Testing_AvgWeekHrsWrked_To_Analysis = df_Final_TestingDataset_AvgWeekHrsWrked
dfa_Testing_AvgWeekHrsWrked_To_Analysis = Final_Target_To_Analysis(dfa_Testing_AvgWeekHrsWrked_To_Analysis, pd, np, pp, sns, df_Final_TestingDataset_AvgWeekHrsWrked_Name)

# %% [markdown]
# Result for testing set for'Average weekly hours worked' by Age group

# %%
dfa_to_cover = 0

dfa_Testing_AvgWeekHrsWrked_To_Analysis.print_result(dfa_to_cover) # Print the result for each Characteristics.
dfa_Testing_AvgWeekHrsWrked_To_Analysis.print_histogram(dfa_to_cover) # Print the histogram for each Characteristics.
dfa_Testing_AvgWeekHrsWrked_To_Analysis.print_classifier(dfa_to_cover) # Print the possible classifier for each Characteristics.

# %% [markdown]
# Result for testing set for'Average weekly hours worked' by Education level

# %%
dfa_to_cover = 1

dfa_Testing_AvgWeekHrsWrked_To_Analysis.print_result(dfa_to_cover) # Print the result for each Characteristics.
dfa_Testing_AvgWeekHrsWrked_To_Analysis.print_histogram(dfa_to_cover) # Print the histogram for each Characteristics.
dfa_Testing_AvgWeekHrsWrked_To_Analysis.print_classifier(dfa_to_cover) # Print the possible classifier for each Characteristics.

# %% [markdown]
# Result for testing set for'Average weekly hours worked' by Gender group

# %%
dfa_to_cover = 2

dfa_Testing_AvgWeekHrsWrked_To_Analysis.print_result(dfa_to_cover) # Print the result for each Characteristics.
dfa_Testing_AvgWeekHrsWrked_To_Analysis.print_histogram(dfa_to_cover) # Print the histogram for each Characteristics.
dfa_Testing_AvgWeekHrsWrked_To_Analysis.print_classifier(dfa_to_cover) # Print the possible classifier for each Characteristics.

# %% [markdown]
# Result for testing set for'Average weekly hours worked' by Immigrant status

# %%
dfa_to_cover = 3

dfa_Testing_AvgWeekHrsWrked_To_Analysis.print_result(dfa_to_cover) # Print the result for each Characteristics. 
dfa_Testing_AvgWeekHrsWrked_To_Analysis.print_histogram(dfa_to_cover) # Print the histogram for each Characteristics.
dfa_Testing_AvgWeekHrsWrked_To_Analysis.print_classifier(dfa_to_cover) # Print the possible classifier for each Characteristics.

# %% [markdown]
# Result for testing set for'Average weekly hours worked' by yearly

# %%
dfa_to_cover = 0

dfa_Testing_AvgWeekHrsWrked_To_Analysis.print_byYearlyGraph('REF_DATE',dfa_to_cover) # Print the possible classifier based on Yearly only on Age group.

# %% [markdown]
# Result for testing set for'Hours Worked'

# %%
# Export dataset to analysis

dfa_Testing_Hrs_Wrked_To_Analysis = df_Final_TestingDataset_Hrs_Wrked
dfa_Testing_Hrs_Wrked_To_Analysis = Final_Target_To_Analysis(dfa_Testing_Hrs_Wrked_To_Analysis, pd, np, pp, sns, df_Final_TestingDataset_Hrs_Wrked_Name)

# %% [markdown]
# Result for testing set for'Hours Worked' by Age group

# %%
dfa_to_cover = 0

dfa_Testing_Hrs_Wrked_To_Analysis.print_result(dfa_to_cover) # Print the result for each Characteristics.
dfa_Testing_Hrs_Wrked_To_Analysis.print_histogram(dfa_to_cover) # Print the histogram for each Characteristics.
dfa_Testing_Hrs_Wrked_To_Analysis.print_classifier(dfa_to_cover) # Print the possible classifier for each Characteristics.

# %% [markdown]
# Result for testing set for'Hours Worked' by Education level

# %%
dfa_to_cover = 1

dfa_Testing_Hrs_Wrked_To_Analysis.print_result(dfa_to_cover) # Print the result for each Characteristics.
dfa_Testing_Hrs_Wrked_To_Analysis.print_histogram(dfa_to_cover) # Print the histogram for each Characteristics.
dfa_Testing_Hrs_Wrked_To_Analysis.print_classifier(dfa_to_cover) # Print the possible classifier for each Characteristics.

# %% [markdown]
# Result for testing set for'Hours Worked' by Gender group

# %%
dfa_to_cover = 2

dfa_Testing_Hrs_Wrked_To_Analysis.print_result(dfa_to_cover) # Print the result for each Characteristics.
dfa_Testing_Hrs_Wrked_To_Analysis.print_histogram(dfa_to_cover) # Print the histogram for each Characteristics.
dfa_Testing_Hrs_Wrked_To_Analysis.print_classifier(dfa_to_cover) # Print the possible classifier for each Characteristics.

# %% [markdown]
# Result for testing set for'Hours Worked' by Immigrant status

# %%
dfa_to_cover = 3

dfa_Testing_Hrs_Wrked_To_Analysis.print_result(dfa_to_cover) # Print the result for each Characteristics. 
dfa_Testing_Hrs_Wrked_To_Analysis.print_histogram(dfa_to_cover) # Print the histogram for each Characteristics.
dfa_Testing_Hrs_Wrked_To_Analysis.print_classifier(dfa_to_cover) # Print the possible classifier for each Characteristics.

# %% [markdown]
# Result for testing set for'Hours Worked' by yearly

# %%
dfa_to_cover = 0

dfa_Testing_Hrs_Wrked_To_Analysis.print_byYearlyGraph('REF_DATE',dfa_to_cover) # Print the possible classifier based on Yearly only on Age group.

# %% [markdown]
# Result for testing set for'Number of jobs'

# %%
# Export dataset to analysis

dfa_Testing_NumOfJob_To_Analysis = df_Final_TestingDataset_NumOfJob
dfa_Testing_NumOfJob_To_Analysis = Final_Target_To_Analysis(dfa_Testing_NumOfJob_To_Analysis, pd, np, pp, sns, df_Final_TestingDataset_NumOfJob_Name)

# %% [markdown]
# Result for testing set for'Number of jobs' by Age group

# %%
dfa_to_cover = 0

dfa_Testing_NumOfJob_To_Analysis.print_result(dfa_to_cover) # Print the result for each Characteristics.
dfa_Testing_NumOfJob_To_Analysis.print_histogram(dfa_to_cover) # Print the histogram for each Characteristics.
dfa_Testing_NumOfJob_To_Analysis.print_classifier(dfa_to_cover) # Print the possible classifier for each Characteristics.

# %%
df_sorted_prove = dfa_Testing_NumOfJob_To_Analysis.get_selectDataset(0)
df_sorted_prove = df_sorted_prove.loc[
    (df_sorted_prove['VALUE'] > 175000) &
    (df_sorted_prove['Age_group'] == 30)
]

print(df_sorted_prove)

df_sorted_prove = dfa_Testing_NumOfJob_To_Analysis.get_selectDataset(0)
df_sorted_prove = df_sorted_prove.loc[
    (df_sorted_prove['VALUE'] < 150000) &
    (df_sorted_prove['VALUE'] > 100000) &
    (df_sorted_prove['Age_group'] == 30)
]

print(df_sorted_prove)

# %% [markdown]
# Result for testing set for'Number of jobs' by Education level

# %%
dfa_to_cover = 1

dfa_Testing_NumOfJob_To_Analysis.print_result(dfa_to_cover) # Print the result for each Characteristics.
dfa_Testing_NumOfJob_To_Analysis.print_histogram(dfa_to_cover) # Print the histogram for each Characteristics.
dfa_Testing_NumOfJob_To_Analysis.print_classifier(dfa_to_cover) # Print the possible classifier for each Characteristics.

# %% [markdown]
# Result for testing set for'Number of jobs' by Gender group

# %%
dfa_to_cover = 2

dfa_Testing_NumOfJob_To_Analysis.print_result(dfa_to_cover) # Print the result for each Characteristics.
dfa_Testing_NumOfJob_To_Analysis.print_histogram(dfa_to_cover) # Print the histogram for each Characteristics.
dfa_Testing_NumOfJob_To_Analysis.print_classifier(dfa_to_cover) # Print the possible classifier for each Characteristics.

# %% [markdown]
# Result for testing set for'Number of jobs' by Immigrant status

# %%
dfa_to_cover = 3

dfa_Testing_NumOfJob_To_Analysis.print_result(dfa_to_cover) # Print the result for each Characteristics. 
dfa_Testing_NumOfJob_To_Analysis.print_histogram(dfa_to_cover) # Print the histogram for each Characteristics.
dfa_Testing_NumOfJob_To_Analysis.print_classifier(dfa_to_cover) # Print the possible classifier for each Characteristics.

# %% [markdown]
# Result for testing set for'Number of jobs' by yearly

# %%
dfa_to_cover = 0

dfa_Testing_NumOfJob_To_Analysis.print_byYearlyGraph('REF_DATE',dfa_to_cover) # Print the possible classifier based on Yearly only on Age group.

# %% [markdown]
# Result for testing set for'Wages and Salaries'

# %%
# Export dataset to analysis

dfa_Testing_WagesAndSalaries_To_Analysis = df_Final_TestingDataset_WagesAndSalaries
dfa_Testing_WagesAndSalaries_To_Analysis = Final_Target_To_Analysis(dfa_Testing_WagesAndSalaries_To_Analysis, pd, np, pp, sns, df_Final_TestingDataset_WagesAndSalaries_Name)

# %% [markdown]
# Result for testing set for'Wages and Salaries' by Age group

# %%
dfa_to_cover = 0

dfa_Testing_WagesAndSalaries_To_Analysis.print_result(dfa_to_cover) # Print the result for each Characteristics.
dfa_Testing_WagesAndSalaries_To_Analysis.print_histogram(dfa_to_cover) # Print the histogram for each Characteristics.
dfa_Testing_WagesAndSalaries_To_Analysis.print_classifier(dfa_to_cover) # Print the possible classifier for each Characteristics.

# %% [markdown]
# Result for testing set for'Wages and Salaries' by Education level

# %%
dfa_to_cover = 1

dfa_Testing_WagesAndSalaries_To_Analysis.print_result(dfa_to_cover) # Print the result for each Characteristics.
dfa_Testing_WagesAndSalaries_To_Analysis.print_histogram(dfa_to_cover) # Print the histogram for each Characteristics.
dfa_Testing_WagesAndSalaries_To_Analysis.print_classifier(dfa_to_cover) # Print the possible classifier for each Characteristics.

# %% [markdown]
# Result for testing set for'Wages and Salaries' by Gender group

# %%
dfa_to_cover = 2

dfa_Testing_WagesAndSalaries_To_Analysis.print_result(dfa_to_cover) # Print the result for each Characteristics.
dfa_Testing_WagesAndSalaries_To_Analysis.print_histogram(dfa_to_cover) # Print the histogram for each Characteristics.
dfa_Testing_WagesAndSalaries_To_Analysis.print_classifier(dfa_to_cover) # Print the possible classifier for each Characteristics.

# %% [markdown]
# Result for testing set for'Wages and Salaries' by Immigrant status

# %%
dfa_to_cover = 3

dfa_Testing_WagesAndSalaries_To_Analysis.print_result(dfa_to_cover) # Print the result for each Characteristics. 
dfa_Testing_WagesAndSalaries_To_Analysis.print_histogram(dfa_to_cover) # Print the histogram for each Characteristics.
dfa_Testing_WagesAndSalaries_To_Analysis.print_classifier(dfa_to_cover) # Print the possible classifier for each Characteristics.

# %%
df_sorted_prove = dfa_Testing_WagesAndSalaries_To_Analysis.get_selectDataset(3)
df_sorted_prove = df_sorted_prove.loc[
    (df_sorted_prove['VALUE'] > 16030)
]

print(df_sorted_prove)

# %% [markdown]
# Result for testing set for'Wages and Salaries' by yearly

# %%
dfa_to_cover = 0

dfa_Testing_WagesAndSalaries_To_Analysis.print_byYearlyGraph('REF_DATE',dfa_to_cover) # Print the possible classifier based on Yearly only on Age group.

# %% [markdown]
# Panda profiling is removed in this portion of the code.



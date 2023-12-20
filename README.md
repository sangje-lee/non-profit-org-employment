# Analysis of non-profit-organization in Canada before 2021

# Process of this analysis
<ul>
  <li>Export csv file into the big dataset.</li>
  <li>Filtered some columns/attributes and removed null values that are founded.</li>
  <li>Division into different datasets based on the Indicators (There's should be seven datasets)</li>
  <li>Division into four different datasets based on the year. Contains three years worth of data (2010-2012, 2013-2015, 2016-2018, 2019-2021)</li>
  <li>Division into four different characteristics into four dataasets.</li>
  <li>Division based on the GEO, provinces.</li>  
</ul>


# Variable names involve during the analysis

<ul>
  <li>df - Whole dataset without any filtering or division</li>
  <li>df_sorted - Whole dataset with any filtering like removing non-important attributes.</li>
  <li>df_sorted_na - Whole dataset with removal of the null values inside the dataset.</li>
</ul>

Division of into new dataset based on Indicator
<ul>
  <li>df_AvgAnnHrsWrk     - Average annual hours worked</li>
  <li>df_AvgAnnWages      - Average annual wages and salaries</li>
  <li>df_AvgHrsWages      - Average hourly wage</li>
  <li>df_AvgWeekHrsWrked  - Average weekly hours worked</li>
  <li>df_Hrs_Wrked        - Hours Worked</li>
  <li>df_NumOfJob         - Number of jobs</li>
  <li>df_WagesAndSalaries - Wages and Salaries</li>
</ul>

Division of into new dataset based on the GEO/year
<ul>
  <li>df_AvgAnnHrsWrk_2010       - Average annual hours worked in 2010</li>
  <li>df_AvgAnnHrsWrk_2013       - Average annual hours worked in 2013</li>
  <li>df_AvgAnnHrsWrk_2016       - Average annual hours worked in 2016</li>
  <li>df_AvgAnnHrsWrk_2019       - Average annual hours worked in 2019</li>
</ul>
Then merge into
<ul>
  <li>training_df_AvgAnnHrsWrk       - Average annual hours worked for training set (2013-2018) </li>
  <li>testing_df_AvgAnnHrsWrk        - Average annual hours worked for testing set (2019-2021) </li>
</ul>
Not being used anymore
<ul>
  <li>df_AvgAnnHrsWrk_below_2016 - Average annual hours worked below 2016</li>
  <li>df_AvgAnnHrsWrk_above_2017 - Average annual hours worked above 2017</li>
</ul>

# Variable names involve during the analysis
Division of into new dataset based on the group of Characteristics
<ul>
  <li>testing_df_WagesAndSalaries_ByAge          - Wages and Salaries By Age For Testing set</li>
  <li>testing_df_WagesAndSalaries_ByGender       - Wages and Salaries By Gender Group For Testing set</li>
  <li>testing_df_WagesAndSalaries_ByEducation    - Wages and Salaries By Education level For Testing set</li>
  <li>testing_df_WagesAndSalaries_ByImmigrant    - Wages and Salaries By Immigrant level For Testing set</li>
  <li>testing_df_WagesAndSalaries_ByIndigenous   - Wages and Salaries By Indigenous status For Testing set</li>
</ul>

Division of into new dataset based on the provinces
<ul>
  <li>testing_df_AvgAnnHrsWrk_ByAge_Provinces        - Average annual hours worked for testing set by age group grouped by provinces</li>
  <li>testing_df_AvgAnnHrsWrk_ByGender_Provinces     - Average annual hours worked for testing set by gender grouped by provinces</li>
  <li>testing_df_AvgAnnHrsWrk_ByEducation_Provinces  - Average annual hours worked for testing set by education level grouped by provinces</li>
  <li>testing_df_AvgAnnHrsWrk_ByImmigrant_Provinces  - Average annual hours worked for testing set by immigrant status  grouped by provinces</li>
  <li>testing_df_AvgAnnHrsWrk_ByIndigenous_Provinces - Average annual hours worked for testing set by indigenous status grouped by provinces</li>
</ul>

ProvinceAnalysis(df_AvgAnnHrsWrk_201x_ByAge, pd, np, pp) - Create new object using ProvinceAnalysis using datasets and other necessary part.
<br />Variables:
<ul>
  <li>self.df = Dataset, the dataset that import</li>
  <li>self.provinces = array of provinces</li>
  <li>self.indicators = array of indicators</li>
  <li>self.characteristics = array of characteristics </li>
  <li>self.year = array of years being analysis</li>
  <li>self.dfProvinces = array of analysis based of division by provinces, do analysis from the df Dataset</li>
</ul>
Methods:
<ul>
  <li>outputAnalysis(province_id) - Output detail analysis including sum, mean, and skewness.</li>
  <li>outputAnalysisSimple(province_id) - Summarized the output details.</li>
  <li>outputList(province_id, num) - Output first "num" amount of dataset.</li>
  <li>outputPandaProfiling(province_id) - Do Panda profiling for specific provinces in specific year.</li>
</ul>

Province Code [0-13]:
['Alberta', 'BC', 'GEO = Canada' , 'Manitoba' , 'New Brunswick', 'Newfoundland', 'Northwest Territories' , 'Nova Scotia' , 'Nunavut', 'Ontario' , 'PEI', 'Quebec', 'Saskatchewan', 'Yukon'] <br />

OutputProvinceAnalysis(df_AvgAnnHrsWrk_201x_ByAge_Provinces, ProCode, "201x", pd, np, pp) - Create new object using ProvinceAnalysis using dataset and other necessary part.
<ul>
  <li>ProCode is code for the provinces mentions above.</li>
  <li>"201x" here is the year of the analysis.</li>
</ul>
<ul>
  <li>self.df_output - dataset that are analyzing</li>
  <li>self.ProCode - province to analysis (in numeric code)</li>
  <li>self.YearOutput - year that was analyized (more for panda-profiling)</li>
  <li>OutputResult(self) - Display the result that was analyzed.</li>
  <li>OutputPandaProfiling(self) - Do Panda Analysis in specific provinces</li>
</ul>

<h3>For custom output for provinces</h3>
<p> For first input (variable categorized_province), </p>
<p> Input the province to analysis, full province name required. Otherwise, error sign will rise. </p>
<p> For second input,</p>
<p> From the numeric code below from 0 - 6 (variable list_indicator), </p>
<ul>
  <li>"0. Average annual hours worked"</li>
  <li>"1. Average annual wages and salaries"</li>
  <li>"2. Average hourly wage"</li>
  <li>"3. Average weekly hours worked"</li>
  <li>"4. Hours Worked"</li> 
  <li>"5. Number of jobs"</li> 
  <li>"6. Wages and Salaries"</li>
</ul>
<p>Input the indicators required, numerics sign required, if not prompted, it will raise error.</p>

# Contents in this pages
<ul>
  <li>Data_Anlaysis_x - Contain last modified work. Last one is Data_Analysis_v07.</li>
  <li>36100651-eng.zip - Contain original dataset employment of non-profit organizations.</li>
  <li>36100651.csv - Contain original dataset employment of non-profit organizations in csv file.</li>
  <li>EDA_Report_v00.pdf - Inital EDA Report before spliting dataset</li>
  <li>data_analysis_categorized_technical_report.ipynb - Contain techncial report in Jupiter Notebook</li>
  <li>data_analysis_categorized_technical_report.py - contain technical report in Python file.</li>
  <li>data_analysis_categorized_technical_report.html - contain technical report in html file.</li>
  <li>data_analysis_categorized_technical_report.pdf - contain technical report in pdf file.</li>
</ul>

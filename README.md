# Analysis of non-profit-organization in Canada before 2021

# Information
Performed by Sangjin (Eric) Lee for CIND820<br />

# Written Final Report
https://github.com/sangje-lee/non-profit-org-employment/blob/master/Document-Report/Sangjin_Eric_Lee_cind820-Final_Report.pdf (Report)<br />
https://github.com/sangje-lee/non-profit-org-employment/blob/master/Document-Report/Sangjin_Eric_Lee_cind820-Final_Report-output.pdf (Output)

# Presentation Slides
https://github.com/sangje-lee/non-profit-org-employment/blob/master/Document-Report/Sangjin_Eric_Lee_cind820_Presentation_Shorten.pdf (Shorten version)<br />
https://github.com/sangje-lee/non-profit-org-employment/blob/master/Document-Report/Sangjin_Eric_Lee_cind820_Presentation_Detailed.pdf (Detailed version)

# Instruction
https://github.com/sangje-lee/non-profit-org-employment/blob/master/Document-Report/Instruction.pdf (Instruction in PDF)<br />
https://github.com/sangje-lee/non-profit-org-employment/blob/master/Instruction.html (Instruction in HTML)

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

Province Code [0-13]:
['Alberta', 'BC', 'GEO = Canada' , 'Manitoba' , 'New Brunswick', 'Newfoundland', 'Northwest Territories' , 'Nova Scotia' , 'Nunavut', 'Ontario' , 'PEI', 'Quebec', 'Saskatchewan', 'Yukon'] <br />

Merge five provinces with numeric code with datasets divided by characteristics
<ul>
  <li>testing_df_AvgAnnHrsWrk_ByAge_FiveProvinces        - Average annual hours worked for testing set by age group grouped by five provinces</li>
  <li>testing_df_AvgAnnHrsWrk_ByGender_FiveProvinces     - Average annual hours worked for testing set by gender grouped by five provinces</li>
  <li>testing_df_AvgAnnHrsWrk_ByEducation_FiveProvinces  - Average annual hours worked for testing set by education level grouped by five provinces</li>
  <li>testing_df_AvgAnnHrsWrk_ByImmigrant_FiveProvinces  - Average annual hours worked for testing set by immigrant status  grouped by five provinces</li>
  <li>testing_df_AvgAnnHrsWrk_ByIndigenous_FiveProvinces - Average annual hours worked for testing set by indigenous status grouped by five provinces</li>
</ul>

# Contents in this pages
<ul>
  <li>Document-Report - Contain report in PDF files. </li>
  <li>Final_Result - Contain script that contain the final result of analysis.</li>
  <li>HTML_Splited_Result - Contain HTML files of each division has performed and the final result</li>
  <li>Result_By_*** - Script and CSV files that contain each item performed</li>
  <li>36100651-eng.zip - Contain original dataset employment of non-profit organizations.</li>
  <li>36100651.csv - Contain original dataset employment of non-profit organizations in csv file.</li>
  <li>Cohort_analysis_Using_Excel.xlsv - Cohort Analaysis of dataset using Excel</li>
  <li>Empty_Result_Set.zip - Empty directory without any csv files inside</li>
  <li>data_analysis_categorized_technical_report-data_analysis_only.py - Contain script to run data preparation to do analysis</li>
  <li>data_analysis_categorized_technical_report.ipynb - Contain techncial report in Jupiter Notebook</li>
  <li>data_analysis_categorized_technical_report.py - contain technical report in Python file.</li>
  <li>data_analysis_categorized_technical_report.html - contain technical report in html file.</li>
  <li>data_analysis_categorized_technical_report.pdf - contain technical report in pdf file.</li>
</ul>
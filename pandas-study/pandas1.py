import pandas as pd



#Data frames are made up of columns and rows where the rows are known as indexes

df = pd.DataFrame([[1, 2, 3], [4,3,5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17]], columns=["A", 'B', "C"], index=["x", "y", "z", "k", "l", "m"]) #Create simple data frame out of python lists


#BASIC PANDAS COMMANDS

"""
print(df) #Vizualize full dataframe
print(df.head()) #Vizualize first 5 rows of data frame (can pass in first number of rows desired to see)
print(df.tail()) #Same as head but last rows
print(df.sample()) #a sample of data (random)
print(df.columns) #the columns in the df
print(df.index.tolist()) #the indexes (rows) in the df
print(df.info) #Vizualize info on the df (bytes used, size)
print(df.describe) #Useful information (count, mean, standard deviation)
print(df.size) #columns and rows

"""


#CREATING DATA FRAMES

coffee = pd.read_csv(r"C:\Users\Owner\ACSAI\Extra\pandas-study\data\coffee.csv") #Create a df by loading a csv (comma separated values) file. ALso possible with other formats
results_parquet = pd.read_parquet(r"C:\Users\Owner\ACSAI\Extra\pandas-study\data\results.parquet") #Loading a parquet file
bios = pd.read_csv(r"https://raw.githubusercontent.com/KeithGalli/complete-pandas-tutorial/refs/heads/master/data/bios.csv") #loading a file from github. Not locally downloaded

#print(coffee)



#ACCESSING DATA

loc_section = coffee.loc[[3, 10], ["Coffee Type"]] #access data by specifying rows and columns (can use indexes and strings) [[rows], [columns]]
iloc_section = coffee.iloc[4:10, [2]] #difference is columns are specified by index instead of name (in iloc upper index is not included)

coffee.loc[5, "Units sold"] = 1203 #Modify a value in the df using loc



#SORTING DATA

coffee_sorted = coffee.sort_values(["Day", "Units Sold"], ascending=False) #Sort the df by chosen column. Can have secondary sort
coffee_sorted2 = coffee.sort_values(["Day", "Units Sold"], ascending=[1, 0]) #Use 0, 1 to choose which column sorting is ascending.

tall_atlhetes = bios.loc[bios["height_cm"] > 210.0, ["name", "height_cm"]] #filtering the data by setting a condition for the values in height column and choosing the desired columns

Diego_athletes = bios.loc[bios["name"].str.startswith("Diego")] #filtering the data to find athletes named Diego

Tucson_athletes = bios.query('born_country == "USA" and born_city == "Tuscon"') #Filtering the data using the query function. Can be easier to write

#print(Tucson_athletes.head())



"""
Index(['athlete_id', 'name', 'born_date', 'born_city', 'born_region',
       'born_country', 'NOC', 'height_cm', 'weight_kg', 'died_date'],
      dtype='object')
"""


#AVERAGE HEIGHT BY COUNTRY

bios_gp = bios.groupby("born_country")["height_cm"].mean() #Groups by data by chosen column and computes the mean of other chosen column

#print(bios_gp.tail(20))



import numpy as np


#Modifying columns


coffee["New price"] = np.where(coffee['Coffee Type'] == "Latte", 14.99, 10.99)  #Using the numpy library, we use a condition to change values in the df. ( n.where("condition", "value if condition is true", "value if not" )

coffee["Surplus"] = np.where(coffee["Units Sold"] >= 20, 1, 0) #Also creating new columns

coffee.drop(columns=["Units sold"], inplace=True) #Remove data from the df. (inplace makes change permanent)

coffee = coffee.rename(columns={"New price": "Price"}) #Renaming column




new_bios = bios.copy()

new_bios["born_year"] = new_bios["born_date"].str.split("-").str[0] #Use specific data from a column

#new_bios.to_csv(r'./data/new_bios') save locally as csv


#Using a lambda function to create a column. Setting conditions for values in column and handling NaN values with pd.isna

new_bios["century_born"] = new_bios["born_year"].apply(lambda x: 'unknonw' if pd.isna(x) else('19th' if int(x) < 1900 else( '20th' if int(x) < 2000 else '21st')))



#Merge


nocs = pd.read_csv('https://raw.githubusercontent.com/KeithGalli/complete-pandas-tutorial/refs/heads/master/data/noc_regions.csv')

#In this case taking the column "born_country" from bios and column "NOC" from nocs. New df contains all rows in bios and those rows which do not contain a matching value assign NaN
#So in this case, rows which are present on bios but not nocs receive NaN values for columns of nocs
#For example if there is an athlete born ina country that is not in nocs, then the region and notes column for this athlete is filled with NaN

new_bios2 =pd.merge(bios, nocs, left_on="born_country", right_on="NOC", how='left')



#Concatenate


brazil = new_bios2[new_bios2['region'] == 'Brazil'].copy() #Select only rows where region is Brazil
italy = new_bios2[new_bios2['region'] == 'Italy'].copy()

bios_BR_IT = pd.concat([brazil, italy]).sort_values("athlete_id")



#Aggregation


#Using the value counts function to count the values in given column

country_count = bios['born_country'].value_counts() #column 'born_country' in this case
american_cities = bios[bios['born_country'] == "ITA"]["born_region"].value_counts() #column 'born_region of a bios df containing only data where 'country_born' is USA

popular_names_brazil = bios[bios["born_country"] == "BRA"]['name'].value_counts()



#Here the data is being grouped by some column. And some data is agregated in some form
#So in this case the mean units sold per day is seen
day_means = coffee.groupby(["Day"])["Units Sold"].mean()

#The agg function possibilitates you to calculate different things of different columns using a dictionary
day_info = coffee.groupby(["Day"]).agg({"Units Sold":"sum", "Price":"mean"})



coffee["Revenue"] = coffee['Price'] * coffee["Units Sold"]



#The fucntion pivot reshapes the df by choosing the new indexes, columns and values of the new df. (These will be selected columns from the original df
coffee_pivot = coffee.pivot(columns="Day", index="Coffee Type", values="Units Sold")



#Shifting data

coffee['Yersterday Revenue'] = coffee['Revenue'].shift(2)







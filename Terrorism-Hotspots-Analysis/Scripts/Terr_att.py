import pandas as pd
from matplotlib import pyplot as plt


#Vizualize all columns in df when printing in terminal
pd.set_option('display.max_columns', 14)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)


#Read filtered Data set

df = pd.read_csv(r"C:\Users\Owner\ACSAI\Extra\Terrorism-Hotspots-Analysis\datasets1\datasets\Filtered_Data.csv")


#Attacks over the years (Global)

att_yrs = df["iyear"].value_counts() #Create series with count of attacks per year
df_att_yrs = pd.DataFrame(att_yrs).reset_index(drop=False) #Convert to a df and fix index
df_att_yrs.rename(columns={"iyear":"Year"}, inplace=True) #Rename column
df_att_yrs.sort_values(["Year"], ascending=False, inplace=True) #Sort by year


#Extrapolate missing data (1993)

df_att_yrs.loc[len(df_att_yrs["Year"])] = [1993, None] #New row with Year missing
df_att_yrs.sort_values(["Year"], ascending=False,  inplace=True) #Sort values again, now with new row
df_att_yrs.reset_index(drop=True, inplace=True) #Reset Index
df_att_yrs.interpolate(degree=2, inplace=True) #Interpolate the data to fill missing 1993 data. In this method a degree 2 polynomial


#Plot

plt.figure(figsize=(8, 6))

plt.bar(df_att_yrs["Year"], df_att_yrs["count"])

plt.xticks(range(1970, 2020, 5))
plt.xlabel("Year")
plt.ylabel("Attacks")

plt.title("Terrorist attacks over the years (Global)", fontweight="bold")

plt.savefig(r"C:\Users\Owner\ACSAI\Extra\Terrorism-Hotspots-Analysis\plots\Terrorist_att_yrs.png")

plt.show()


#Average attacks over the years (Global)


#Create a new df with the number of attacks per country per year.

#Use groupby year, then by country and then sum

df_global_att = df[["iyear", "country_txt"]] #Filter df
df_global_att =df_global_att.groupby(["iyear", "country_txt"]).size() #Group by year, then by country and then count the incidents of these groups
df_global_att = df_global_att.unstack(fill_value=0) #Pivot the table so the countries become columns

df_global_att_filered = df_global_att.loc[:, (df_global_att != 0).any()] #Create a df with countries that do not contain only zeroes (In this case none do)

df_global_att_filered["Average"] = df_global_att_filered.mean(axis=1) #Create new column with the average attacks throughout countries each year

#df_global_att_filered.to_csv(r"C:\Users\Owner\ACSAI\Extra\Terrorism-Hotspots-Analysis\datasets1\datasets\Global_att_country.csv")


#Find countries with the most attacks

df_global_test = df_global_att_filered.copy()
df_global_test.loc[len(df_global_att_filered)] = [df_global_att_filered[i].sum() for i in df_global_att_filered.columns] #Create new row with the total incidents per country

most_atts = df_global_test.loc[:, df_global_test.iloc[47] > 6000] # Filter out countries with less than 6000 total attacks
most_atts_lis = list(most_atts.columns) # 7 countries with the most terrorist attacks

least_atts = df_global_test.loc[:, df_global_test.iloc[47] < 2] #Countries with the least attacks


#Plot

plt.figure(figsize=(10,8)) #graph size

#Plot a line for every country in the list
for i in most_atts_lis:
    plt.plot(df_global_att_filered.index, df_global_att_filered[i], label= i)

#Plot the average

#plt.plot(df_global_att_filered.index, df_global_att_filered["Average"], label="Global Average", linewidth=2, linestyle="--", color= "red") #Highlight the average line

#Details
plt.legend() #Show line labels
plt.xlabel("Year")
plt.ylabel("Attacks")
plt.xticks(range(1970, 2025, 5))
plt.title("Countries with the most terrorist attacks since 1970", fontweight="bold") #title

#plt.savefig(r"C:\Users\Owner\ACSAI\Extra\Terrorism-Hotspots-Analysis\plots\Most_attacks.png")

#plt.show()






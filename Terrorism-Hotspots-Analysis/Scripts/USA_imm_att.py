import pandas as pd
from matplotlib import pyplot as plt

#Display settings
pd.set_option('display.max_columns', 14)
pd.set_option('display.width', 1000)

#Read files
df_terr = pd.read_csv(r"C:\Users\Owner\ACSAI\Extra\Terrorism-Hotspots-Analysis\datasets1\datasets\Global_att_country.csv") #Df with attacks per country over the years
df_imm = pd.read_excel(r"C:\Users\Owner\ACSAI\Extra\Terrorism-Hotspots-Analysis\datasets1\datasets\MPI-Data-Hub_Imm_N-Percent-US-Pop_2023.xlsx", skiprows=7) # Skip the first 7 rows of the excel file to filter the data needed

#Filter df_imm
df_imm = df_imm.iloc[:29] #Excel contains rows with extra information other than needed data so only some rows must be considered
df_imm["Year"] = df_imm["Year"].astype(int) #Year column in excel data was as type str
df_imm_filtered = df_imm[(df_imm["Year"] > 1970) & (df_imm["Year"] < 2018)].reset_index(drop=True) #Filter Years so are equal to of other data set. Reset indexes

yr_used = [1980, 1990, 2000, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017] #Years to be used by both data sets

#Filter df_terr
df_terr_filtered = df_terr[df_terr["iyear"].isin(yr_used)].reset_index(drop=True) #Set years used. Reset indexes


#Plot


plt.figure(figsize=(8,8))
plt.scatter(df_imm_filtered['Immigrants as a Percentage of the U.S. Population'] *100, df_terr_filtered["United States"])

#Annotations (Years)
offsets = [(5, 10), (5, -15), (5, 10), (-25, -5),  (0, -12), (-20, 5), (20, 0), (20, 0), (20, 0), (-20, 0), (5, 10)] #Annotation positions. (Done manually)
for i in range(len(df_imm_filtered)): #Loop to annotate each point
    plt.annotate(df_imm_filtered["Year"][i], (df_imm_filtered['Immigrants as a Percentage of the U.S. Population'][i], df_terr_filtered["United States"][i]),
                 textcoords="offset points", xytext=offsets[i], ha='center')

#Correlation coefficient
correlation_coefficient = df_imm_filtered['Immigrants as a Percentage of the U.S. Population'].corr(df_terr_filtered["United States"]) #How much does y depend on x?

#Graph details
plt.yticks(range(0, 75, 5)) #Ticks
plt.xlabel('Immigrants percentage of US population (%)') #Labels
plt.ylabel("Terrorist attacks")
plt.title("Immigration and terrorist attacks (USA_percentage)") #Title
plt.text(10, 67, f"Correlation coefficient: {round(correlation_coefficient, 2)}", fontsize=12, ha='center', va='center') #Enter text with correlation coefficient
plt.grid()

#plt.savefig("Immigration_and_terrorist_attacks_(USA_percentage).png")

plt.show()








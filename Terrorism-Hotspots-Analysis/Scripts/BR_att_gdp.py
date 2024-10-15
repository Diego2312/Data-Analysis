import pandas as pd
from matplotlib import pyplot as plt


df_terr = pd.read_csv(r"C:\Users\Owner\ACSAI\Extra\Terrorism-Hotspots-Analysis\datasets1\datasets\Global_att_country.csv") #Df with attacks over the years by country
df_gdp_br = pd.read_csv(r"C:\Users\Owner\ACSAI\Extra\Terrorism-Hotspots-Analysis\datasets1\datasets\UNdata_Export_20241003_192202257\UNdata_Export_20241003_192202257.csv") #Brazil gdp data

df_terr_year = df_terr[(df_terr["iyear"] >= 1990)] #Filter data to 1990 -> 2017
df_gdp_br_year = df_gdp_br[(df_gdp_br["Year"] != 1993) & (df_gdp_br["Year"]<= 2017)] #Filter data to 1990 -> 2017. Not include 1993 (missing data)

#Plot

plt.figure(figsize=(8,8))
plt.scatter(df_gdp_br_year["Value"], df_terr_year["Brazil"])

#Correlation coefficient
corr_coef = df_gdp_br_year["Value"].corr(df_terr_year["Brazil"])

#Graph details
plt.title("Brazil terrorist attacks as GDP changes", fontdict={"weight": "bold"})
plt.xlabel("GDP U$D (Billions)")
plt.ylabel("Attacks")
plt.text(2e12, 38, f"Correlation coefficient: {round(corr_coef, 2)}", fontsize=12, ha='center', va='center') #Enter text with correlation coefficient

#plt.savefig(r"C:\Users\Owner\ACSAI\Extra\Terrorism-Hotspots-Analysis\plots\Brazil_att_gdp.png")

plt.show()
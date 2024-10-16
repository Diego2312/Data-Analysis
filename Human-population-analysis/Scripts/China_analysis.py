import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

#Data sets

#World population data set
df_world_pop = pd.read_csv(r"C:\Users\Owner\ACSAI\Extra\Data-Analysis\Human-population-analysis\Datasets\API_SP.POP.TOTL_DS2_en_csv_v2_31753\API_SP.POP.TOTL_DS2_en_csv_v2_31753.csv", skiprows=4)
df_world_CO2 = pd.read_csv(r"C:\Users\Owner\ACSAI\Extra\Data-Analysis\Human-population-analysis\Datasets\China_CO2\API_EN.ATM.CO2E.KT_DS2_en_csv_v2_32234.csv", skiprows=4)

#China poppulation

df_ch_pop_raw = df_world_pop[df_world_pop["Country Name"] == "China"]
df_melted = pd.melt(df_ch_pop_raw, id_vars=['Country Name'], var_name='Year', value_name='Population')
df_ch_pop = df_melted.loc[3:66, ["Year", "Population"]]

df_ch_pop["Year"] = df_ch_pop["Year"].astype(int)
df_ch_pop_filtered = df_ch_pop[(df_ch_pop["Year"] <= 2020) & (df_ch_pop["Year"] >= 1990)]

#China CO2 emissions

df_ch_CO2_raw = df_world_CO2[df_world_CO2["Country Name"] == "China"]
df_ch_melted = pd.melt(df_ch_CO2_raw, id_vars=['Country Name'], var_name='Year', value_name='CO2')
df_ch_CO2 = df_ch_melted.loc[3:, ["Year", "CO2"]]

df_ch_CO2_filtered = df_ch_CO2[df_ch_CO2["CO2"].notna()]  #Take only years that contain data


#Linear regression

CO2_linear = np.array(df_ch_CO2_filtered["CO2"])
CH_pop_linear = np.array(df_ch_pop_filtered["Population"])

CH_pop_linear = CH_pop_linear.reshape(-1, 1)

model = LinearRegression()
model.fit(CH_pop_linear, CO2_linear)

predicted = model.predict(CH_pop_linear)

#Plot


plt.figure(figsize=(7, 7))

#Population vs CO2
plt.scatter(df_ch_pop_filtered["Population"], df_ch_CO2_filtered["CO2"])

#Linear regression line
plt.plot(df_ch_pop_filtered["Population"], predicted, label="Linear Regression", color="red")

#Correlation coefficient
corr_coef_CO2_Pop = df_ch_pop_filtered["Population"].corr(df_ch_CO2_filtered["CO2"]) #Calculate coefficient
plt.text(1.27e9,1.1e7, f"Correlation coefficient: {round(corr_coef_CO2_Pop, 2)}", fontsize=11, ha='center', va='center') #Enter text with correlation coefficient

#Plot details
plt.xlabel("Population (Billion)")
plt.ylabel("CO2 emission (Kt Million)")
plt.title("China CO2 emission as population changes")
plt.legend()

#plt.savefig(r"C:\Users\Owner\ACSAI\Extra\Data-Analysis\Human-Population-Analysis\Plots\China\China_CO2_population.png")

plt.show()

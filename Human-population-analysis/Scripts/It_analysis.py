import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

#Vizualize all columns in df when printing in terminal
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)


#Read Data

df_italy_pop = pd.read_csv(r"C:\Users\Owner\ACSAI\Extra\Data-Analysis\Human-population-analysis\Datasets\Italy_population.csv")
df_data = pd.read_csv(r"C:\Users\Owner\ACSAI\Extra\Data-Analysis\Human-population-analysis\Datasets\P_Human development index\c1f5091c-bf27-4fb7-b5c1-92612ba89b37_Data.csv")


#Italy life expextancy over the years

df_lfex_italy = df_data[(df_data["Series Name"] == "Life expectancy at birth, total (years)") & (df_data["Country Name"] == "Italy")]
df_lfex_italy = pd.melt(df_lfex_italy, id_vars=['Country Name'], var_name='Year', value_name='Age')
df_lfex_italy = df_lfex_italy.loc[3:65] #Not include last row that has no value
df_lfex_italy["Age"] = round(df_lfex_italy["Age"].astype(float), 1) #Convert Ages into floats then round them
df_lfex_italy["Year"] = df_lfex_italy["Year"].str.split(" ").str[0] #Split values in Year column to select only years



#Italy population and life expectancy

df_italy_pop_plt = df_italy_pop.loc[:62] #Want only up to 2022

#df_italy_pop_plt.to_csv(r"C:\Users\Owner\ACSAI\Extra\Data-Analysis\Human-population-analysis\Datasets\Italy_population.csv")
#df_lfex_italy.to_csv(r"C:\Users\Owner\ACSAI\Extra\Data-Analysis\Human-population-analysis\Datasets\Italy_life_expectancy.csv")



plt.scatter(df_lfex_italy["Age"], df_italy_pop_plt["Population"])

#Correlation coefficient
corr_coef =  df_lfex_italy["Age"].corr(df_italy_pop_plt["Population"])

#Plot details
plt.text(76, 6.02e7,  f"Correlation coefficient: {round(corr_coef, 2)}", fontsize=11, ha='center', va='center') #Enter text with correlation coefficient
plt.title("Italy population as life expectancy changes")
plt.xlabel("Age")
plt.ylabel("Population (10 million)")

#plt.savefig(r"C:\Users\Owner\ACSAI\Extra\Human-population-analysis\Plots\Italy_population_life_expectancy.png")

plt.show()


#Population Density

#Italy land area in km**2

it_land_km = 302073
it_land_m = it_land_km * 10e6

df_it_dens = df_italy_pop.copy()

df_it_dens["Population"] = df_it_dens["Population"].apply(lambda x: round((x/it_land_km), 2))
df_it_dens = df_it_dens[["Year", "Population"]]
df_it_dens.rename(columns={"Population": "Population density km**2"}, inplace=True)


#Population and density

plt.figure(figsize=(7, 7))


corr_coef_dens =  df_italy_pop_plt["Population"].corr(df_it_dens["Population density km**2"])

plt.scatter(df_italy_pop_plt["Population"], df_it_dens["Population density km**2"])

plt.text(5.5e7,198, f"Correlation coefficient: {round(corr_coef_dens, 2)}", fontsize=11, ha='center', va='center') #Enter text with correlation coefficient
plt.xlabel("Population (10 million)")
plt.ylabel("Population density (km**2)")
plt.title("Italy population density as population changes")

#Linear regression line

population = np.array(df_italy_pop_plt["Population"])
density = np.array(df_it_dens["Population density km**2"])

population = population.reshape(-1, 1)

model =LinearRegression()
model.fit(population, density)

predicted_density = model.predict(population)

plt.plot(population, predicted_density, label="Linear regression line", color="red")

plt.legend()
plt.savefig(r"C:\Users\Owner\ACSAI\Extra\Data-Analysis\Human-population-analysis\Plots\Italy_population_density_1.png")

plt.show()
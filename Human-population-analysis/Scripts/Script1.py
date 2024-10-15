import pandas as pd
from matplotlib import pyplot as plt

#Vizualize all columns in df when printing in terminal
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

#Read data

df_world_pop = pd.read_csv(r"C:\Users\Owner\ACSAI\Extra\Human-population-analysis\Datasets\API_SP.POP.TOTL_DS2_en_csv_v2_31753\API_SP.POP.TOTL_DS2_en_csv_v2_31753.csv", skiprows=4)
df_world_gdp = pd.read_csv(r"C:\Users\Owner\ACSAI\Extra\Human-population-analysis\Datasets\API_NY.GDP.MKTP.CD_DS2_en_csv_v2_31795\API_NY.GDP.MKTP.CD_DS2_en_csv_v2_31795.csv", skiprows=4) #Brazil gdp data


#Create new col for total pop
df_world_pop1 = df_world_pop.copy()
df_world_pop1.loc[len(df_world_pop.index), "1960":] = [df_world_pop1[i].sum() for i in df_world_pop1.columns[4:]]
df_world_pop1.loc[266, "Country Name"] = "Year Total"

#World Population
df_tot_pop = df_world_pop1.loc[266, "1960":"2023"] #Series with year and total population
df_tot_pop = pd.DataFrame(df_tot_pop).reset_index(drop=False) #Convert to df
df_tot_pop.rename(columns={"index":"Year", 266: "Total pop"}, inplace=True) #Rename cols
df_tot_pop["Year"] =df_tot_pop["Year"].astype(int) #Convert str cols to ints

#Plot

plt.plot(df_tot_pop["Year"], df_tot_pop["Total pop"])

#Plot details
plt.title("World Population", fontdict={"fontweight":"bold"})
plt.ylabel("Population (Billions)")
plt.xlabel("Year")
plt.xticks(range(1960, 2030, 10))

#plt.show()



#GDP change and pop change Italy

df_it_gdp = df_world_gdp[df_world_gdp["Country Name"] == "Italy"]
df_melted = pd.melt(df_it_gdp, id_vars=['Country Name'], var_name='Year', value_name='GDP')
df_it_gdp = df_melted.loc[3:, ["Year", "GDP"]]

df_it_pop = df_world_pop1[df_world_pop1["Country Name"] == "Italy"]
df_melted2 = pd.melt(df_it_pop, id_vars=['Country Name'], var_name='Year', value_name='Population')
df_it_pop = df_melted2.loc[3:, ["Year", "Population"]]

#df_it_pop.to_csv(r"C:\Users\Owner\ACSAI\Extra\Human-population-analysis\Datasets\Italy_population.csv")

corr_coef = df_it_gdp["GDP"].corr(df_it_pop["Population"])

#Plot

plt.scatter(df_it_gdp["GDP"], df_it_pop["Population"])

#Plot details
plt.text(0.7e12, 6.02e7,  f"Correlation coefficient: {round(corr_coef, 2)}", fontsize=11, ha='center', va='center') #Enter text with correlation coefficient
plt.ylabel("Population (10 million)")
plt.xlabel("GDP (trillion)")
plt.title("Italy population and GDP", fontweight="bold")

#plt.savefig(r"C:\Users\Owner\ACSAI\Extra\Human-population-analysis\Plots\IT_Population_GDP.png")

#plt.show()


#Life expectancy





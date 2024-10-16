import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


#Predict Brazil Population for next 50 Years

#Use shift in population to input previous and output next year

#Data set
df_world_pop = pd.read_csv(r"C:\Users\Owner\ACSAI\Extra\Data-Analysis\Human-population-analysis\Datasets\API_SP.POP.TOTL_DS2_en_csv_v2_31753\API_SP.POP.TOTL_DS2_en_csv_v2_31753.csv", skiprows=4)


#Brazil population
df_br_pop_raw = df_world_pop[df_world_pop["Country Name"] == "Brazil"]
df_melted = pd.melt(df_br_pop_raw, id_vars=['Country Name'], var_name='Year', value_name='Population')
df_br_pop = df_melted.loc[3:66, ["Year", "Population"]]

#Ill use as x my normal brasil pop df starting from 1960 up to 2022 . Ill use as y, my brazil pop df starting from 1961 up to 2023

#In the end, for population it doesnt make much sense, because we would predicting the new population given an old population. So this model would work if for example you want to know what the population will be year +1 knowing the population in year is x.
#Filter data
df_br_y_pop = df_br_pop.loc[4:] # 1961 - 2023
df_br_x_pop = df_br_pop.loc[:65] # 1960 - 2022

pop_x = np.array(df_br_x_pop["Population"])
pop_y = np.array(df_br_y_pop["Population"])

pop_x = pop_x.reshape(-1, 1)

model = LinearRegression()
model.fit(pop_x, pop_y)

predicted = model.predict(pop_x)

plt.plot(pop_x, predicted)

plt.show()

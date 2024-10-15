from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


#Gas prices csv file

#Gas prices over the years
gas = pd.read_csv("https://raw.githubusercontent.com/KeithGalli/matplotlib_tutorial/refs/heads/master/gas_prices.csv")


#Resize graph for better vizualization
plt.figure(figsize=(7, 9))


#Use a loop to plot all the columns in the df over the years
for i in gas.columns[1:]: #First column is year so start after it
    plt.plot(gas["Year"], gas[i], label=i)

#Graph title
plt.title("Gas prices over the Years", fontdict={"fontweight": "bold"})

#Choose axis labels
plt.xlabel("Year")
plt.ylabel("US$")

#Set appropriate ticks
y_val = np.arange(0, 8, 0.5)
plt.yticks(y_val)
plt.xticks(range(1990, 2012, 2))

#Display legend
plt.legend()

#Show grid
plt.grid()

#plt.savefig("Gas-over-the-Years.png", dpi=200)

plt.show()
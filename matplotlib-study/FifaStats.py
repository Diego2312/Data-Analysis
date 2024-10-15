from matplotlib import pyplot as plt
import pandas as pd

#Fifa stats csv file



#Overalls histogram

fifa = pd.read_csv("https://raw.githubusercontent.com/KeithGalli/matplotlib_tutorial/refs/heads/master/fifa_data.csv")

#Select bins of histogram
bins = range(40, 110, 10)

#Plot histogram
plt.hist(fifa["Overall"], bins= bins)

#Hist title
plt.title("Fifa 18 Overall Distribution", fontdict={"fontweight":"bold"})

#Labels
plt.xlabel("Overall")
plt.ylabel("Players (un)")

#Ticks
plt.xticks(bins)

#plt.savefig("Fifa-18-Overall-Distribution.png")


plt.show()



#Pie chart player positions

#Create a series with the counts of players in each position
position_series = fifa["Position"].value_counts()

#Convert to df
position_df = pd.DataFrame(position_series)


#Create a new "position" "other" which include the number of players under threshold
thresh = 300
other_sum = position_df[position_df['count'] < thresh].sum().values[0] #Sum of the postions under threshold
position_df.loc["Other"] = other_sum #Update thresh with new row index "other" and its count

#Filter df for rows with count over the thresh
position_df_filtered = position_df[position_df["count"] > thresh]

#Resize figure to fit player positions
plt.figure(figsize=(12, 12))

#Plot the pie chart
plt.pie(position_df_filtered['count'], labels=position_df_filtered.index) #To plot a pie chart choose which column to be include as values and the index will be the categories

#Title
plt.title("Fifa 18 player position distribution", fontdict={"fontweight": "bold"})

#Axis labels
plt.xlabel("")
plt.ylabel("")

#plt.savefig("Fifa-18-player-position-distribution.png")

plt.show()



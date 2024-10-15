import math

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


#First plot

#Line one
x = [1, 2, 3, 4]
y = [i*2 for i in x]

#Resizing the graph by choosing length and width and pixels per inch.
plt.figure(figsize=(5, 4), dpi=150) #Note that the figure being resized includes both lines. If it came in between the plot of both lines there would be two different figuers (one resized and one not).


plt.plot(x, y, label="2x", color="purple",linestyle='--', linewidth=2) #basic line y in function of x with name for line. Other parameters can be passed check online documentation



#Line two
x2 = np.arange(0, 4, 0.5) # Using a numpy ndarray.
plt.plot(x2[:2], x2[:2]**2, label="x^2") #Also possible to choose which points by slicing the "list"(x or y values)

plt.title("First Plot", fontdict={"fontname": "Times New Roman", "fontsize": 20}) #Set title. Use fontdict to choose font and size
plt.xlabel("x axis") #Set X label
plt.ylabel("y axis") #Set Y label

plt.xticks([0, 1, 2, 3, 4]) #Set what values appear on the x or y axis
#plt.yticks([0, 1, 2, 3])

plt.legend() #Display the label set in the .plot()

plt.savefig("first.png", dpi=200) #We can save the graph and choose the dpi we want to save it.

plt.show()


#Bar graph

#Most functions are the same as with line graph

#Cities visited by country
labels = ["US", "Brazil", "Italy"]
values = [10, 7, 12]

bar = plt.bar(labels, values, color="purple")
plt.title("Cities visited around the world", fontdict={"fontname": "Times New Roman", "fontsize": 20})

plt.xlabel("countries")
plt.ylabel("amount visited")

plt.yticks(range(15))

bar[0].set_hatch('O') #Customise the bars. Striped, full of circles, ect...
bar[1].set_hatch('*')
bar[2].set_hatch('/')


#plt.savefig("cities-visited.png", dpi=200)
plt.show()
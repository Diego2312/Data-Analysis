import pandas as pd


#Vizualize all columns in df when printing in terminal
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


"""
Since the Dataset is very big and a great part of it is not going to be used, the data set will be filtered into what
 will be used to avoid reading unecessary data every time

"""

#Filter the data set into desired columns and rows.
df = pd.read_csv(r"C:\Users\Owner\ACSAI\Extra\Terrorism-Hotspots-Analysis\datasets1\datasets\globalterrorismdb_0718dist.tar.bz2", compression="bz2", usecols=[ 'eventid', 'iyear', 'imonth', 'iday', 'country', 'country_txt', 'region', 'city', 'summary', 'success','suicide', 'attacktype1', 'attacktype1_txt', 'targtype1', 'targtype1_txt', 'natlty1', 'natlty1_txt', 'gname', 'nkill', 'nperps', 'nkillus', 'weaptype1', 'weapsubtype1'])


#Save filtered data into csv file to be used
df.to_csv(r"C:\Users\Owner\ACSAI\Extra\Terrorism-Hotspots-Analysis\datasets1\datasets\Filtered_Data.csv", index=False)






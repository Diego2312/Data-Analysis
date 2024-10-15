import pandas as pd

# Two DataFrames to merge
df1 = pd.DataFrame({
    'id': [1, 2, 3, 4],
    'name': ['Alice', 'Bob', 'Charlie', 'David']
})

df2 = pd.DataFrame({
    'id': [1, 5, 3, 2],
    'age': [24, 19, 22, 23]
})


# DataFrames with different column names
df3 = pd.DataFrame({'key1': [1, 2, 3, 2], 'name': ['Alice', 'Bob', 'Charlie', 'Axel']})
df4 = pd.DataFrame({'key2': [1, 2, 4], 'age': [24, 19, 22]})

# Merging on different column names
merged_df = pd.merge(df3, df4, left_on='key1', right_on='key2')



#The idea of the parameters chosen in the merge function is to choose which one column or columns to be checked in order to merge.

# Inner: In this case the dataframes merge on the column 'id' only taking those rows that are the same on both dfs.
# So the new df will the columns in both the datasets, but the only rows present in the df are those that contain the same value for id

merged_df = pd.merge(df1, df2, on='id')


#Left: in this case the dfs are also merging on the column 'id', but all the rows on df1 (on the left) are kept assigning NaN to a row which is not present in df2.
#So in this case since there is no id 4 in df2, age in id 4 on left_join_df is set to Nan.

left_join_df = pd.merge(df1, df2, on='id', how='left')


# Right: Same happens with right except it`s for the right df (df2)
#So in this case since df1 does not contain id 5, name for id 5 on right_join_df is set to NaN

right_join_df = pd.merge(df1, df2, on='id', how='right')


# Outer: Takes all rows from both dfs, assigning NaN to rows that do not match
#So in this case name on id 5 is NaN and age on id 4 is NaN, since none match

outer_join_df = pd.merge(df1, df2, on='id', how='outer')



# Merging on different column names

#Takes rows which match values from column key1 and key2
new_merged_df = pd.merge(df3, df4, left_on='key1', right_on='key2')



#The rows kept are all the ones on the left and if they don`t match key2 NaN is assigned
#So in this case the rows are the same as in df3 and since id3 is not present in key1, NaN is assigned in age and key2
new_left_join_df = pd.merge(df3, df4, left_on='key1', right_on='key2', how='left')


#Same thing for right except takes rows from df4 in this case
#So in this case only the rows in df4 are present. Since key2 4 does not match df3, name and key are assigned NaN
new_right_join_df = pd.merge(df3, df4, left_on='key1', right_on='key2', how='right')



#All rows are included assigning NaN to rows that do not match
new_outer_join_df = pd.merge(df3, df4, left_on='key1', right_on='key2', how='outer')



print(df1)

print(" ")

print(df2)

print("   ")

print(outer_join_df)

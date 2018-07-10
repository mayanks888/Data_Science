import pandas as pd

df=pd.read_csv("/home/mayank-s/PycharmProjects/Datasets/pascal_voc_csv.csv")
print(df.head())
print (df.info())
#new_data= (df.groupby('image_name'))
#print (new_data)
grouped = df.groupby('image_name')
bbox=[]
for name,group in grouped:
    print (name)
    print (group)
    print(group.iloc[0])
    print (group.values)
    bbox_val=group.values[:,4:8]
    label= group.values[:,4:8] 

    cool1.append(cool)
    print (cool)
    print(group.xmax.values)
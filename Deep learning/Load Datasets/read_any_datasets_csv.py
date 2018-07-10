import pandas as pd
from batchup import data_source
import numpy as np
df=pd.read_csv("/home/mayank-s/PycharmProjects/Datasets/pascal_voc_csv.csv")
print(df.head())
print (df.info())
#new_data= (df.groupby('image_name'))
#print (new_data)
grouped = df.groupby('image_name')
bbox=[]
label=[]
image_name=[]
loop=0
for name,group in grouped:
    loop+=1
    print (name)
    image_name.append(name)
    # print (group)
    # print(group.iloc[0])
    # print (group.values)
    bbox_val=group.values[:,4:8]
    label_val= group.values[:,3]

    bbox.append(bbox_val)
    label.append(label_val)
    print(image_name)
    print (bbox)
    print(label)
    sample_index = np.random.choice(sample_number,batch_size,replace=True)
    if loop>100:
        break



print (bbox.__sizeof__())    #print(group.xmax.values)
ds = data_source.ArrayDataSource([image_name,bbox])

for (im, bb) in ds.batch_iterator(batch_size=10, shuffle=True):
    print(im,bb)




batch








batch








batches = sample_number // batch_size
        for i in range(batches):
            images = []
            boxes = []
            sample_index = np.random.choice(sample_number,batch_size,replace=True)
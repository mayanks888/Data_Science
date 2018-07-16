import cv2
import os
import xml.etree.ElementTree as ET
import  pandas as pd
import numpy as np

# dataset = pd.read_csv("SortedXmlresult.csv")
# /home/mayank-s/PycharmProjects/Datasets/Berkely DeepDrive/bdd100k/images/100k/train
# /home/mayank-s/PycharmProjects/Datasets/Berkely DeepDrive/berkely_train.csv
root="/home/mayank-s/PycharmProjects/Datasets/Berkely_DeepDrive"
csv_name="berkely_train.csv"
csv_path=os.path.join(root, csv_name)
dataset = pd.read_csv(csv_path)
# dataset=dataset.iloc[np.random.permutation(len(dataset))]
x = dataset.iloc[:, 0].values
y = dataset.iloc[:, 4:8].values
val=0
#for loop,xml_aray in zip(x, y):
for loop in x:
# [Xmin,ymin,xmax,ymax]
    top = (y[val,0], y[val,3])
    bottom = (y[val,2], y[val,1])
    image_path=os.path.join(root, "bdd100k/images/100k/train",loop+".jpg")
    my_image=cv2.imread(image_path,1)
    # print(my_image[0,0])
    #image_scale=cv2.resize(my_image,dsize=(224,224),interpolation=cv2.INTER_NEAREST)
    # cv2.rectangle(image_scale, pt1=(125,84),pt2=(159,38),color= (0,255,0), thickness=2)
    cv2.rectangle(my_image, pt1=top, pt2=bottom, color=(0, 255, 0), thickness=2)
    cv2.imshow('streched_image',my_image)
    # output_folder_path="/home/mayank-s/PycharmProjects/Datasets/aptive/object_detect/output/output_merccedes.png"
    # filepath=output_folder_path+my_image+".png"
    # cv2.imwrite(filepath,my_image)
    cv2.waitKey(500)
    cv2.destroyAllWindows()
    val+=1
    print (val)
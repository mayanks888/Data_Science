import os
import xml.etree.ElementTree as ET
import pandas as pd

imageNameFile="fsg"
# vocPath="/home/mayank-s/PycharmProjects/Datasets/single_object_detecetion"

#windows
# vocPath="C:/Users/mayank/Documents/Datasets/single_object_detection"
vocPath=r"C:\Users\mayank\Documents\Datasets\single_object_detection"
all_xml=[]
for file in os.listdir(os.path.join(vocPath,'annotations','xmls')):
    if file.endswith(".xml"):
        #print(os.path.join("/mydir", file))
        all_xml.append(file)



def parseXML(xmlPath, labels,pixel_size):
    """
    Args:
      xmlPath: The path of the xml file of this image
      labels: label names of pascal voc dataset
      side: an image is divided into side*side grid
    """
    tree = ET.parse(xmlPath)
    root = tree.getroot()

    width = int(root.find('size').find('width').text)
    height = int(root.find('size').find('height').text)

    for obj in root.iter('object'):
        class_num = labels.index(obj.find('name').text)#finding the index of label mention so as to store data into correct label
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        h = ymax - ymin
        w = xmax - xmin
        # which cell this obj falls into
        centerx = (xmax + xmin) / 2.0
        centery = (ymax + ymin) / 2.0
        #448 is size of the input image
        newx = (pixel_size / width) * centerx
        newy = (pixel_size / height) * centery

        h_new = h * (pixel_size / height)
        w_new = w * (pixel_size / width)

        new_xmin=int(newx-(w_new/2))
        new_xmax=int(newx+(w_new/2))
        new_ymin=int(newy-(h_new/2))
        new_ymax=int(newy+(h_new/2))
        return (new_xmax,new_xmin,new_ymax,new_ymin,class_num)


my_imagelist =[]
my_xmlist=[]
xml_val_list=[]
label = ['dog', 'cat']
for xml_file in all_xml:
   myslpit=xml_file.split(".xml")
   new_path=os.path.join(vocPath,'annotations','xmls',xml_file)
   my_xmlist.append(new_path)
   my_imagelist.append(os.path.join(vocPath,'images',myslpit[0])+'.jpg')
   # new_path=r"C:\Users\mayank\Documents\Datasets\single_object_detection\annotations\xmls\Abyssinian_10.xml"
   k,l,m,n,cls_label=parseXML(new_path, label, pixel_size=224)
   xml_val_list.append([cls_label,k,l,m,n])

df=pd.DataFrame(data=(my_imagelist))
df2=pd.DataFrame(data=(xml_val_list))
result = pd.concat([df, df2], axis=1, join='inner')
#df.append(xml_val_list)
result.to_csv('SortedXmlresult')
# First I will try to create a raw training data of all image raw data and annotation data
'''def prepareBatch(start,end,imageNameFile,vocPath):
    my_imagelist = []
    my_xmlist=[]
    label = ['dog', 'cat']
    for xml_file in all_xml:
        myslpit=xml_file.split(".xml")
        my_xmlist.append(os.path.join(vocPath,xml_file))
        my_imagelist(os.path.join(vocPath,'images',myslpit[0])+'.jpg')
        k,l,m,n=parseXML((vocPath,xml_file),label,pixel_size=220)
        #imgPath = os.path.join(vocPath,'JPEGImages',imgName)+'.jpg'
       #xmlPath = os.path.join(vocPath,'Annotations',imgName)+'.xml'
        



    imageList = []
    labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
    file = open(imageNameFile)
    imageNames = file.readlines()
    
    for i in range(start,end):
        imgName = imageNames[i].strip('/n')
        imgPath = os.path.join(vocPath,'JPEGImages',imgName)+'.jpg'
        xmlPath = os.path.join(vocPath,'Annotations',imgName)+'.xml'
        img = image(side=7,imgPath=imgPath)
        img.parseXML(xmlPath,labels,7)
        imageList.append(img)

    return imageList'''


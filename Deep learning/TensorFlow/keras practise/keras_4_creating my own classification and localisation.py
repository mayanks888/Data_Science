import os
import xml.etree.ElementTree as ET
import pandas as pd


imageNameFile="fsg"
vocPath="/home/mayank-s/PycharmProjects/Datasets/single_object_detecetion"
all_xml=[]
for file in os.listdir(os.path.join(vocPath,'annotations','xmls')):
    if file.endswith(".xml"):
        #print(os.path.join("/mydir", file))
        all_xml.append(file)


my_imagelist =[]
my_xmlist=[]
label = ['dog', 'cat']
for xml_file in all_xml:
   myslpit=xml_file.split(".xml")
   my_xmlist.append(os.path.join(vocPath,xml_file))
   my_imagelist.append(os.path.join(vocPath,'images',myslpit[0])+'.jpg')


# First I will try to create a raw training data of all image raw data and annotation data
def prepareBatch(start,end,imageNameFile,vocPath):
    my_imagelist = []
    my_xmlist=[]
    label = ['dog', 'cat']
    for xml_file in all_xml:
        myslpit=xml_file.split(".xml")
        xmlist.append(os.path.join(vocPath,xml_file)
        my_imagelist(os.path.join(vocPath,'images',myslpit[0])+'.jpg')
        
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

    return imageList

def parseXML(xmlPath, labels, side):
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
        # objif = objInfo(xmin/448.0,ymin/448.0,np.sqrt(ymax-ymin)/448.0,np.sqrt(xmax-xmin)/448.0,class_num)

        # which cell this obj falls into
        centerx = (xmax + xmin) / 2.0
        centery = (ymax + ymin) / 2.0
        #448 is size of the input image
        newx = (448.0 / width) * centerx
        newy = (448.0 / height) * centery

        h_new = h * (448.0 / height)
        w_new = w * (448.0 / width)

        cell_size = 448.0 / side
        col = int(newx / cell_size)
        row = int(newy / cell_size)
        # print "row,col:",row,col,centerx,centery

        cell_left = col * cell_size
        cell_top = row * cell_size
        cord_x = (newx - cell_left) / cell_size
        cord_y = (newy - cell_top) / cell_size

        #objif = objInfo(cord_x, cord_y, np.sqrt(h_new / 448.0), np.sqrt(w_new / 448.0), class_num)
        #self.boxes[row][col].has_obj = True
        #self.boxes[row][col].objs.append(objif)

# mylabel='head'
side=7
mylabel = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
parseXML(xml_source,mylabel,side)
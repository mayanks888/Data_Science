import argparse
import os
import xml.etree.ElementTree as ElementTree
from PIL import Image
import io
import h5py
import numpy as np
from collections import namedtuple, OrderedDict
import pandas as pd
import cv2
classes=['person','motor','bus','car','motorbike','traffic light','traffic sign', 'truck', 'train']

def class_text_to_int(row_label):
    if row_label == 'person':
        return 1
    elif row_label == 'motor':
        return 2
    elif row_label == 'bus':
        return 3
    elif row_label == 'car':
        return 4
    elif row_label == 'bike':
        return 5
    elif row_label == 'traffic light':
        return 6
    elif row_label == "traffic sign":
        return 7
    elif row_label == 'truck':
        return 8
    elif row_label == 'train':
        return 9
    else:
        None
def get_image_for_id(voc_path, image_id):
    voc_path='/home/mayank-s/PycharmProjects/Datasets/Berkely_DeepDrive/bdd100k/images/100k/train'

    """Get image data as uint8 array for given image.
    Parameters
    ----------
    voc_path : str
        Path to VOCdevkit directory.
    year : str
        Year of dataset containing image. Either '2007' or '2012'.
    image_id : str
        Pascal VOC identifier for given image.
    Returns
    -------
    image_data : array of uint8
        Compressed JPEG byte string represented as array of uint8.
    """
    fname = os.path.join(voc_path,  image_id)
    with open(fname, 'rb') as in_file:
        data = in_file.read()
    # Use of encoding based on: https://github.com/h5py/h5py/issues/745
    return np.fromstring(data, dtype='uint8')


def add_to_dataset(voc_path, year, ids, images, boxes, start=0):
    """Process all given ids and adds them to given datasets."""
    for i, voc_id in enumerate(ids):
        image_data = get_image_for_id(voc_path, year, voc_id)
        image_boxes = get_boxes_for_id(voc_path, year, voc_id)
        images[start + i] = image_data
        boxes[start + i] = image_boxes
    return i

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    image_name=group.filename+".jpg"
    with tf.gfile.GFile(os.path.join(path, '{}'.format(image_name)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

def _main(args):
    voc_path = os.path.expanduser(args.path_to_voc)
    train_ids = get_ids(voc_path, train_set)
    val_ids = get_ids(voc_path, val_set)
    test_ids = get_ids(voc_path, test_set)
    train_ids_2007 = get_ids(voc_path, sets_from_2007)
    total_train_ids = len(train_ids) + len(train_ids_2007)

    # Create HDF5 dataset structure
    print('Creating HDF5 dataset structure.')
    fname = os.path.join(voc_path, 'pascal_voc_07_12.hdf5')
    voc_h5file = h5py.File(fname, 'w')
    uint8_dt = h5py.special_dtype(
        vlen=np.dtype('uint8'))  # variable length uint8
    vlen_int_dt = h5py.special_dtype(
        vlen=np.dtype(int))  # variable length default int
    train_group = voc_h5file.create_group('train')
    val_group = voc_h5file.create_group('val')
    test_group = voc_h5file.create_group('test')

    # store class list for reference class ids as csv fixed-length numpy string
    voc_h5file.attrs['classes'] = np.string_(str.join(',', classes))

    # store images as variable length uint8 arrays
    train_images = train_group.create_dataset(
        'images', shape=(total_train_ids, ), dtype=uint8_dt)
    val_images = val_group.create_dataset(
        'images', shape=(len(val_ids), ), dtype=uint8_dt)
    test_images = test_group.create_dataset(
        'images', shape=(len(test_ids), ), dtype=uint8_dt)

    # store boxes as class_id, xmin, ymin, xmax, ymax
    train_boxes = train_group.create_dataset(
        'boxes', shape=(total_train_ids, ), dtype=vlen_int_dt)
    val_boxes = val_group.create_dataset(
        'boxes', shape=(len(val_ids), ), dtype=vlen_int_dt)
    test_boxes = test_group.create_dataset(
        'boxes', shape=(len(test_ids), ), dtype=vlen_int_dt)

    # process all ids and add to datasets
    print('Processing Pascal VOC 2007 datasets for training set.')
    last_2007 = add_to_dataset(voc_path, '2007', train_ids_2007, train_images,
                               train_boxes)
    print('Processing Pascal VOC 2012 training set.')
    add_to_dataset(
        voc_path,
        '2012',
        train_ids,
        train_images,
        train_boxes,
        start=last_2007 + 1)
    print('Processing Pascal VOC 2012 val set.')
    add_to_dataset(voc_path, '2012', val_ids, val_images, val_boxes)
    print('Processing Pascal VOC 2007 test set.')
    add_to_dataset(voc_path, '2007', test_ids, test_images, test_boxes)

    print('Closing HDF5 file.')
    voc_h5file.close()
    print('Done.')





if __name__ == '__main__':

    voc_path = '/home/mayank-s/PycharmProjects/Datasets/Berkely_DeepDrive'
    csv_path = "/home/mayank-s/PycharmProjects/Datasets/Berkely_DeepDrive/berkely_train.csv"
    print('Creating HDF5 dataset structure.')
    fname = os.path.join(voc_path, 'My_developed.hdf5')

    voc_h5file = h5py.File(fname, 'w')

    uint8_dt = h5py.special_dtype(vlen=np.dtype('uint8'))  # variable length uint8
    vlen_int_dt = h5py.special_dtype(vlen=np.dtype(int))  # variable length default int

    # Creating struture for hdf5 fromat
    train_group = voc_h5file.create_group('train')
    val_group = voc_h5file.create_group('val')
    test_group = voc_h5file.create_group('test')

    # store class list for reference class ids as csv fixed-length numpy string
    voc_h5file.attrs['classes'] = np.string_(str.join(',', classes))

    total_train_ids=10942
    val_ids=2000
    test_ids=20


    # store images as variable length uint8 arrays
    train_images = train_group.create_dataset(
        'images', shape=(total_train_ids,), dtype=uint8_dt)
    val_images = val_group.create_dataset(
        'images', shape=((val_ids),), dtype=uint8_dt)
    test_images = test_group.create_dataset(
        'images', shape=((test_ids),), dtype=uint8_dt)

    # store boxes as class_id, xmin, ymin, xmax, ymax
    train_boxes = train_group.create_dataset(
        'boxes', shape=(total_train_ids,), dtype=vlen_int_dt)
    val_boxes = val_group.create_dataset(
        'boxes', shape=((val_ids),), dtype=vlen_int_dt)
    test_boxes = test_group.create_dataset(
        'boxes', shape=((test_ids),), dtype=vlen_int_dt)


    # _main(parser.parse_args())
    examples = pd.read_csv(csv_path)
    grouped = split(examples, 'filename')
    loop=0
    for group in grouped:
        image_name = group.filename + ".jpg"
        image_data = get_image_for_id("bdbd", image_name)
        train_images[loop]=image_data
        #img = Image.open(io.BytesIO(image_data))#this will convert image back to normal img pixel values
        # This took ~30ms for a hi-res 3-channel image (~2000x2000
        # img.show()
        print(loop)
        print(image_name)
        boxes = []
        for index, row in group.object.iterrows():
            # print(index)
            # print (row)
            class_no=(class_text_to_int(row['class']))
            # name=row['class'].encode('utf8')
            xmin,xmax,ymin,ymax=row['xmin'],row['xmax'] ,row['ymin'],row['ymax']
            bbox=(class_no,xmin,ymin,xmax,ymax)
            boxes.extend(bbox)
            final_bbox= np.array(boxes)
            train_boxes[loop]=final_bbox
            
        loop+=1
    #

    print('Closing HDF5 file.')
    voc_h5file.close()
    print('Done.')



    img = Image.open(io.BytesIO(dset[0]))  # This took ~30ms for a hi-res 3-channel image (~2000x2000)
    img.show()
import argparse
import os
import h5py
import numpy as np
import datetime
import time
ts = time.time()
class Image_To_HDF5():


    def image_hdf5(self, input_path, output_path):

        if not os.path.exists(input_path):
            print("Input file not found")
            return 1

        if not os.path.exists(output_path):
            print("Output folder not present")
            return 1

        st = datetime.datetime.fromtimestamp(ts).strftime('%d_%m_%Y_%H_%M_%S')
        Hdf5_file_name="INPUT_"+st+".hdf5"
        # image_path = '/home/mayank-s/Desktop/Link to Datasets/aptive/object_detect/input'

        print('Creating HDF5 dataset structure.')

        fname = os.path.join(output_path, Hdf5_file_name)

        voc_h5file = h5py.File(fname, 'w')

        uint8_dt = h5py.special_dtype(vlen=np.dtype('uint8'))  # variable length uint8
        # vlen_int_dt = h5py.special_dtype(vlen=np.dtype(int))  # variable length default int
        test_group = voc_h5file.create_group('test')

        start = 0
        for root, _, filenames in os.walk(input_path):
                if (len(filenames) == 0):
                    print("Input folder is empty")
                    return 1
                test_ids = len(filenames)
                test_images = test_group.create_dataset(name='input_images', shape=((test_ids),), dtype=uint8_dt)
                # test_images = test_group.create_dataset(name='images', shape=(3, 2), dtype=uint8_dt)
                for filename in filenames:
                    try:
                        image_data = self.get_image_for_id(input_path, filename)
                        # if len(image_data.shape) is None:
                            # print("{it} datasets is not present in hdf5 file".format(it=image_type))
                        test_images[start] = image_data
                        start += 1
                        print(start)
                    except IOError:
                        print("Existing Image to HDF5 conversion...")
                    except:
                        print('Error in image conversion for file name {fn}, jumping to next file'.format(fn=filename))
                    else:
                        1


        print('Closing HDF5 file.')
        voc_h5file.close()
        print('Converted sucessfully')


    def get_image_for_id(self,path, image_id):
        fname = os.path.join(path, image_id)
        with open(fname, 'rb') as in_file:
            data = in_file.read()
        # Use of encoding based on: https://github.com/h5py/h5py/issues/745
        return np.fromstring(data, dtype='uint8')

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Object detection on imgage started..')
    # parser.add_argument('--input_path', help="Input Folder")
    # parser.add_argument('--output_path', help="Output folder")
    parser.add_argument('--input_path', help="Input Folder",
                        default='/home/mayank-s/PycharmProjects/Datasets/aptive/object_detect/input')
    parser.add_argument('--output_path', help="Output folder",
                        default='/home/mayank-s/PycharmProjects/Datasets/aptive/object_detect')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args=parse_args()
    # print('\n',"Extracting Images from HDF5 file...","\n")
    print('Reading hdf5 from path :', args.input_path)
    model = Image_To_HDF5()
    ret = model.image_hdf5(args.input_path,args.output_path)
    # ret = model.find_detection(args.input_path)
    if ret==1:
        print("\n","File Error.....")
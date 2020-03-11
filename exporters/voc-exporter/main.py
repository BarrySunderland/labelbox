import argparse
import json
import os
import logging
import traceback as tb
import numpy as np

import voc_exporter

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

def create_segmentation_splits(image_output, splits=(.8,.2)):

    """image_output: (str) path the image output folder
    splits: (tuple of floats) ratio to split images into train and val"""
    
    img_list = os.listdir(image_output)

    num_imgs = len(img_list)
    num_train_imgs = int(splits[0] * num_imgs)
    num_val_imgs = int(splits[1] * num_imgs)
    
    #randomise the images used for training
    np.random.shuffle(img_list)
    
    #split into train and val
    train_imgs = img_list[:num_train_imgs]
    val_imgs = img_list[num_train_imgs:]
    
    img_dict= {'train' : train_imgs,
               'val' : val_imgs,
               'trainval' : img_list }
    
    #use foldernames from Pascal VOC 2012
    fldrpath = os.path.normpath(image_output + '/../ImageSets/Segmentation')
    if not os.path.exists(fldrpath):
        os.makedirs(fldrpath)
        
    #write the files to the folder
    for key, lst in img_dict.items():
        fpath = os.path.join(fldrpath, '{}.txt'.format(key))
        with open(fpath, 'w') as f:
            for item in lst:
                item = item[:item.find('.')]
                f.write("%s\n" % item)
                
    return

    
def export(file_input, file_output, image_output):
    "Uses VOC exporter function from_json to convert labelbox JSON into MS VOC format."

    try:
        os.makedirs(file_output, exist_ok=True)
        os.makedirs(image_output, exist_ok=True)
        LOGGER.info('Creating voc export')

        voc_exporter.from_json(file_input, file_output, image_output)

        LOGGER.info('Done saving voc export')

    except Exception as e:
        tb.print_exc()
        

# #if __name__ == '__main__':
# parser = argparse.ArgumentParser()
# parser.add_argument('file_input', help='File path to Labelbox JSON to parse export')
# parser.add_argument('file_output', help='File path to desired output directory for export asset')


# args = parser.parse_args()

# args.file_input = arg0
# file_input = args.file_input
# assert file_input

# args.file_output=arg1
# file_output = args.file_output
# assert file_output

file_input = os.path.normpath('../../../deeplab\datasets\PQR\dataset\export-2020-03-10T17_21_12.415Z.json')
file_output = os.path.normpath('../../../deeplab\datasets\PQR\dataset')

image_output = file_output + '/JPEGImages'

export(file_input, file_output, image_output)

create_segmentation_splits(image_output)

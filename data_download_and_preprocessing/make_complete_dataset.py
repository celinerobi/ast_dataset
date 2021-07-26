"""
Correct inconsistent labels
"""

"""
Import Packages
"""
import shutil
import xml.etree.ElementTree as et
import argparse

import os
import sys
from PIL import Image
import numpy as np
from glob import glob
import data_eng.az_proc as ap

def get_args_parse():
    parser = argparse.ArgumentParser(
        description='This script adds a subdirectory of xmls to correct possible inconsistent labels')
    parser.add_argument('--parent_directory', type=str, default=None,
                        help='path to parent directory, holding the annotation directory.')
    parser.add_argument('--original', type=bool, default=False,
                        help='use original (True), or corrected (False) annotations')
    args = parser.parse_args()
    return args

def main(args):
    ### Get the subdirectories within the subdirectories (the folders from each of the allocations)
    sub_directories = list()
    for annotator_directory in ap.list_of_sub_directories(args.parent_directory):
        for root,dirs,files in os.walk(annotator_directory):
            if "chips" in dirs: #excludes the annotations from Jackson that need to be reformated
                sub_directories.append(root)

    ### Move the annotations + images 
    counter_annotations = 0
    counter_images = 0

    for i in range(len(sub_directories)):
        annotator = sub_directories[i].rsplit("/",1)[1].split("\\")[0] #get the annotator name
        sub_directory = sub_directories[i].rsplit("/",1)[1] #get the sub folders for each annotator 

        print("The current subdirectory:", sub_directory)

        #Functions to move the annotations + images into folders 
        dist = ap.annotator(sub_directory)
        dist.state_dcc_directory(args.parent_directory)
        dist.make_subdirectories()    
        annotations, images = dist.move_images_annotations_to_complete_dataset(args.original)

        counter_annotations += annotations # count the number of annotations 
        counter_images += images #count the number of images
        print(counter_annotations, counter_images) #print the counters

if __name__ == '__main__':
    ### Get the arguments 
    args = get_args_parse()
    main(args)


        
        


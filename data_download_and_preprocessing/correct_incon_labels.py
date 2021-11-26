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
                        help='path to parent directory, holding the annotation sub directories.')
    parser.add_argument('--complete_dataset_directory', type=str, default=None,
                        help='path to complete dataset directory.')
    args = parser.parse_args()
    return args

def main(args): 
    if args.parent_directory is not None:
        #get the subdirectories within the subdirectories (the folders from each of the allocations)
        sub_directories = list()
        for annotator_directory in ap.list_of_sub_directories(parent_directory):
            for root,dirs,files in os.walk(annotator_directory):
                if "chips" in dirs: #excludes the annotations from Jackson that need to be reformated
                    sub_directories.append(root)

        for i in range(len(sub_directories)):
            annotator = sub_directories[i].rsplit("/",1)[1].split("\\")[0] #get the annotator name
            sub_directory = sub_directories[i].rsplit("/",1)[1] #get the sub folders for each annotator
            dist = ap.annotator(sub_directory)
            dist.state_dcc_directory(parent_directory)
            dist.make_subdirectories()
            dist.correct_inconsistent_labels_xml()   
            
    if args.complete_dataset_directory is not None:
        dist = ap.annotator(args.complete_dataset_directory.rsplit("\\",1)[1])
        dist.state_dcc_directory(args.complete_dataset_directory.rsplit("\\",1)[0])
        dist.make_subdirectories()
        dist.correct_inconsistent_labels_xml()   

if __name__ == '__main__':
    ### Get the arguments 
    args = get_args_parse()
    main(args)


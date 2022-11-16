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
    parser.add_argument('--parent_dir', type=str, default=None,
                        help='path to parent directory, holding the annotation sub directories.')
    parser.add_argument('--complete_dir', type=str, default=None,
                        help='path to complete dataset directory.')
    args = parser.parse_args()
    return args

def main(args): 
    if args.parent_dir is not None:
        #get the subdirectories within the subdirectories (the folders from each of the allocations)
        sub_directories = list()
        for annotator_directory in ap.list_of_sub_directories(args.parent_dir):
            for root, dirs, files in os.walk(annotator_directory):
                if "chips" in dirs: #excludes the annotations from Jackson that need to be reformated
                    sub_directories.append(root)

        for i in range(len(sub_directories)):
            sub_directory = sub_directories[i].rsplit("/",1)[1] #get the sub folders for each annotator
            dist = ap.annotator(sub_directory)
            dist.state_dcc_directory(args.parent_dir)
            dist.make_subdirectories()
            dist.correct_inconsistent_labels_xml()   
            
    if args.complete_dir is not None:
        dist = ap.annotator(os.path.basename(args.complete_dir))
        dist.state_dcc_directory(os.path.dirname(args.complete_dir))
        dist.make_subdirectories()
        dist.correct_inconsistent_labels_xml()   

if __name__ == '__main__':
    ### Get the arguments 
    args = get_args_parse()
    main(args)


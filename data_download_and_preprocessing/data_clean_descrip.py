"""
Data Summary of Labeled Images (To - Date)
"""

"""
Import Packages
"""
from PIL import Image
import os
import pandas as pd
import numpy as np
import shutil
from lxml.etree import Element,SubElement,tostring
import xml.dom.minidom
from xml.dom.minidom import parseString
import xml.etree.ElementTree as ET
import argparse

import os
import sys
from glob import glob
import data_eng.az_proc as ap

def main():
    parser = argparse.ArgumentParser(
        description='This script adds a subdirectory of xmls to correct possible inconsistent labels')
    parser.add_argument('--parent_directory', type=str, default=None,
                        help='path to parent directory, holding the annotation directory.')
    parser.add_argument('--tiles_remaining', type=str, default=None,
                        help='The name of the numpy array specifying the tiles that remain to be annotated.')
    parser.add_argument('--tiles_labeled', type=str, default=None,
                        help='The name of the numpy array specifying the tiles that have been annotated.')
    args = parser.parse_args()
    return args.parent_directory, args.tiles_remaining, args.tiles_labeled

    
if __name__ == '__main__':
    parent_directory, tiles_remaining, tiles_labeled = main()
    ap.dataset_summary_assessment(ap.img_path_anno_path(ap.list_of_sub_directories(parent_directory)))
    #ap.tile_progress(tiles_labeled, tiles_remaining)
    
    


        
        


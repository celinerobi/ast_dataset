#Make into module, function specify annotator, directory as args
#URG: updae tile_undone_npy

#!/usr/bin/env python
# coding: utf-8
#python modules
import os
import os.path
import urllib
import urllib.request
import shutil
import tempfile
import argparse
import math
import warnings
from zipfile import ZipFile
import pickle
from contextlib import suppress
import sys

#python packages
import numpy as np
import pandas as pd
import progressbar # pip install progressbar2, not progressbar
import PIL
from PIL import Image
import rtree
from geopy.geocoders import Nominatim
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

import data_eng.az_proc as ap

def main():
    parser = argparse.ArgumentParser(
        description='This script supports seperating positive and negative chips after annotations')
    parser.add_argument('--annotation_directory', type=str, default=None,
                        help='path to annotation files directory.')
    parser.add_argument('--parent_directory', type=str, default=None,
                        help='path to parent directory, holding the annotation directory.')
    args = parser.parse_args()
    return args.annotation_directory, args.parent_directory

    
if __name__ == '__main__':
    annotation_directory, parent_directory = main()
    #create the processing class
    dist = ap.annotator(annotation_directory)
    dist.state_dcc_directory(parent_directory)
    dist.make_subdirectories()
    
    #check if positive directory is created)
    print(dist.chips_positive_dir)
    
    #seperate out positive and negative images
    dist.copy_positive_images()
    dist.copy_negative_images()
    
    

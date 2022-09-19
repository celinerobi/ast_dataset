import os
import cv2
import copy
import argparse

import math
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from glob import glob
import tqdm
from skimage.metrics import structural_similarity as compare_ssim
import shutil
#Parsing/Modifying XML
from lxml.etree import Element,SubElement,tostring
import xml.dom.minidom
from xml.dom.minidom import parseString
import xml.etree.ElementTree as et
from xml.dom import minidom

import data_eng.az_proc as ap
import data_eng.form_calcs as fc

def get_args_parse():
    parser = argparse.ArgumentParser(
        description='This script adds a subdirectory of xmls to correct possible inconsistent labels')
    parser.add_argument('--tile_name', type=str, default=None,
                        help='tile name.')
    parser.add_argument('--compile_dir', type=str, default=None,
                        help='path to dir to store all correct images.')
    parser.add_argument('--tile_dir_path', type=str, default=None,
                        help='path to all tiles')
    args = parser.parse_args()
    return args

def main(args)
    correct_chip_dir_path = os.path.join(args.compile_dir, args.tile_name, "chips")
    os.makedirs(correct_chip_dir_path, exist_ok=True)        
    fc.make_tile_dir_and_get_correct_imgs(args.tile_name, args.compile_dir, args.tile_dir_path, correct_chip_dir_path)
     
if __name__ == '__main__':
    ### Get the arguments 
    args = get_args_parse()
    main(args)


    
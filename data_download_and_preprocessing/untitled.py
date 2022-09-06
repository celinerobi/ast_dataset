import os
import cv2
import copy

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

from multiprocessing import Pool
import multiprocessing as mp

import data_eng.az_proc as ap
import data_eng.form_calcs as fc

def get_args_parse():
    parser = argparse.ArgumentParser(
        description='This script adds a subdirectory of xmls to correct possible inconsistent labels')
    parser.add_argument('--verified_state_year_subfolders_path', type=str, default=None,
                        help='path to dir containing verified data with state year formats.')
    parser.add_argument('--verified_standard_quad_subfolders_path', type=str, default=None,
                        help='path to dir containing verified data with standard quad format.')
    parser.add_argument('--directory', type=bool, default=False,
                        help='use original (True), or corrected (False) annotations')
    args = parser.parse_args()
    return args

def main(args):
    all_tile_names =  fc.read_list(os.path.join(directory,"tile_names.json"))
    state_year_img_paths = fc.read_list(os.path.join(directory,"state_year_img_paths.json"))
    state_year_xml_paths = fc.read_list(os.path.join(directory,"state_year_xml_paths.json"))

    six_digit_index_list = fc.read_list(os.path.join(directory,"six_digit_index_list.json"))
    
if __name__ == '__main__':
    ### Get the arguments 
    args = get_args_parse()
    main(args)

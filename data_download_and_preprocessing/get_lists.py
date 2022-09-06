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
    all_verified_state_year_subfolders_path = ap.img_path_anno_path(ap.list_of_sub_directories(args.verified_state_year_subfolders_path)) 
    all_verified_standard_quad_subfolders_path = ap.img_path_anno_path(ap.list_of_sub_directories(args.verified_standard_quad_subfolders_path)) 
    all_verified_paths = np.concatenate((all_verified_state_year_subfolders_path, all_verified_standard_quad_subfolders_path))
    
    all_img_paths, all_xml_paths = fc.get_img_xml_paths(all_verified_paths)
    all_tile_names, all_img_names = fc.get_tile_names(all_img_paths) #identify tiles in each folder
    all_tile_names = np.unique(all_tile_names)
    fc.write_list(all_tile_names, os.path.join(directory,"tile_names.json"))
    
    state_year_img_paths, state_year_xml_paths = fc.get_img_xml_paths(all_verified_state_year_subfolders_path)
    fc.write_list(state_year_img_paths, os.path.join(directory,"state_year_img_paths.json"))
    fc.write_list(state_year_xml_paths, os.path.join(directory,"state_year_xml_paths.json"))

    six_digit_index_list = fc.get_six_digit_index_from_img_path(state_year_img_paths)
    fc.write_list(six_digit_index_list, os.path.join(directory,"six_digit_index_list.json"))
    
if __name__ == '__main__':
    ### Get the arguments 
    args = get_args_parse()
    main(args)


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
    parser.add_argument('--verified_state_year_subfolders_path', type=str, default=None,
                        help='path to dir containing verified data with state year formats.')
    parser.add_argument('--tile_name', type=str, default=None,
                        help='tile name.')
    parser.add_argument('--compile_dir', type=str, default=None,
                        help='path to dir to store all correct images.')
    parser.add_argument('--correct_img_path', type=str, default=False,
                        help='path to all tiles')
    parser.add_argument('--directory', type=str, default=None,
                        help='use original (True), or corrected (False) annotations')
    
    #parser.add_argument('--y', type=str, default=None,
    #                    help='y idx for a given tile.')
    #parser.add_argument('--x', type=str, default=False,
    #                    help='x index for given tile')
    #parser.add_argument('--six_digit_idx', type=str, default=False,
    #                    help='six_digit_idx for a given tile')
    args = parser.parse_args()
    
    return args

def main(args):
    all_verified_state_year_subfolders_path = ap.img_path_anno_path(ap.list_of_sub_directories(args.verified_state_year_subfolders_path)) 
    state_year_img_paths, state_year_xml_paths = fc.get_img_xml_paths(all_verified_state_year_subfolders_path)
    state_year_six_digit_idx_list = fc.get_six_digit_index_from_img_path(state_year_img_paths)

    #state_year_img_paths = fc.read_list(os.path.join(args.directory,"state_year_img_paths.json"))
    #state_year_xml_paths = fc.read_list(os.path.join(args.directory,"state_year_xml_paths.json"))
    #all_img_six_digit_idx_list = fc.read_list(os.path.join(args.directory,"six_digit_idxs.json"))
    fc.compare_imgs_xmls_x_y_index_dcc(args.tile_name, args.correct_img_path, state_year_six_digit_idx_list, state_year_img_paths, state_year_xml_paths, args.compile_dir)
    
if __name__ == '__main__':
    ### Get the arguments 
    args = get_args_parse()
    main(args)
    
 
    

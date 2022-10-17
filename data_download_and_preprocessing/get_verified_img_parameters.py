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
import data_eng.compare as compare
def get_args_parse():
    parser = argparse.ArgumentParser(
        description='This script adds a subdirectory of xmls to correct possible inconsistent labels')
    parser.add_argument('--verified_state_year_subfolders_path', type=str, default=None,
                        help='path to state year verified img subfolder.')
    parser.add_argument('--verified_standard_quad_subfolders_path', type=str, default=None,
                        help='path to standard verified img subfolder.')
    parser.add_argument('--param_directory', type=str, default=None,
                        help='path to directory to hold parameters')
    args = parser.parse_args()
    return args

def main(args):
    #state year
    all_verified_state_year_subfolders_path = ap.img_path_anno_path(ap.list_of_sub_directories(args.verified_state_year_subfolders_path)) 
    state_year_img_paths, state_year_xml_paths = fc.get_img_xml_paths(all_verified_state_year_subfolders_path)
    fc.write_list(state_year_img_paths, os.path.join(args.param_directory, "state_year_img_paths.json"))
    fc.write_list(state_year_xml_paths, os.path.join(args.param_directory, "state_year_xml_paths.json"))
    ##six digit idx
    state_year_six_digit_idx_list = compare.get_six_digit_index_from_img_path(state_year_img_paths)
    fc.write_list(state_year_six_digit_idx_list, os.path.join(args.param_directory,"state_year_six_digit_idx_list.json"))
    #standard quad
    all_verified_standard_quad_subfolders_path = ap.img_path_anno_path(ap.list_of_sub_directories(args.verified_standard_quad_subfolders_path)) 
    standard_img_paths, standard_xml_paths = fc.get_img_xml_paths(all_verified_standard_quad_subfolders_path)
    fc.write_list(standard_img_paths, os.path.join(args.param_directory,"standard_img_paths.json"))
    fc.write_list(standard_xml_paths, os.path.join(args.param_directory,"standard_xml_paths.json"))
    #yx indx
    yx_array = compare.get_x_y_index(standard_img_paths)
    np.save(os.path.join(args.param_directory,"yx_array.npy"), yx_array)
    
if __name__ == '__main__':
    ### Get the arguments 
    args = get_args_parse()
    main(args)


    
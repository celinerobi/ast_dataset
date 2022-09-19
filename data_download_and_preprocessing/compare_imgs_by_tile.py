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
    parser.add_argument('--compile_dir', type=str, default=None,
                        help='path to dir to store all correct images.')
    parser.add_argument('--by_tile_correct_chips_dir', type=str, default=False,
                        help='path to all tiles')
    parser.add_argument('--param_directory', type=str, default=None,
                        help='use original (True), or corrected (False) annotations')
    args = parser.parse_args()
    
    return args

def main(args):
    state_year_img_paths = fc.read_list(os.path.join(args.param_directory,"state_year_img_paths.json"))
    state_year_xml_paths = fc.read_list(os.path.join(args.param_directory,"state_year_xml_paths.json"))
    state_year_six_digit_idx_list = fc.read_list(os.path.join(args.param_directory,"state_year_six_digit_idx_list.json"))
    
    standard_img_paths = fc.read_list(os.path.join(args.param_directory,"standard_img_paths.json"))
    standard_xml_paths = fc.read_list(os.path.join(args.param_directory,"standard_xml_paths.json"))
    yx_list = np.load(os.path.join(args.param_directory,"yx_list.npy"))

    fc.remove_thumbs(args.by_tile_correct_chips_dir)
    by_tile_correct_chips_paths = glob(args.by_tile_correct_chips_dir + "/*.jpg", recursive = True)
    
    for correct_img_path in by_tile_correct_chips_paths:
        correct_img = cv2.imread(correct_img_path)
        if np.sum(correct_img) != 0:
            fc.compare_imgs_state_year_standard_from_six_digit_xy_idxs_dcc(correct_img, correct_img_path, args.compile_dir,
                                                       state_year_six_digit_idx_list, state_year_img_paths, state_year_xml_paths,
                                                       yx_list, standard_img_paths, standard_xml_paths)


if __name__ == '__main__':
    ### Get the arguments 
    args = get_args_parse()
    main(args)
    
 
    

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
    parser.add_argument('--compile_dir', type=str, default=None,
                        help='path to dir to store all correct images.')
    parser.add_argument('--by_tile_correct_chips_wo_black_sq_dir_path', type=str, default=False,
                        help='path to correct chips without black pixels')
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
    yx_array = np.load(os.path.join(args.param_directory,"yx_array.npy"))
                                                            
    #fc.remove_thumbs(args.by_tile_correct_chips_w_black_sq_dir_path)
    fc.remove_thumbs(args.by_tile_correct_chips_wo_black_sq_dir_path)

    by_tile_correct_chips_wo_black_sq_dir_paths = sorted(glob(args.by_tile_correct_chips_wo_black_sq_dir_path + "/*.jpg", recursive = True))
    print(len(by_tile_correct_chips_wo_black_sq_dir_paths))
    for by_tile_correct_chips_wo_black_sq_dir_path in by_tile_correct_chips_wo_black_sq_dir_paths:
        correct_img_wo_black_sq = cv2.imread(by_tile_correct_chips_wo_black_sq_dir_path)
        compare.compare_imgs_wo_blk_pxls_state_yr_std_from_6_digit_xy_idxs_test(0.925, correct_img_wo_black_sq, by_tile_correct_chips_wo_black_sq_dir_path, args.compile_dir, 
                                                                           state_year_six_digit_idx_list, state_year_img_paths, state_year_xml_paths,
                                                                           yx_array, standard_img_paths, standard_xml_paths)
if __name__ == '__main__':
    ### Get the arguments 
    args = get_args_parse()
    main(args)
    
 
    

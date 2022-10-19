import os
import cv2
import argparse

import numpy as np
import pandas as pd
import shutil

from glob import glob

import data_eng.form_calcs as fc
import data_eng.compare as compare


def get_args_parse():
    parser = argparse.ArgumentParser(
        description='This script adds a subdirectory of xmls to correct possible inconsistent labels')
    parser.add_argument('--compile_dir', type=str, default=None,
                        help='path to dir to store all correct images.')
    parser.add_argument('--by_tile_correct_chips_wo_black_sq_dir', type=str, default=False,
                        help='path to directory correct chips without black pixels')
    parser.add_argument('--param_directory', type=str, default=None,
                        help='use original (True), or corrected (False) annotations')
    args = parser.parse_args()
    return args


def main(args):
    state_year_img_paths = fc.read_list(os.path.join(args.param_directory, "state_year_img_paths.json"))
    state_year_xml_paths = fc.read_list(os.path.join(args.param_directory, "state_year_xml_paths.json"))
    state_year_six_digit_idx_list = fc.read_list(os.path.join(args.param_directory, "state_year_six_digit_idx_list.json"))
    
    standard_img_paths = fc.read_list(os.path.join(args.param_directory, "standard_img_paths.json"))
    standard_xml_paths = fc.read_list(os.path.join(args.param_directory, "standard_xml_paths.json"))
    yx_array = np.load(os.path.join(args.param_directory, "yx_array.npy"))

    fc.remove_thumbs(args.by_tile_correct_chips_wo_black_sq_dir)
    correct_chips_wo_black_sq_paths = sorted(glob(args.by_tile_correct_chips_wo_black_sq_dir + "/*.jpg", recursive=True))
    print(len(correct_chips_wo_black_sq_paths))
    for correct_chip_wo_black_sq_path in correct_chips_wo_black_sq_paths:
        correct_chip_wo_black_sq = cv2.imread(correct_chip_wo_black_sq_path)
        compare.compare_imgs_wo_blk_pxls_state_yr_std_from_6_digit_xy_idxs_test(correct_chip_wo_black_sq,
                        correct_chip_wo_black_sq_path, args.compile_dir, state_year_six_digit_idx_list,
                        state_year_img_paths, state_year_xml_paths, yx_array, standard_img_paths, standard_xml_paths)

if __name__ == '__main__':
    # Get the arguments
    args = get_args_parse()
    main(args)

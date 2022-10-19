import os
import cv2
import copy
import argparse

import math
import pandas as pd
import numpy as np
from glob import glob
import tqdm
from skimage.metrics import structural_similarity as compare_ssim
import shutil

import data_eng.az_proc as ap
import data_eng.form_calcs as fc


def specify_compare_threshold(row_dim, col_dim, verified_img_name, correct_img_name):
    if min(row_dim, col_dim) <= 25:  # compare function has a minimum window set to 3 pixels
        return 0.925
    else:
        if verified_img_name == correct_img_name:
            return 0.875
        else:
            return 0.9


def img_path_to_std_img_name(img_path):
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    if img_name.count("_") > 9:
        std_img_name = img_name.split("_", 4)[-1]  # state included in image name
        return std_img_name
    else:
        return img_name


###################################################################################
###############    verified image parameter functions   ###########################
###################################################################################

def get_six_digit_index_from_img_path(state_year_img_paths):
    """ Create folder to store verified imagery by tile 
    Args:
        state_year_img_paths(list): list of paths to verified images with state year file name formatting 
    Returns:
        six_digit_index(list): list with six digit indices corresponding to position of image within tile
    """
    six_digit_index = []
    for img_path in state_year_img_paths:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        assert img_name.count("_") > 9, "Not state year format"
        six_digit_index.append(img_name.rsplit("_", 1)[-1])
    return (six_digit_index)


def get_x_y_index(standard_img_paths):
    """ Create folder to store verified imagery by tile 
    Args:
        state_year_img_paths(list): list of paths to verified images with state year file name formatting 
    Returns:
        yx_array(array): array with y,x indices corresponding corresponding to position of image within tile
    """
    xs = []
    ys = []
    for img_path in standard_img_paths:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        assert (img_name.count("_") < 9) and (img_name.split("_", 1)[0] == "m"), "Not standard format"
        y, x = img_name.split("_")[-2:]  # y=row;x=col
        ys.append(y)
        xs.append(x)
    yx_array = np.vstack((ys, xs)).T
    return (yx_array)


###################################################################################
###############        compile directory structure      ###########################
###################################################################################

def make_by_tile_dirs(root_dir, tile_name):
    """ Create folder to store verified imagery by tile 
    Args:
        root_dir(str): path of directory that will contain verified annotations and correct imagery by tile 
        tile_name(str): name of tile without of extension
    Returns:
        tile_dir(str): path to tile directory
    """
    # create folder to store corrected chips/xmls
    tile_dir = os.path.join(root_dir, tile_name)  # sub folder for each tile
    chips_positive_path = os.path.join(tile_dir, "chips_positive")  # images path
    chips_positive_xml_path = os.path.join(tile_dir, "chips_positive_xml")  # xmls paths
    os.makedirs(tile_dir, exist_ok=True)
    os.makedirs(chips_positive_path, exist_ok=True)
    os.makedirs(chips_positive_xml_path, exist_ok=True)
    return (tile_dir)


def read_tile(tile_path, item_dim=int(512)):
    """ Read tile and determine the corresponding number of img chips 
    Args:
        tile_path(str): path of tile in tif format
        item_dim(int): the dimensions of chipped images. Chipped images are assumed to be square.
    Returns:
        tile(array): numpy array corresponding to tile path
        row_index (int): the number of image chips along the y axis
        col_index (int): the number of image chips along the x axis
    """
    tile = cv2.imread(tile_path, cv2.IMREAD_UNCHANGED)
    tile_height, tile_width, tile_channels = tile.shape  # the size of the tile
    row_index = math.ceil(tile_height / item_dim)  # y
    col_index = math.ceil(tile_width / item_dim)  # x
    return (tile, row_index, col_index)


def make_tile_dir_and_get_correct_imgs(tile_name, root_dir, tile_dir, correct_chip_dir):
    """ For a given tile, create a directory to store verified images and chip and store correct image
    Args:
        tile_name(str): name of tile without of extension
        root_dir(str): path of directory that will contain verified annotations and correct imagery by tile 
        tile_dir(str): path to directory that stores all tiles (tif file format)
        correct_chip_dir(int): path to directory that stores all correctly chipped images 
    """
    compile_tile_dir = make_by_tile_dirs(root_dir, tile_name)  # make directory to store positive chips and xmls
    tile, row_index, col_index = read_tile(os.path.join(tile_dir, tile_name + ".tif"))  # read in tile

    count = 1
    for y in range(0,
                   row_index):  # rows #use row_index to account for the previous errors in state/year naming conventions
        for x in range(0, row_index):  # cols
            t_2_chip = fc.tile_to_chip_array(tile, x, y, int(512))  # get correct chip from tile
            six_digit_idx = str(count).zfill(6)
            cv2.imwrite(os.path.join(correct_chip_dir,
                                     tile_name + "-" + f"{y:02}" + "-" + f"{x:02}" + "-" + six_digit_idx + ".jpg"),
                        t_2_chip)  # save images
            count += 1


def make_tile_dir_and_get_correct_imgs_w_and_wo_black_sq(tile_name, root_dir, tile_dir,
                                                         correct_chip_w_black_sq_dir, correct_chip_wo_black_sq_dir):
    """ For a given tile, create a directory to store verified images and chip and store correct images (both with and witho black pixels added)
    Args:
        tile_name(str): name of tile without of extension
        root_dir(str): path of directory that will contain verified annotations and correct imagery by tile 
        tile_dir(str): path to directory that stores all tiles (tif file format)
        correct_chip_w_black_sq_dir(str): path to directory that stores all correctly chipped images (with the black squares to form a 512 x512 image included)
        correct_chip_wo_black_sq_dir(str): path to directory that stores all correctly chipped images  (with the black squares to form a 512 x512 image excluded)
    """
    compile_tile_dir = make_by_tile_dirs(root_dir, tile_name)  # make directory to store positive chips and xmls
    tile, row_index, col_index = read_tile(os.path.join(tile_dir, tile_name + ".tif"))  # read in tile
    item_dim = (int(512))
    count = 1
    for y in range(0,
                   row_index):  # rows #use row_index to account for the previous errors in state/year naming conventions
        for x in range(0, row_index):  # cols
            # define image name
            six_digit_idx = str(count).zfill(6)
            t_2_chip_wo_black_sq_img_name = tile_name + "-" + f"{y:02}" + "-" + f"{x:02}" + "-" + six_digit_idx + ".jpg"  # for compare analysis
            standard_quad_img_name_wo_ext = tile_name + '_' + f"{y:02}" + '_' + f"{x:02}" + ".jpg"  # row_col #for save

            # save images without black pixels added
            t_2_chip_wo_black_sq = tile[y * item_dim:y * item_dim + item_dim, x * (item_dim):x * (item_dim) + item_dim]
            if t_2_chip_wo_black_sq.size != 0:
                # write image without black pixels added
                cv2.imwrite(os.path.join(correct_chip_wo_black_sq_dir, t_2_chip_wo_black_sq_img_name),
                            t_2_chip_wo_black_sq)
                # write and save black pixels added
                t_2_chip_w_black_sq = fc.tile_to_chip_array(tile, x, y, int(512))  # get correct chip from tile
                cv2.imwrite(os.path.join(correct_chip_w_black_sq_dir, standard_quad_img_name_wo_ext),
                            t_2_chip_w_black_sq)  # write images
            count += 1

        ###################################################################################


###############                compare images           ###########################
###################################################################################

def compare_images(t_2_chip, labeled_img, compare_threshold):
    """ For a given tile, create a directory to store verified images and chip and store correct image
    Args:
        t_2_chip(array): image chipped from tile
        labeled_img(array): annotated image
        compare_threshold(str): Threshold for structural similarity score
    Return
        bool: Returns True if the structural similarity score between the two images is above the compare_threshold; Else False,
    """
    gray_t_2_chip = cv2.cvtColor(t_2_chip.astype(np.uint8), cv2.COLOR_BGR2GRAY)  # make gray
    gray_labeled_image = cv2.cvtColor(labeled_img.astype(np.uint8),
                                      cv2.COLOR_BGR2GRAY)  # image that has been chipped from tile

    score = compare_ssim(gray_t_2_chip, gray_labeled_image,
                         win_size=3)  # set window size so that is works on the edge peices
    if score >= compare_threshold:  # If the labeled image is correct
        # chip_name_incorrectly_chip_names[index]
        return (True)
    else:  # if it is incorrect
        ## move incorrectly named image if it one of the same name has not already been moved
        return (False)


def copy_and_replace_images_xml(img_name, img_path, xml_path, new_dir):
    """ copy a given image and xml file to a new directory
    Args:
        img_name(str): image name without the ext
        img_path(str): path to image 
        xml_path(str): path to xml file
        new_dir(str): path to new directory
    """
    ####    
    new_img_path = os.path.join(new_dir, "chips_positive", img_name + ".jpg")
    shutil.copy(img_path, new_img_path)

    new_xml_path = os.path.join(new_dir, "chips_positive_xml", img_name + ".xml")
    shutil.copy(xml_path, new_xml_path)  # destination


def compare_imgs_wo_blk_pxls_state_yr_std_from_6_digit_xy_idxs(compare_threshold, correct_img_wo_black_sq,
                                                               correct_img_wo_black_sq_path, compile_dir,
                                                               state_year_six_digit_idx_list, state_year_img_paths,
                                                               state_year_xml_paths,
                                                               yx_array, standard_img_paths, standard_xml_paths):
    """ copy a given image and xml file to a new directory
    Args:
        compare_threshold(int): Threshold for structural similarity score
        correct_img_wo_black_sq(array): path to directory that stores all correctly chipped images (with the black squares to form a 512 x512 image included)
        correct_img_wo_black_sq_path(str): path to directory that stores all correctly chipped images  (with the black squares to form a 512 x512 image excluded)
        compile_dir(str):
        six_digit_index(list): list with six digit indices corresponding to position of image within tile
        yx_array(array): array with y,x indices corresponding to position of image within tile

    """
    # process correct img (wo black sq) info
    correct_img_name = os.path.splitext(os.path.basename(correct_img_wo_black_sq_path))[0]  # get correct img name
    row_dim = correct_img_wo_black_sq.shape[0]  # get row dim
    col_dim = correct_img_wo_black_sq.shape[1]  # get col dim
    if min(row_dim, col_dim) <= 25:  # compare function has a minimum windo set to 3 pixels
        compare_threshold = 0.95
    else:
        compare_threshold = 0.90

    if min(row_dim, col_dim) >= 3:  # compare function has a minimum windo set to 3 pixels
        tile_name, y, x, six_digit_idx = correct_img_name.rsplit("-",
                                                                 3)  # identify tile name and indicies from correct img name
        by_tile_dir = os.path.join(compile_dir, tile_name)  # sub folder for correct directory

        # get standard and state idxs that match the correct img
        state_idxs, = np.where(np.array(state_year_six_digit_idx_list) == six_digit_idx)
        standard_idxs, = np.where((yx_array == (y, x)).all(axis=1))
        # turn the y/x into integers
        y = int(y)
        x = int(x)
        standard_quad_img_name_wo_ext = tile_name + '_' + f"{y:02}" + '_' + f"{x:02}"  # (row_col) get standard and state_year img_names
        # identify imgs/xmls that match the chip position (state imgs)
        for idx in state_idxs:
            # get verified img/xml path
            img_path = state_year_img_paths[idx]
            xml_path = state_year_xml_paths[idx]
            img = cv2.imread(img_path)
            img = img[0:row_dim, 0:col_dim]
            if (np.sum(img) != 0) & (compare_images(correct_img_wo_black_sq, img,
                                                    compare_threshold)):  # move images if they are 1) not all black 2)match the correct image
                copy_and_replace_images_xml(standard_quad_img_name_wo_ext, img_path, xml_path,
                                            by_tile_dir)  # use standard name and copy to compiled directory

        # identify imgs/xmls that match the chip position (standard imgs)
        for idx in standard_idxs:
            img_path = standard_img_paths[idx]
            xml_path = standard_xml_paths[idx]
            img = cv2.imread(img_path)
            img = img[0:row_dim, 0:col_dim]
            if (np.sum(img) != 0) & (compare_images(correct_img_wo_black_sq, img, compare_threshold)):
                copy_and_replace_images_xml(standard_quad_img_name_wo_ext, img_path, xml_path,
                                            by_tile_dir)  # use standard name and copy to compiled directory


def compare_imgs_wo_blk_pxls_state_yr_std_from_6_digit_xy_idxs_test1(compare_threshold, correct_img_wo_black_sq,
                                                                     correct_img_wo_black_sq_path, compile_dir,
                                                                     state_year_six_digit_idx_list,
                                                                     state_year_img_paths, state_year_xml_paths,
                                                                     yx_array, standard_img_paths, standard_xml_paths):
    # process correct img (wo black sq) info
    correct_img_name = os.path.splitext(os.path.basename(correct_img_wo_black_sq_path))[0]  # get correct img name
    print(correct_img_name)
    row_dim = correct_img_wo_black_sq.shape[0]  # get row dim
    col_dim = correct_img_wo_black_sq.shape[1]  # get col dim
    if min(row_dim, col_dim) >= 3:  # compare function has a minimum window set to 3 pixels
        tile_name, y, x, six_digit_idx = correct_img_name.rsplit("-",
                                                                 3)  # identify tile name and indicies from correct img name
        by_tile_dir = os.path.join(compile_dir, tile_name)  # sub folder for correct directory

        # get standard and state idxs that match the correct img
        state_idxs, = np.where(np.array(state_year_six_digit_idx_list) == six_digit_idx)
        standard_idxs, = np.where((yx_array == (y, x)).all(axis=1))
        # turn the y/x into integers
        y = int(y)
        x = int(x)
        standard_quad_img_name_wo_ext = tile_name + '_' + f"{y:02}" + '_' + f"{x:02}"  # (row_col) get standard and state_year img_names

        # identify imgs/xmls that match the chip position (state imgs)
        scores = []
        for idx in state_idxs:
            # get verified img/xml path
            img_path = state_year_img_paths[idx]
            xml_path = state_year_xml_paths[idx]
            img = cv2.imread(img_path)
            img = img[0:row_dim, 0:col_dim]
            # if (np.sum(img) != 0) & (compare_images(correct_img_wo_black_sq, img)): #only move images if they are not all black and they match the correct image
            if (np.sum(img) != 0):
                match, scores = compare_images(correct_img_wo_black_sq, img,
                                               scores)  # only move images if they are not all black and they match the correct image
                if match:
                    copy_and_replace_images_xml(standard_quad_img_name_wo_ext, img_path, xml_path,
                                                by_tile_dir)  # use standard name and copy to compiled directory
        if len(scores) > 0:
            print(max(scores))

        # identify imgs/xmls that match the chip position (standard imgs)
        for idx in standard_idxs:
            img_path = standard_img_paths[idx]
            xml_path = standard_xml_paths[idx]
            img = cv2.imread(img_path)
            img = img[0:row_dim, 0:col_dim]
            # if (np.sum(img) != 0) & (compare_images(correct_img_wo_black_sq, img)):
            # copy_and_replace_images_xml(standard_quad_img_name_wo_ext, img_path, xml_path, by_tile_dir) #use standard name and copy to compiled directory
            # print("match", correct_img_path, img_path)
            if (np.sum(img) != 0):
                match, scores = compare_images(correct_img_wo_black_sq, img,
                                               scores)  # only move images if they are not all black and they match the correct image
                if match:
                    copy_and_replace_images_xml(standard_quad_img_name_wo_ext, img_path, xml_path,
                                                by_tile_dir)  # use standard name and copy to compiled directory
        if len(scores) > 0:
            print(max(scores))


def compare_images_test(t_2_chip, labeled_img, scores):
    gray_t_2_chip = cv2.cvtColor(t_2_chip.astype(np.uint8), cv2.COLOR_BGR2GRAY)  # make gray
    gray_labeled_image = cv2.cvtColor(labeled_img.astype(np.uint8),
                                      cv2.COLOR_BGR2GRAY)  # image that has been chipped from tile

    score = compare_ssim(gray_t_2_chip, gray_labeled_image,
                         win_size=3)  # set window size so that is works on the edge peices
    scores.append(score)
    return (scores)


def compare_imgs_wo_blk_pxls_state_yr_std_from_6_digit_xy_idxs_test(correct_img_wo_black_sq,
                                                                    correct_img_wo_black_sq_path,
                                                                    compile_dir, state_year_six_digit_idx_list,
                                                                    state_year_img_paths, state_year_xml_paths,
                                                                    yx_array, standard_img_paths, standard_xml_paths):
    # process correct img (wo black sq) info
    correct_img_name = os.path.splitext(os.path.basename(correct_img_wo_black_sq_path))[0]  # get correct img name
    row_dim = correct_img_wo_black_sq.shape[0]  # get row dim
    col_dim = correct_img_wo_black_sq.shape[1]  # get col dim
    if min(row_dim, col_dim) <= 25:  # compare function has a minimum windo set to 3 pixels
        compare_threshold = 0.925
    else:
        compare_threshold = 0.90

    if min(row_dim, col_dim) >= 3:  # compare function has a minimum window set to 3 pixels
        tile_name, y, x, six_digit_idx = correct_img_name.rsplit("-",
                                                                 3)  # identify tile name and indicies from correct img name
        by_tile_dir = os.path.join(compile_dir, tile_name)  # sub folder for correct directory

        # get standard and state idxs that match the correct img
        state_idxs, = np.where(np.array(state_year_six_digit_idx_list) == six_digit_idx)
        standard_idxs, = np.where((yx_array == (y, x)).all(axis=1))
        # turn the y/x into integers
        y = int(y)
        x = int(x)
        standard_quad_img_name_wo_ext = tile_name + '_' + f"{y:02}" + '_' + f"{x:02}"  # (row_col) get standard and state_year img_names

        # identify imgs/xmls that match the chip position (state imgs)
        scores = []
        img_paths = []
        xml_paths = []
        for idx in state_idxs:
            # get verified img/xml path
            img_path = state_year_img_paths[idx]
            xml_path = state_year_xml_paths[idx]
            img = cv2.imread(img_path)
            img = img[0:row_dim, 0:col_dim]

            if (np.sum(img) != 0):
                img_paths.append(img_path)
                xml_paths.append(xml_path)
                scores = compare_images_test(correct_img_wo_black_sq, img,
                                             scores)  # only move images if they are not all black and they match the correct image

        # identify imgs/xmls that match the chip position (standard imgs)
        for idx in standard_idxs:
            img_path = standard_img_paths[idx]
            xml_path = standard_xml_paths[idx]
            img = cv2.imread(img_path)
            img = img[0:row_dim, 0:col_dim]

            if (np.sum(img) != 0):
                img_paths.append(img_path)
                xml_paths.append(xml_path)
                scores = compare_images_test(correct_img_wo_black_sq, img,
                                             scores)  # only move images if they are not all black and they match the correct image
        # print(max(scores))
        matches = pd.DataFrame(data={'scores': scores, 'img_paths': img_paths, 'xml_paths': xml_paths})
        match = matches.loc[matches['scores'] == max(scores)].iloc[0]
        print(correct_img_name, row_dim, col_dim)
        print(compare_threshold, match["scores"], match["img_paths"], match["xml_paths"])
        if match["scores"] > compare_threshold:
            print("match")
            # copy_and_replace_images_xml(standard_quad_img_name_wo_ext, match["img_paths"], match["xml_paths"], by_tile_dir) #use standard name and copy to compiled directory

import os
import cv2
import copy
import argparse

import math
import numpy as np
from glob import glob
import tqdm
from skimage.metrics import structural_similarity as compare_ssim
import shutil

import data_eng.az_proc as ap
import data_eng.form_calcs as fc
import data_eng.compare as compare

def make_by_tile_dirs(home_dir, tile_name):
    #create folder to store corrected chips/xmls
    tile_dir = os.path.join(home_dir, tile_name) #sub folder for each tile 
    chips_positive_path = os.path.join(tile_dir,"chips_positive") #images path
    chips_positive_xml_path = os.path.join(tile_dir,"chips_positive_xml") #xmls paths
    os.makedirs(tile_dir, exist_ok=True)
    os.makedirs(chips_positive_path, exist_ok=True)
    os.makedirs(chips_positive_xml_path, exist_ok=True)
    return(tile_dir)

def read_tile(tile_path, item_dim = int(512)):
    tile = cv2.imread(tile_path, cv2.IMREAD_UNCHANGED) 
    tile_height,  tile_width,  tile_channels = tile.shape #the size of the tile 
    row_index = math.ceil(tile_height/item_dim) #y
    col_index = math.ceil(tile_width/item_dim) #x
    return(tile, row_index, col_index)

def make_tile_dir_and_get_correct_imgs(tile_name, compile_dir_path, tile_dir_path, correct_chip_dir_path):
    compile_tile_dir = make_by_tile_dirs(compile_dir_path, tile_name) #make directory to store positive chips and xmls
    tile, row_index, col_index = read_tile(os.path.join(tile_dir_path, tile_name + ".tif")) #read in tile
    
    count = 1
    for y in range(0, row_index): #rows #use row_index to account for the previous errors in state/year naming conventions
        for x in range(0, row_index): #cols   
            t_2_chip = tile_to_chip_array(tile, x, y, int(512)) #get correct chip from tile
            six_digit_idx = str(count).zfill(6)
            cv2.imwrite(os.path.join(correct_chip_dir_path, tile_name + "-" + f"{y:02}"  + "-" + f"{x:02}" + "-" + six_digit_idx+".jpg"), t_2_chip) #save images  
            count += 1  

def make_tile_dir_and_get_correct_imgs_w_and_wo_black_sq(tile_name, compile_dir_path, tile_dir_path,
                                                         correct_chip_w_black_sq_dir_path,correct_chip_wo_black_sq_dir_path):
    compile_tile_dir = make_by_tile_dirs(compile_dir_path, tile_name) #make directory to store positive chips and xmls
    tile, row_index, col_index = read_tile(os.path.join(tile_dir_path, tile_name + ".tif")) #read in tile
    item_dim = (int(512))
    count = 1
    for y in range(0, row_index): #rows #use row_index to account for the previous errors in state/year naming conventions
        for x in range(0, row_index): #cols  
            #define image name
            six_digit_idx = str(count).zfill(6)
            t_2_chip_wo_black_sq_img_name=tile_name + "-" + f"{y:02}"  + "-" + f"{x:02}" + "-" + six_digit_idx + ".jpg" #for compare analysis
            standard_quad_img_name_wo_ext=tile_name + '_' + f"{y:02}"  + '_' + f"{x:02}" + ".jpg" # row_col #for save

            #save images without black pixels added  
            t_2_chip_wo_black_sq = tile[y*item_dim:y*item_dim+item_dim, x*(item_dim):x*(item_dim)+item_dim]
            if t_2_chip_wo_black_sq.size != 0:
                #write image without black pixels added 
                cv2.imwrite(os.path.join(correct_chip_wo_black_sq_dir_path, t_2_chip_wo_black_sq_img_name), t_2_chip_wo_black_sq) 
                #write and save black pixels added  
                t_2_chip_w_black_sq = fc.tile_to_chip_array(tile, x, y, int(512)) #get correct chip from tile
                cv2.imwrite(os.path.join(correct_chip_w_black_sq_dir_path, standard_quad_img_name_wo_ext), t_2_chip_w_black_sq) #write images 
            count += 1 

def compare_images(t_2_chip, labeled_img, compare_threshold):  
    gray_t_2_chip = cv2.cvtColor(t_2_chip.astype(np.uint8), cv2.COLOR_BGR2GRAY) # make gray
    gray_labeled_image = cv2.cvtColor(labeled_img.astype(np.uint8), cv2.COLOR_BGR2GRAY) #image that has been chipped from tile
    
    score = compare_ssim(gray_t_2_chip, gray_labeled_image, win_size = 3) #set window size so that is works on the edge peices 
    if score >= compare_threshold: #If the labeled image is correct
        #chip_name_incorrectly_chip_names[index]
        return(True)
    else: #if it is incorrect
        ## move incorrectly named image if it one of the same name has not already been moved
        return(False)
    
def copy_and_replace_images_xml(img_name, img_path, xml_path, copy_dir):                  
    ####    
    new_img_path = os.path.join(copy_dir, "chips_positive", img_name + ".jpg")
    shutil.copy(img_path, new_img_path)
        
    new_xml_path = os.path.join(copy_dir, "chips_positive_xml", img_name + ".xml")
    shutil.copy(xml_path, new_xml_path) #destination
    
def compare_imgs_wo_blk_pxls_state_yr_std_from_6_digit_xy_idxs(compare_threshold, correct_img_wo_black_sq, correct_img_wo_black_sq_path, compile_dir,
                                                               state_year_six_digit_idx_list, state_year_img_paths, state_year_xml_paths,
                                                               yx_list, standard_img_paths, standard_xml_paths):
    #process correct img (wo black sq) info
    correct_img_name = os.path.splitext(os.path.basename(correct_img_wo_black_sq_path))[0] #get correct img name
    row_dim = correct_img_wo_black_sq.shape[0] #get row dim
    col_dim = correct_img_wo_black_sq.shape[1] #get col dim
    if min(row_dim, col_dim) >= 3:#compare function has a minimum window set to 3 pixels
        tile_name, y, x, six_digit_idx = correct_img_name.rsplit("-",3) #identify tile name and indicies from correct img name
        by_tile_dir = os.path.join(compile_dir, tile_name) #sub folder for correct directory 

        #get standard and state idxs that match the correct img
        state_idxs, = np.where(np.array(state_year_six_digit_idx_list) == six_digit_idx)
        standard_idxs, = np.where((yx_list == (y, x)).all(axis=1))
        #turn the y/x into integers
        y = int(y)
        x = int(x)
        standard_quad_img_name_wo_ext = tile_name + '_' + f"{y:02}"  + '_' + f"{x:02}" # (row_col) get standard and state_year img_names

        #identify imgs/xmls that match the chip position (state imgs)
        for idx in state_idxs:
            #get verified img/xml path
            img_path = state_year_img_paths[idx]
            xml_path = state_year_xml_paths[idx]
            img = cv2.imread(img_path)
            img = img[0:row_dim, 0:col_dim]
            if (np.sum(img) != 0) & (compare_images(correct_img_wo_black_sq, img, compare_threshold)): #move images if they are 1) not all black 2)match the correct image
                copy_and_replace_images_xml(standard_quad_img_name_wo_ext, img_path, xml_path, by_tile_dir) #use standard name and copy to compiled directory 
                
        #identify imgs/xmls that match the chip position (standard imgs)
        for idx in standard_idxs:
            img_path = standard_img_paths[idx]
            xml_path = standard_xml_paths[idx]
            img = cv2.imread(img_path)
            img = img[0:row_dim, 0:col_dim]
            if (np.sum(img) != 0) & (compare_images(correct_img_wo_black_sq, img, compare_threshold)):
                copy_and_replace_images_xml(standard_quad_img_name_wo_ext, img_path, xml_path, by_tile_dir) #use standard name and copy to compiled directory
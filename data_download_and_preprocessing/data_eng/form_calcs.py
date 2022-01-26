"""
Functions to process, format, and conduct calculations on the annotated or verified dataset
"""
# Standard packages
import tempfile
import warnings
import urllib
import shutil
import os
# Less standard, but still pip- or conda-installable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#import rasterio
import re
import rtree
import shapely
import pickle
import data_eng.az_proc as ap
import data_eng.form_calcs as fc

#from cartopy import crs
import collections
import cv2
import math
from glob import glob
from tqdm.notebook import tqdm_notebook

def add_formatted_and_standard_tile_names_to_tile_names_time_urls(tile_names_tile_urls):
    #get a list of the formated tile names
    tile_names = []
    for tile_url in tile_names_tile_urls:
        tile_url = tile_url[1].rsplit("/",3)
        #get the quad standard tile name 
        tile_name = tile_url[3]
        tile_name = os.path.splitext(tile_name)[0] 
        #format tile_names to only include inital capture date 1/20
        if tile_name.count("_") > 5:
            tile_name = tile_name.rsplit("_",1)[0]
        #get the tile name formated (1/14/2022)
        tile_name_formatted = tile_url[1] + "_" + tile_url[2] + "_" + tile_url[3]
        tile_name_formatted = os.path.splitext(tile_name_formatted)[0] 
        tile_names.append([tile_name, tile_name_formatted])
    
    #create array that contains the formated tile_names and tile_names
    tile_names_tile_urls_formatted_tile_names = np.hstack((tile_names_tile_urls, np.array(tile_names)))
    
    return(tile_names_tile_urls_formatted_tile_names)


def unique_formatted_standard_tile_names(tile_names_tile_urls_complete_array):
    unique_tile_name_formatted, indicies = np.unique(tile_names_tile_urls_complete_array[:,3], return_index = True)
    tile_names_tile_urls_complete_array_unique_formatted_tile_names = tile_names_tile_urls_complete_array[indicies,:]
    print("unique formatted tile names", tile_names_tile_urls_complete_array_unique_formatted_tile_names.shape) 

    unique_tile_name_standard, indicies = np.unique(tile_names_tile_urls_complete_array[:,2], return_index = True)
    tile_names_tile_urls_complete_array_unique_standard_tile_names = tile_names_tile_urls_complete_array[indicies,:]
    print("unique standard tile names", tile_names_tile_urls_complete_array_unique_standard_tile_names.shape) 
    
    return(tile_names_tile_urls_complete_array_unique_standard_tile_names, tile_names_tile_urls_complete_array_unique_formatted_tile_names)

def jpg_path_to_tile_name_formatted(tile_paths):
    tile_names = []
    for tile_path in tile_paths:
        base = os.path.basename(tile_path)
        jpg = os.path.splitext(base)[0] #name of tif with the extension removed
        tile_name_formated_name = jpg.rsplit("_",1)[0] #name of tif with the extension removed
        tile_names.append(tile_name_formated_name)
    return(tile_names)

def jpg_path_to_jpg_name_formatted(jpg_paths):
    jpgs_ext = []
    jpgs_without_ext = []
    for jpg_path in jpg_paths:
        jpg_ext = os.path.basename(jpg_path)
        jpg_without_ext = os.path.splitext(jpg_ext)[0] #name of tif with the extension removed
        jpgs_ext.append(jpg_ext)
        jpgs_without_ext.append(jpg_without_ext)
    return(jpgs_ext, jpgs_without_ext)

def remove_thumbs(path_to_folder_containing_thumbs):
    """ Remove Thumbs.db file from a given folder
    Args: 
    path_to_folder_containing_thumbs(str): path to folder containing thumbs.db
    Returns:
    None
    """
    os.remove(glob(path_to_folder_containing_thumbs + "/*.db", recursive = True)[0])
    
def remove_thumbs_all_positive_chips(parent_directory):
    """ Remove Thumbs.db file from all chips_positive folders in parent directory
    Args: 
    parent_directory(str): path to parent directory
    Returns:
    None
    """
    for r, d, f in os.walk(parent_directory):
        folder_name = os.path.basename(r) #identify folder name
        if 'chips_positive' == folder_name: #Specify folders that contain positive chips
            if len(glob(r + "/*.db", recursive = True)) > 0:
                remove_thumbs(r)
    
def unique_positive_jpgs_from_parent_directory(parent_directory):
    files = []
    paths = []
    counter = 0
    # r=root, d=directories, f = files
    # https://mkyong.com/python/python-how-to-list-all-files-in-a-directory/
    for r, d, f in os.walk(parent_directory):
        folder_name = os.path.basename(r) #identify folder name
        if 'chips_positive' == folder_name: #Specify folders that contain positive chips
            for file in f:
                if '.jpg' in file:
                    paths.append(os.path.join(r, file))
                    files.append(file)
                    counter += 1
    positive_jpgs = np.array((files,paths)).T
    unique_tile_name_formatted_positive_jpgs, indicies = np.unique(positive_jpgs[:,0], return_index = True)
    unique_positive_jpgs = positive_jpgs[indicies]
    print(unique_positive_jpgs.shape)

    return(unique_positive_jpgs)


## Processing Tiles
def move_tiles_of_verified_images_to_complete_dataset(tile_img_annotation, tiles_complete_dataset_path, path_to_verified_sets):
    """Move already downloaded tiles to completed dataset
    """
    #obtain the paths of tifs in the verified sets
    path_to_tifs_in_verified_sets = glob(path_to_verified_sets + "/**/*.tif", recursive = True)
    print("Number of tifs to be moved", len(path_to_tifs_in_verified_sets))

    #move verified tifs 
    for path in path_to_tifs_in_verified_sets:
        base = os.path.basename(path)
        tif = os.path.splitext(base)[0] #name of tif with the extension removed
        if tif in tile_img_annotation[:,0]:
            shutil.move(path, os.path.join(tiles_complete_dataset_path,base)) # copy images with matching .xml files in the "chips_tank" folder
            
def tiles_in_complete_dataset(tiles_complete_dataset_path):
    #Make a list of the tiles in the completed dataset
    os.makedirs(tiles_complete_dataset_path, exist_ok=True)
    
    tiles_downloaded = os.listdir(tiles_complete_dataset_path)
    tiles_downloaded_with_ext_list = []
    tiles_downloaded_without_ext_list = []
    
    for tile in tiles_downloaded:
        tiles_downloaded_with_ext_list.append(tile)
        tiles_downloaded_without_ext_list.append(os.path.splitext(tile)[0]) #name of tif with the extension removed
    return(np.array(tiles_downloaded_with_ext_list), np.array(tiles_downloaded_without_ext_list))

def download_tiles_of_verified_images(path_positive_images_complete_dataset, tiles_complete_dataset_path, tiles_downloaded, tile_names_tile_urls_complete_array):
    """
    # Download remaining tiles that correspond to ONLY to verified images
    #Gather the locations of tiles that have already been downlaoded and verified 
    """
    jpg_path_positive_images_complete_dataset = glob(path_positive_images_complete_dataset + "/*.jpg", recursive = True)
    print("number of positive and verified images:", len(jpg_path_positive_images_complete_dataset))

    # Determine which tiles corresponding to jpg that have been annotated #jpg_tiles
    tiles_of_verified_positive_jpg = []
    for path in jpg_path_positive_images_complete_dataset:
        base = os.path.basename(path)
        img = os.path.splitext(base)[0] #name of tif with the extension removed
        tile = img.rsplit("_",1)[0]
        tile = tile.split("_",4)[4] #get the tile names to remove duplicates from being downloaded
        tiles_of_verified_positive_jpg.append(tile)
    tiles_of_verified_positive_jpg = np.unique(tiles_of_verified_positive_jpg)
    print("the number of tiles corresponding to verified images:", len(tiles_of_verified_positive_jpg))

    # Identify tiles that have not already been downloaded
    tile_names_to_download = []
    for tile in tiles_of_verified_positive_jpg: #index over the downloaded tiled
        if tile not in tiles_downloaded: #index over the tiles that should be downloded
            tile_names_to_download.append(tile)
    print("the number of tiles that need to be downloaded:", len(tile_names_to_download))
    
    # Download Tiles  
    destination_filenames = []
    for tile in tile_names_to_download:
        
        ### download the tiles if they are not in the tiles folder 
        #check if the tile name is contained in the string of complete arrays
        tile_name = [string for string in tile_names_tile_urls_complete_array[:,0] if tile_name in string]          
        if len(tile_name) == 1: #A single tile name # get tile url from the first (only) entry
            tile_url = tile_names_tile_urls_complete_array[tile_names_tile_urls_complete_array[:,0]==tile_name[0]][0][1] 
        elif len(np.unique(tile_name)) > 1: # Multiple (different tiles) possibly the same tiles in different states, possible different years
            tile_url = tile_names_tile_urls_complete_array[tile_names_tile_urls_complete_array[:,0]==tile_name[0]][0][1]# get tile url
        elif (len(tile_name) > 1): #Multiple different tile names that are the same, probably different naip storage locations
            # get tile url from the second entry 
            tile_url = tile_names_tile_urls_complete_array[tile_names_tile_urls_complete_array[:,0]==tile_name[1]][1][1] 
        
        #get file name
        file_name = tile_name[0]
        if tile_name[0].count("_") > 5:
            tile_name = tile_name[0].rsplit("_",1)[0]
            file_name = tile_name + ".tif"
        print(file_name)
        ### Download tile
        destination_filenames.append(ap.download_url(tile_url, tiles_complete_dataset_path,
                                                     destination_filename = file_name,       
                                                             progress_updater=ap.DownloadProgressBar()))
        """
        #destination_filenames.append(ap.download_url(tile_url, tiles_complete_dataset_path,
        #                                             destination_filename = tile_name,       
                                                             progress_updater=ap.DownloadProgressBar()))
        
    # Rename Tiles 
    for destination_filepath in destination_filenames: 
        tile_dir = destination_filepath.rsplit("\\",1)[0]
        tile_name = destination_filepath.rsplit("\\",1)[1]
        tile_name_split = tile_name.split('_')
        new_tile_path = os.path.join(tile_dir, tile_name_split[6]+ '_' + tile_name_split[7] + '_' + tile_name_split[8] + '_' + \
                                     tile_name_split[9] + '_' + tile_name_split[10] + '_' + tile_name_split[11] + '_' + \
                                     tile_name_split[12] + '_' + tile_name_split[13] + '_' + tile_name_split[14] + '_' + \
                                     tile_name_split[15].split(".")[0]+".tif")


        if os.path.isfile(new_tile_path):
            print('Bypassing download of already-downloaded file {}'.format(os.path.basename(new_tile_path)))
        else:
            os.rename(destination_filepath, new_tile_path)
        """
##
def image_characteristics(tiles_dir, unique_positive_jpgs):
    """
    Only characterisizes images for which the corresponding tile is downloaded
    Args:
    tiles_dir(str): path to the directory containing tiles
    Returns:
    image_characteristics(pandadataframe):containing image characterisitcs 
    """
    #initialize lists
    state = []
    resolution = []
    year = []
    capture_date  = []
    utm_zone  = []

    standard_tile_name = []
    six_digit_chip_name = []
    NW_coordinates = []
    SE_coordinates = []
    row_indicies = []
    col_indicies = []
    full_path  = []
    root = []
    for tile_name in tqdm_notebook(os.listdir(tiles_dir)): #index over the tiles in the tiles_dir 
        file_name, ext = os.path.splitext(tile_name) # File name
        count = 1      
        
        item_dim = int(512)          
        tile = cv2.imread(os.path.join(tiles_dir, tile_name)) 
        tile_height,  tile_width,  tile_channels = tile.shape #the size of the tile 

        #divide the tile into 512 by 512 chips (rounding up)
        row_index = math.ceil(tile_height/512) 
        col_index = math.ceil(tile_width/512)

        for x in range(0, col_index):
            for y in range(0, row_index):
                six_digit_chip_name_temp = file_name+ '_'+ str(count).zfill(6) + '.jpg'
                count += 1  
                if six_digit_chip_name_temp in unique_positive_jpgs[:,0]: #only record values for images that are annotated
                    #image characteristics
                    six_digit_chip_name.append(six_digit_chip_name_temp) # The index is a six-digit number like '000023'.
                    NW_coordinates.append([x*item_dim, y*(item_dim)]) #NW (Top Left) 
                    SE_coordinates.append([x*item_dim+item_dim-1, y*(item_dim)+item_dim-1]) #SE (Bottom right) 
                    row_indicies.append(y)
                    col_indicies.append(x)
                    #tile characteristics
                    standard_tile_name.append(file_name.split("_",4)[4]) #standard_tile_name
                    state.append(file_name.split("_",9)[0]) #state
                    resolution.append(file_name.split("_",9)[1]) #resolution
                    utm_zone.append(file_name.split("_",9)[7]) #utm
                    year.append(file_name.split("_",9)[2]) #year
                    capture_date.append(file_name.split("_",9)[-1]) #capture date
                    #path
                    full_path = unique_positive_jpgs[unique_positive_jpgs[:,0] == six_digit_chip_name_temp][0][1]
                    root = full_path.split("\\",2)[0]

    #create pandas dataframe
    image_characteristics = pd.DataFrame(data={ "root":root,
                                                'state': state,
                                                'resolution': resolution,
                                                'year': year,
                                                'capture_date':capture_date,
                                                'utm_zone': utm_zone,
                                                'standard_tile_name': standard_tile_name,
                                                'six_digit_chip_name':six_digit_chip_name,
                                                'NW_pixel_coordinates': NW_coordinates,
                                                'SE_pixel_coordinates': SE_coordinates,
                                                'row_indicies': row_indicies,
                                                'col_indicies': col_indicies})
    return(image_characteristics)


################################################################
#############  Format tiles in tiles folder  ###################
def formatted_tile_name_to_standard_tile_name(tile_name):
    #format tile_names to only include inital capture date 1/20
    tile_name = os.path.splitext(tile_name.split("_",4)[4])[0]
    if tile_name.count("_") > 5:
        tile_name = tile_name.rsplit("_",1)[0]
    tile_name_with_ext = tile_name + ".tif"
    return(tile_name_with_ext)

def rename_formatted_tiles(tiles_complete_dataset_path):
    """
    """
    for tile in os.listdir(tiles_complete_dataset_path):
        #format tile_names to only include inital capture date 1/20
        if tile.count("_") > 5:
            old_tile_path = os.path.join(tiles_complete_dataset_path, tile)
            
            new_tile_name = formatted_tile_name_to_standard_tile_name(tile)
            new_tile_path = os.path.join(tiles_complete_dataset_path, new_tile_name)
            
            if not os.path.exists(new_tile_path): #If the new tile path does not exist, convert the tile to standard format
                os.rename(old_tile_path, new_tile_path)
            if os.path.exists(new_tile_path) and os.path.exists(old_tile_path): #If the new tile path already exists, delete the old tile path (if it still exists)
                os.remove(old_tile_path)
                
#fc.remove_thumbs(os.path.join(parent_directory, complete_dataset_path,"tiles"))
#fc.rename_formatted_tiles(os.path.join(parent_directory, complete_dataset_path, "tiles"))
f"""
Functions to process, format, and conduct calculations on the annotated or verified dataset
"""
# Standard packages
#from __future__ import print_function

import warnings
import urllib
import shutil
import os
# Less standard, but still pip- or conda-installable
import numpy as np
import pandas as pd
#import rasterio
#import re
#import rtree
#import shapely
#import pickle
import data_eng.az_proc as ap
import cv2
import tqdm
from glob import glob
import math 

from skimage.metrics import structural_similarity as compare_ssim
import imutils
import psutil

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

def jpg_path_to_tile_name_formatted(jpg_paths):
    tile_names = []
    for jpg_path in jpg_paths:
        base = os.path.basename(jpg_path)
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

def remove_thumbs(path_to_folder_containing_images):
    """ Remove Thumbs.db file from a given folder
    Args: 
    path_to_folder_containing_images(str): path to folder containing images
    Returns:
    None
    """
    if len(glob(path_to_folder_containing_images + "/*.db", recursive = True)) > 0:
        os.remove(glob(path_to_folder_containing_images + "/*.db", recursive = True)[0])
    
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
            remove_thumbs(r)
    
def unique_positive_jpgs_from_parent_directory(parent_directory):
    files = []
    paths = []
    counter = 0
    # r=root, d=directories, f = files
    # https://mkyong.com/python/python-how-to-list-all-files-in-a-directory/
    for r, d, f in tqdm.tqdm(os.walk(parent_directory)):
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

def jpg_paths_to_tiles_without_ext(jpg_paths):
    """
    Determine which tiles corresponding to jpg that have been annotated #jpg_tiles
    Get a numpy array of the unique standard tile names corresponding to a list of jpg paths
    Args:
    jpgs_paths(list): list of jpg paths
    Returns:
    tiles(numpy): 
    """
    tiles = []
    for path in jpg_paths:
        base = os.path.basename(path)
        img = os.path.splitext(base)[0] #name of tif with the extension removed
        tile = img.rsplit("_",1)[0]
        tile = tile.split("_",4)[4] #get the tile names to remove duplicates from being downloaded
        tiles.append(tile)
    return(np.unique(tiles))

def download_tiles_of_verified_images(positive_images_complete_dataset_path, tiles_complete_dataset_path, tiles_downloaded, tile_names_tile_urls_complete_array):
    """
    # Download remaining tiles that correspond to ONLY to verified images
    #Gather the locations of tiles that have already been downlaoded and verified 
    """
    # Make a list of the tiles moved to completed dataset
    tiles_downloaded_with_ext, tiles_downloaded = tiles_in_complete_dataset(tiles_complete_dataset_path)
    
    positive_jpg_paths = glob(positive_images_complete_dataset_path + "/*.jpg", recursive = True)
    print("number of positive and verified images:", len(positive_jpg_paths))
    
    #  Determine which tiles corresponding to jpg that have been annotated #jpg_tiles
    positive_jpg_tiles = jpg_paths_to_tiles_without_ext(positive_jpg_paths)
    print("the number of tiles corresponding to verified images:", len(positive_jpg_tiles))

    # Identify tiles that have not already been downloaded
    tiles_to_download = []
    for tile in positive_jpg_tiles: #index over the downloaded tiled
        if tile not in tiles_downloaded: #index over the tiles that should be downloded
            tiles_to_download.append(tile)
    print("the number of tiles that need to be downloaded:", len(tiles_to_download))
    
    # Download Tiles  
    tile_names = []
    tile_urls = []
    file_names = []
    tile_names_without_year = []
    for tile in tiles_to_download:   
        ### download the tiles if they are not in the tiles folder 
        #check if the tile name is contained in the string of complete arrays
        tile_name = [string for string in tile_names_tile_urls_complete_array[:,0] if tile in string]          
        if len(tile_name) == 1: #A single tile name # get tile url from the first (only) entry
            tile_url = tile_names_tile_urls_complete_array[tile_names_tile_urls_complete_array[:,0]==tile_name[0]][0][1] 
            tile_names.append(tile_name[0])
            tile_urls.append(tile_url)
        elif len(np.unique(tile_name)) > 1: # Multiple (different tiles) possibly the same tiles in different states, possible different years
            tile_url = tile_names_tile_urls_complete_array[tile_names_tile_urls_complete_array[:,0]==tile_name[0]][0][1]# get tile url
            tile_names.append(tile_name[0])
            tile_urls.append(tile_url)
        elif (len(tile_name) > 1): #Multiple different tile names that are the same, probably different naip storage locations
            # get tile url from the second entry 
            tile_url = tile_names_tile_urls_complete_array[tile_names_tile_urls_complete_array[:,0]==tile_name[1]][1][1] 
            tile_names.append(tile_name[1])
            tile_urls.append(tile_url)
            
        #get file name
        file_name = tile_name[0]
        if tile_name[0].count("_") > 5:
            tile_name = tile_name[0].rsplit("_",1)[0]
            file_name = tile_name + ".tif"
        print(file_name)
        ### Download tile
        file_names.append(ap.download_url(tile_url, tiles_complete_dataset_path,
                                                     destination_filename = file_name,       
                                                             progress_updater=ap.DownloadProgressBar()))
    #get the tile_names without the year
    for file_name in file_names:
        tile_names_without_year.append(file_name.rsplit("_",1)[0])
        
    return(np.array((tile_names, tile_urls, file_names, tile_names_without_year)).T)


def downloaded_tifs_tile_names_tile_urls_file_names_tile_names_without_year(tile_path, tile_names_tile_urls_complete_array):
    #remove thumbs
    remove_thumbs(tile_path)
    tif_paths = glob(tile_path + "/**/*.tif", recursive = True)
    
    tile_names = []
    tile_urls = []
    file_names = [] 
    tile_names_without_year = []  
    
    for path in tif_paths:
        base = os.path.basename(path)
        tile_name = os.path.splitext(base)[0] #name of tif with the extension removed
        #check if the tile name is contained in the string of complete arrays
        tile_name = [string for string in tile_names_tile_urls_complete_array[:,0] if tile_name in string]      
        
        if len(tile_name) == 1: #A single tile name # get tile url from the first (only) entry
            tile_url = tile_names_tile_urls_complete_array[tile_names_tile_urls_complete_array[:,0]==tile_name[0]][0][1] 
            tile_names.append(tile_name[0])
            tile_urls.append(tile_url)
        elif len(np.unique(tile_name)) > 1: # Multiple (different tiles) possibly the same tiles in different states, possible different years
            tile_url = tile_names_tile_urls_complete_array[tile_names_tile_urls_complete_array[:,0]==tile_name[0]][0][1]# get tile url
            tile_names.append(tile_name[0])
            tile_urls.append(tile_url)
        elif (len(tile_name) > 1): #Multiple different tile names that are the same, probably different naip storage locations
            # get tile url from the second entry 
            tile_url = tile_names_tile_urls_complete_array[tile_names_tile_urls_complete_array[:,0]==tile_name[1]][1][1] 
            tile_names.append(tile_name[1])
            tile_urls.append(tile_url)

        #get file name
        file_name = tile_name[0]
        if tile_name[0].count("_") > 5:
            tile_name = tile_name[0].rsplit("_",1)[0]
            file_name = tile_name + ".tif"
        file_names.append(file_name)
        ### Download tile
        
    #get the tile_names without the year
    for file_name in file_names:
        tile_names_without_year.append(file_name.rsplit("_",1)[0])
    
    return(np.array((tile_names, tile_urls, file_names, tile_names_without_year)).T)

##

def image_characteristics(tiles_dir, verified_positive_jpgs):
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

    standard_tile_names = []
    chip_names = []
    NW_coordinates = []
    SE_coordinates = []
    row_indicies = []
    col_indicies = []
    full_path  = []
    root = []
    for tile_name in tqdm.tqdm(os.listdir(tiles_dir)): #index over the tiles in the tiles_dir 
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
                #Tile names no longer match chip file names!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                chip_name_temp = file_name+ '_'+ str(count).zfill(6) + '.jpg'
                count += 1  
                if chip_name_temp in verified_positive_jpgs[:,0]: #only record values for images that are annotated
                    #image characteristics
                    chip_names.append(chip_name_temp) # The index is a six-digit number like '000023'.
                    NW_coordinates.append([x*item_dim, y*(item_dim)]) #NW (Top Left) 
                    SE_coordinates.append([x*item_dim+item_dim-1, y*(item_dim)+item_dim-1]) #SE (Bottom right) 
                    row_indicies.append(y)
                    col_indicies.append(x)
                    #tile characteristics
                    ##  Get tile url using tile name
                    standard_tile_names.append(tile_name)
                    #path
                    full_path = verified_positive_jpgs[verified_positive_jpgs[:,0] == chip_name_temp][0][1]
                    root = full_path.split("\\",2)[0]

    #create pandas dataframe
    image_characteristics = pd.DataFrame(data={ "root":root,
                                                'standard_tile_name': standard_tile_names,
                                                'chip_name': chip_names,
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

def formatted_chip_names_to_standard_names(chip):
    """
    """
    chip = os.path.splitext(chip)[0]#remove ext
    chip_name, chip_number = chip.rsplit("_",1)
    tile_name = chip_name.split("_",4)[4]
    if tile_name.count("_") > 5:
        tile_name = tile_name.rsplit("_",1)[0]
    standard_chip_name = tile_name + "_"+ chip_number 
    return(standard_chip_name)

def rename_formatted_chips_images_xmls(complete_dataset_path):
    """
    Rename chips (jps/xmls)
    """
    positive_images_path = os.path.join(complete_dataset_path,"chips_positive")
    for chip in os.listdir(positive_images_path):
        #format tile_names to only include inital capture date 1/20
        if chip.count("_") > 6:
            #old path
            old_chip_path = os.path.join(positive_images_path, chip)
            
            #new name
            new_chip_name = formatted_chip_names_to_standard_names(chip)
            
            #copy images
            if not os.path.exists(os.path.join(complete_dataset_path,"standard_chips_positive", new_chip_name+".jpg")): #If the new tile path does not exist, convert the tile to standard format
                shutil.copyfile(old_chip_path, os.path.join(complete_dataset_path,"standard_chips_positive", new_chip_name+".jpg"))                
            elif not os.path.exists(os.path.join(complete_dataset_path,"dups_chips_positive", new_chip_name+".jpg")): #If the new tile path does not exist, convert the tile to standard format
                shutil.copyfile(old_chip_path, os.path.join(complete_dataset_path,"dups_chips_positive", new_chip_name+".jpg"))                
            else:
                print("so many dups")
                
            #copy annotations
            if not os.path.exists(os.path.join(complete_dataset_path,"standard_chips_positive_xml", new_chip_name+".xml")): #If the new tile path does not exist, convert the tile to standard format
                shutil.copyfile(old_chip_path, os.path.join(complete_dataset_path,"standard_chips_positive_xml", new_chip_name+".xml"))
            elif not os.path.exists(os.path.join(complete_dataset_path,"dups_chips_positive_xml", new_chip_name+".xml")): #If the new tile path does not exist, convert the tile to standard format
                shutil.copyfile(old_chip_path, os.path.join(complete_dataset_path,"dups_chips_positive_xml", new_chip_name+".xml"))                
            else:
                print("so many dups")
            
            #if os.path.exists(new_tile_path) and os.path.exists(old_tile_path): #If the new tile path already exists, delete the old tile path (if it still exists)
            #    os.remove(old_tile_path)
            
            
###
def identify_verified_jpgs_missing_annotations(verified_sets_parent_dir, verified_set_dir):
    """
    Args:
    verified_sets_parent_dir(str): Name of the parent folder holding verified images; Ex:"verified/verified_sets"
    verified_set_dir(str): Name of verified set folder containing images without corresponding annotations; Ex:"verify_jaewon_poonacha_cleave_1"
    
    Return: 
    jpgs_missing_xmls(list): list of jpgs without corresponding annotations in the verified folder of interest
    jpgs_missing_xmls_path(list): list of paths containing xmls matching the jpgs missing annotations
    """
    #get the xml ids w/o the ext
    xmls = os.listdir(os.path.join(parent_directory,verified_sets_parent_dir, verified_set_dir, "chips_positive_xml"))
    xmls_without_ext = []
    for xml in xmls:
        xmls_without_ext.append(os.path.splitext(xml)[0])
        
    #get the jpg ids w/o the ext
    jpgs = os.listdir(os.path.join(parent_directory, verified_sets_parent_dir, verified_set_dir,"chips_positive"))
    jpgs_without_ext = []
    for jpg in jpgs:
        jpgs_without_ext.append(os.path.splitext(jpg)[0])

    #identify jpgs tht are missing xmls
    jpgs_missing_xmls = []
    for xml in xmls_without_ext:
        if xml not in jpgs_without_ext:
            jpgs_missing_xmls.append(xml)

    #identify possible xml path 
    all_xmls = glob(parent_directory + "/**/*.xml", recursive = True)
    jpgs_missing_xmls_path =[]
    for jpg in jpgs_missing_xmls:
        jpg_path = [string for string in all_xmls if jpg in string]   
        if len(jpg_path) > 0:
            jpgs_missing_xmls_path.append(jpg_path)
    
    return(jpgs_missing_xmls, jpgs_missing_xmls_path)









## Identify Duplicates
        
def unique_by_first_dimension(a, images):
    #https://stackoverflow.com/questions/41071116/how-to-remove-duplicates-from-a-3d-array-in-python
    tmp = a.reshape(a.shape[0], -1)
    b = np.ascontiguousarray(tmp).view(np.dtype((np.void, tmp.dtype.itemsize * tmp.shape[1])))
    
    _, idx = np.unique(b, return_index=True)
    unique_images = images[idx]
    
    u, c = np.unique(b, return_counts=True)
    dup = u[c > 1]
    duplicate_images = images[np.where(np.isin(b,dup))[0]]
    return(unique_images, duplicate_images)

def intersection_of_sets(arr1, arr2, arr3):
    # Converting the arrays into sets
    s1 = set(arr1)
    s2 = set(arr2)
    s3 = set(arr3)
      
    # Calculates intersection of sets on s1 and s2
    set1 = s1.intersection(s2)         #[80, 20, 100]
      
    # Calculates intersection of sets on set1 and s3
    result_set = set1.intersection(s3)
      
    # Converts resulting set to list
    final_list = list(result_set)
    print(len(final_list))
    return(final_list)

def move_images(old_image_dir, new_image_dir, image_names):
    #Ensure directory exists
    os.makedirs(new_image_dir, exist_ok = True)
    #move images
    for image in image_names:
        shutil.copyfile(os.path.join(old_image_dir,image), 
                        os.path.join(new_image_dir,image))                

def sorted_list_of_files(dups_chips_positive_path):
    #https://thispointer.com/python-get-list-of-files-in-directory-sorted-by-size/#:~:text=order%20by%20size%3F-,Get%20list%20of%20files%20in%20directory%20sorted%20by%20size%20using,%2C%20using%20lambda%20x%3A%20os.
    #Get list of files in directory sorted by size using os.listdir()

    #list_of_files = filter( lambda x: os.path.isfile(os.path.join(dir_name, x)), os.listdir(dir_name) )
    # Sort list of file names by size 
    #list_of_files = sorted( list_of_files,key =  lambda x: os.stat(os.path.join(dir_name, x)).st_size)
    
    sizes = []
    dup_images = np.array(os.listdir(dups_chips_positive_path))
    for image in dup_images:
        sizes.append(os.stat(os.path.join(dups_chips_positive_path,image)).st_size)
    sizes = np.array(sizes)
    
    df = pd.DataFrame({'dups': dup_images,
                       'sizes': sizes})
    df = df.sort_values(by=['sizes'])
    df.to_csv('dup tile names.csv') 

    return(df)

def list_of_lists_positive_chips(chips_positive_path):
    positive_chips = os.listdir(chips_positive_path)
    positive_chips_lists = [positive_chips[x:x+1000] for x in range(0, len(positive_chips), 1000)]
    return(positive_chips_lists)

def directory_tile_names(directory, output_file_name): 
    tiles = []
    for image in os.listdir(directory):
            img = os.path.splitext(image)[0] #name of tif with the extension removed
            tile = img.rsplit("_",1)[0]
            #print(tile.split("_",4)[4])
            #tile = tile.split("_",4)[4] #get the tile names to remove duplicates from being downloaded
            tiles.append(tile)
    tiles = np.unique(tiles)
    pd.DataFrame(tiles, columns = [output_file_name]).to_csv(output_file_name+'.csv') 

def identify_all_paths_to_duplicate_images(parent_directory, duplicate_images):
    entire_jpg = glob(parent_directory + "/**/*.jpg", recursive = True)
    full_path = []
    jpg_name = []
    for jpg in entire_jpg:
        if jpg.rsplit("\\")[-1] in duplicate_images:
            full_path.append(jpg)
            jpg_name.append(jpg.rsplit("\\")[-1])

    df = pd.DataFrame({'jpg name': jpg_name,
                       'full path': full_path})
    df.to_csv("duplicate_jpgs_full_path.csv")


def get_tile_names_from_chip_names(directory):
    remove_thumbs(directory)
    tile_names = []
    for chip_name in os.listdir(directory):
        chip_name = os.path.splitext(chip_name)[0]#remove ext
        chip_name, _ = chip_name.rsplit("_",1)
        tile_names.append(chip_name.split("_",4)[4] + ".tif")
    return(np.unique(tile_names))

def positive_images_to_array(images_dir_path):
    images = np.array(os.listdir(os.path.join(images_dir_path)))
    image_array = np.zeros((len(images),512,512, 3), dtype='uint8')
    image_directory = np.array([images_dir_path] *len(images))
    for num in range(len(images)):    
        image = cv2.imread(os.path.join(images_dir_path, images[num])) #open image
        image_array[num,:,:,:] = image
        
    return(images, image_array, image_directory)

def positive_images_to_array_rgb(images_dir_path):
    images = np.array(os.listdir(os.path.join(images_dir_path)))
    imgsr = np.zeros((len(images),512,512), dtype='uint8')
    imgsg = np.zeros((len(images),512,512), dtype='uint8')
    imgsb = np.zeros((len(images),512,512), dtype='uint8')
    
    for num in range(len(images)):    
        image = cv2.imread(os.path.join(images_dir_path, images[num])) #open image
        imgsr[num,:,:] = image[:,:,0]
        imgsg[num,:,:] = image[:,:,1]
        imgsb[num,:,:] = image[:,:2]
    return(images, imgsr, imgsg, imgsb)

def tile_to_chip_array(tile, x, y, item_dim):
    """
    x: row index
    y: col index
    """
    chip_img = tile[y*item_dim:y*item_dim+item_dim, x*(item_dim):x*(item_dim)+item_dim]
    #add in back space if it is the edge of an image
    if chip_img.shape[0] != 512:  #Height
        black_height = 512  - chip_img.shape[0] #Height
        black_width = 512 #- chip_img.shape[1] #width
        black_img = np.zeros((black_height,black_width,3), np.uint8)
        chip_img = np.concatenate([chip_img, black_img])

    if chip_img.shape[1] != 512: #width
        black_height = 512 #- chip_img.shape[0] #Height
        black_width = 512 - chip_img.shape[1] #width
        black_img = np.zeros((black_height,black_width,3), np.uint8)
        chip_img = np.concatenate([chip_img, black_img],1)
    
    return(chip_img)

def identify_incorrect_images(tile_dir, tiles_in_directory, 
                          images_in_directory, images_in_directory_array,
                             image_directories):
    """
    Find images that do not align with the tile chip
    Confirm that the standard tile name in the chip and the contents of the chip match
    """
    #index over the tiles with corresponding images in the given directory
    tile_names = []
    incorrect_chip_names = []
    incorrect_chip_paths = []
    ys = []
    xs = []
    #correct_0_incorrect_1_images = []

    #same_image_counter = 0
    for tile_name in tiles_in_directory: 
        file_name, ext = os.path.splitext(tile_name) # File name
        
        #get tile shape
        item_dim = int(512)          
        tile = cv2.imread(os.path.join(tile_dir, tile_name)) 
        tile_height,  tile_width,  tile_channels = tile.shape #the size of the tile #determine tile dimensions
        row_index = math.ceil(tile_height/512) #divide the tile into 512 by 512 chips (rounding up)
        col_index = math.ceil(tile_width/512)

        count = 1  
        for y in range(0, col_index):
            for x in range(0, row_index):
                chip_name_temp = file_name+ '_' + str(count).zfill(6) + '.jpg'
                #create a numpy array of each correctly chipped images 
                correct_image = tile_to_chip_array(tile, x, y, item_dim)
                count += 1  

                #Identify if images that are contained in the directory of interest
                confirmed_chips = [string for string in images_in_directory if chip_name_temp in string]
                if len(confirmed_chips) > 0:
                    for confirmed_chip in confirmed_chips: #there may be duplicate images corresponding to the same standard tile name (nj and ny overlap)
                    #obtain a numpy array of the image in the directory of interest
                        index, = np.where(images_in_directory == confirmed_chip)
                        image_in_directory_array = images_in_directory_array[index[0]] #use the actual value of index (saved as an array)
                        image_directory = image_directories[index[0]]
                        ##https://pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/
                        #https://pyimagesearch.com/2014/09/15/python-compare-two-images/
                        gray_image_in_directory_array = cv2.cvtColor(image_in_directory_array, cv2.COLOR_BGR2GRAY)
                        gray_correct_image = cv2.cvtColor(correct_image, cv2.COLOR_BGR2GRAY)
                        (score, diff) = compare_ssim(gray_image_in_directory_array, gray_correct_image, full=True)
                        diff = (diff * 255).astype("uint8")
                        #if score >= 0.90:
                        #    correct_0_incorrect_1_images.append(0)

                        if score < 0.90: 
                            #print("different image")
                            tile_names.append(tile_name)
                            xs.append(x)
                            ys.append(y)
                            incorrect_chip_names.append(confirmed_chip)
                            incorrect_chip_paths.append(os.path.join(image_directory,confirmed_chip))
                            #print("SSIM: {}".format(score))
                            #correct_0_incorrect_1_images.append(1)

    return(tile_names, xs, ys, incorrect_chip_names, incorrect_chip_paths)

def identify_incorrect_images_simultaneous(tile_dir, tiles_in_directory, images_path):
    """
    Find images that do not align with the tile chip
    Confirm that the standard tile name in the chip and the contents of the chip match
    """
    #index over the tiles with corresponding images in the given directory
    tile_names = []
    incorrect_chip_names = []
    incorrect_chip_paths = []
    ys = []
    xs = []
    #correct_0_incorrect_1_images = []

    #same_image_counter = 0
    for tile_name in tqdm.tqdm(tiles_in_directory): 
        file_name, ext = os.path.splitext(tile_name) # File name
        
        #get tile shape
        item_dim = int(512)          
        tile = cv2.imread(os.path.join(tile_dir, tile_name)) 
        tile_height,  tile_width,  tile_channels = tile.shape #the size of the tile #determine tile dimensions
        row_index = math.ceil(tile_height/512) #divide the tile into 512 by 512 chips (rounding up)
        col_index = math.ceil(tile_width/512)

        count = 1  
        for y in range(0, col_index):
            for x in range(0, row_index):
                chip_name_temp = file_name+ '_' + str(count).zfill(6) + '.jpg'
                #create a numpy array of each correctly chipped images 
                correct_image = tile_to_chip_array(tile, x, y, item_dim)
                count += 1  

                #Identify if images that are contained in the directory of interest
                labeled_chip_paths = [string for string in images_path if chip_name_temp in string]
                if len(labeled_chip_paths) > 0:
                    for labeled_chip_path in labeled_chip_paths: #there may be duplicate images corresponding to the same standard tile name (nj and ny overlap)
                    #obtain a numpy array of the image in the directory of interest
                        index, = np.where(images_path == labeled_chip_path)
                        labeled_chip_array = cv2.imread(os.path.join(images_path[index[0]])) #open image

                        ##https://pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/
                        #https://pyimagesearch.com/2014/09/15/python-compare-two-images/
                        gray_labeled_chip_array = cv2.cvtColor(labeled_chip_array, cv2.COLOR_BGR2GRAY)
                        gray_correct_image = cv2.cvtColor(correct_image, cv2.COLOR_BGR2GRAY)
                        (score, diff) = compare_ssim(gray_labeled_chip_array, gray_correct_image, full=True)
                        diff = (diff * 255).astype("uint8")
                        #if score >= 0.90:
                        #    correct_0_incorrect_1_images.append(0)

                        if score < 0.90: 
                            #print("different image")
                            tile_names.append(tile_name)
                            xs.append(x)
                            ys.append(y)
                            incorrect_chip_paths.append(labeled_chip_path)
                            #print("SSIM: {}".format(score))
                            #correct_0_incorrect_1_images.append(1)

    return(tile_names, xs, ys, incorrect_chip_paths)

    
    
#Long way round
def list_of_lists_positive_chips(chips_positive_path, blocks):
    positive_chips = os.listdir(chips_positive_path)
    positive_chips_lists = [positive_chips[x:x+int(blocks)] for x in range(0, len(positive_chips), int(blocks))]
    return(positive_chips_lists)

def identify_identical_images(images_dir_path, blocks, block):#o_images = None,):
    """
    Args:
    images_dir_path(str): path to directory containing images of interest
    Returns
    same_images(list of lists): lists of images that contain that same information
    https://pysource.com/2018/07/19/check-if-two-images-are-equal-with-opencv-and-python/
    """                         
    same_images_o_images = [] #Make a list to hold the identical images
    same_images_d_images = [] #Make a list to hold the identical images
         
    #Make a list of the images to check for duplicates (images in directory or provided as arugment in function)
    d_images = os.listdir(os.path.join(images_dir_path))

    o_images = list_of_lists_positive_chips(images_dir_path, int(blocks))[int(block)]

    for o in tqdm.tqdm(range(len(o_images))):
        o_image = o_images[o]
        original = cv2.imread(os.path.join(images_dir_path, o_image)) #open image

        for d_image in d_images:
            duplicate = cv2.imread(os.path.join(images_dir_path, d_image)) #open image

            #check for similar characteristics
            if original.shape == duplicate.shape:
                difference = cv2.subtract(original, duplicate)
                b, g, r = cv2.split(difference)

            if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
                if o_image != d_image:
                    same_images_o_images.append([o_image]) #Make a list to hold the identical images
                    same_images_d_images.append([d_image]) #Make a list to hold the identical images
                    if d_image in o_images:
                        o_images.remove(d_image) #remove duplicate images, because you have already at least one version to use to find others
        
        d_images.remove(o_image) #remove o_image from d_images list, because you have already checked it against each image
    
    same_images = np.array(same_images_o_images, same_images_d_images)
    return(same_images)
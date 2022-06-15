"""
Functions to process, format, and conduct calculations on the annotated or verified dataset
"""
# Standard packages
#from __future__ import print_function
import warnings
import urllib
import shutil
import os
import math 
import json
import tqdm
from glob import glob
import numpy as np
import pandas as pd

import fiona #must be import before geopandas
import geopandas as gpd
import rasterio
import rioxarray
import re
import rtree
import pyproj
import shapely
from shapely.ops import transform
from shapely.geometry import Polygon, Point, MultiPoint, MultiPolygon, MultiLineString

import cv2
import matplotlib 
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as compare_ssim
#import imutils
#import psutil

import data_eng.az_proc as ap

#Parsing/Modifying XML
from lxml.etree import Element,SubElement,tostring
import xml.dom.minidom
from xml.dom.minidom import parseString
import xml.etree.ElementTree as et
from xml.dom import minidom


####### add chips to rechip folder ############################################################
def add_chips_to_chip_folders(rechipped_image_path, tile_name):
    """ 
    Args:
    remaining_chips_path(str): path to folder that will contain all of the remaining images that have not been labeled and correspond to tiles that have labeled images
    tile_name(str): name of tile without of extension
    Returns:
    """
    chips_path = os.path.join(rechipped_image_path, tile_name, "chips")
    os.makedirs(chips_path, exist_ok=True)
    
    item_dim = int(512)
    tile = cv2.imread(os.path.join(tiles_complete_dataset_path, tile_name + ".tif")) 
    tile_height,  tile_width,  tile_channels = tile.shape #the size of the tile 
    row_index = math.ceil(tile_height/512) 
    col_index = math.ceil(tile_width/512)
    #print(row_index, col_index)

    count = 1            
    for y in range(0, row_index): #rows
        for x in range(0, col_index): #cols
            chip_img = fc.tile_to_chip_array(tile, x, y, item_dim)
            #specify the chip names
            chip_name_correct_chip_name = tile_name + '_' + f"{y:02}"  + '_' + f"{x:02}" + '.jpg' # The index is a six-digit number like '000023'.
            if not os.path.exists(os.path.join(chips_path, chip_name_correct_chip_name)):
                cv2.imwrite(os.path.join(chips_path, chip_name_correct_chip_name), chip_img) #save images  

############################################################################################################
###########################################   Remove Thumbs    #############################################
############################################################################################################
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

########### Extract information from tile_names_tile urls numpy arrays ##########
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

################## Get tiles names or jpg names from jpg paths ####################
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
##############################################################################################################################
###################################                  Chip Tiles              #################################################
##############################################################################################################################
def tile_to_chip_array(tile, x, y, item_dim):
    """
    ##
    x: col index
    y: row index
    """
    chip_img = tile[y*item_dim:y*item_dim+item_dim, x*(item_dim):x*(item_dim)+item_dim]
    #add in back space if it is the edge of an image
    if (chip_img.shape[0] != 512) & (chip_img.shape[1] != 512): #width
        #print("Incorrect Width")
        chip = np.zeros((512,512,3))
        chip[0:chip_img.shape[0], 0:chip_img.shape[1]] = chip_img
        chip_img = chip
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


############## Download Tiles ##########################################################################################
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

###################################################################################################################
#################################   Obtain Location Data (UTM to WGS84)  ##########################################
###################################################################################################################   
def tile_dimensions_and_utm_coords(tile_path):
    """ Obtain tile band, height and width and utm coordinates
    Args: tile_path(str): the path of the tile 
    Returns: 
    utmx(np array): the x utm coordinates
    utmy(np array): the y utm coordinates
    tile_band(int): the number of bands
    tile_height(int): the height of the tile (in pixels)
    tile_width(int): the width of the tile (in pixels)
    """
    ## Get tile locations
    da = rioxarray.open_rasterio(tile_path) ## Read the data
    # Compute the lon/lat coordinates with rasterio.warp.transform
    # lons, lats = np.meshgrid(da['x'], da['y'])
    tile_band, tile_height, tile_width = da.shape[0], da.shape[1], da.shape[2]
    utmx = np.array(da['x'])
    utmy = np.array(da['y'])
    return(utmx, utmy, tile_band, tile_height, tile_width)
def get_utm_proj(tile_path):
    """ Obtain utm projection as a string 
    Args: tile_path(str): the path of the tile 
    Returns: 
    utm_proj(str): the UTM string as the in term of EPSG code
    """
    da = rasterio.open(tile_path)
    utm_proj = da.crs.to_string()
    return(utm_proj)
def transform_point_utm_to_wgs84(utm_proj, utm_xcoord, utm_ycoord):
    """ Convert a utm pair into a lat lon pair 
    Args: 
    utm_proj(str): the UTM string as the in term of EPSG code
    utmx(int): the x utm coordinate of a point
    utmy(int): the y utm coordinates of a point
    Returns: 
    (wgs84_pt.x, wgs84_pt.y): the 'EPSG:4326' x and y coordinates 
    """
    #https://gis.stackexchange.com/questions/127427/transforming-shapely-polygon-and-multipolygon-objects
    #get utm projection
    utm = pyproj.CRS(utm_proj)
    #get wgs84 proj
    wgs84 = pyproj.CRS('EPSG:4326')
    #specify utm point
    utm_pt = Point(utm_xcoord, utm_ycoord)
    #transform utm into wgs84 point
    project = pyproj.Transformer.from_crs(utm, wgs84, always_xy=True).transform
    wgs84_pt = transform(project, utm_pt)
    return(wgs84_pt.x, wgs84_pt.y)

###################################################################################################################
##########################   Create dataframe of Image and Tile Characteristics  ##################################
###################################################################################################################   
def image_tile_characteristics(images_and_xmls_by_tile_path, tiles_dir, tile_name_tile_url):#, verified_positive_jpgs):
    tile_names_by_tile = []
    tile_paths_by_tile = []
    tile_urls_by_tile = []

    tile_heights = []
    tile_widths = []
    tile_depths = []
    min_utmx_tile = [] #NW_coordinates
    min_utmy_tile = []  #NW_coordinates
    max_utmx_tile = [] #SE_coordinates
    max_utmy_tile = [] #SE_coordinates
    utm_projection = [] 

    min_lon_tile = [] #NW_coordinates
    min_lat_tile = []  #NW_coordinates
    max_lon_tile = [] #SE_coordinates
    max_lat_tile = [] #SE_coordinates

    chip_names = []
    tile_names_by_chip = []
    tile_paths_by_chip = []
    tile_urls_by_chip = []

    minx_pixel = []
    miny_pixel = []
    maxx_pixel = []
    maxy_pixel = []
    min_lon_chip = [] #NW_coordinates
    min_lat_chip = [] #NW_coordinates
    max_lon_chip = [] #SE_coordinates
    max_lat_chip = [] #SE_coordinates
    row_indicies = []
    col_indicies = []
    image_paths  = []
    xml_paths = []
    item_dim = int(512)
    folders_of_images_xmls_by_tile = os.listdir(images_and_xmls_by_tile_path)
    for tile_name in tqdm.tqdm(folders_of_images_xmls_by_tile):
        #specify image/xml paths for each tile
        positive_image_dir = os.path.join(images_and_xmls_by_tile_path, tile_name, "chips_positive")
        remove_thumbs(positive_image_dir)
        positive_xml_dir = os.path.join(images_and_xmls_by_tile_path, tile_name, "chips_positive_xml")
        #load a list of images/xmls for each tile
        positive_images = os.listdir(positive_image_dir)
        positive_xmls = os.listdir(positive_xml_dir)
        #read in tile
        tile_path = os.path.join(tiles_dir, tile_name + ".tif")
        #obtain tile url
        tile_url = [string for string in tile_name_tile_url[:,1] if tile_name in string][0]
        #tile name/path/urls by tile
        tile_names_by_tile.append(tile_name)
        tile_paths_by_tile.append(tile_path)
        tile_urls_by_tile.append(tile_url)
        #determine the utm coords for each tile 
        utmx, utmy, tile_band, tile_height, tile_width = tile_dimensions_and_utm_coords(tile_path)
        tile_heights.append(tile_height)
        tile_widths.append(tile_width)
        tile_depths.append(tile_band)
        min_utmx_tile.append(utmx[0]) #NW_coordinates
        min_utmy_tile.append(utmy[0])  #NW_coordinates
        max_utmx_tile.append(utmx[-1]) #SE_coordinates
        max_utmy_tile.append(utmy[-1]) #SE_coordinates
        utm_proj = get_utm_proj(tile_path)
        utm_projection.append(utm_proj)
        minlon, minlat = transform_point_utm_to_wgs84(utm_proj, utmx[0], utmy[0])
        maxlon, maxlat = transform_point_utm_to_wgs84(utm_proj, utmx[-1], utmy[-1]) 
        min_lon_tile.append(minlon) #NW_coordinates
        min_lat_tile.append(minlat) #NW_coordinates
        max_lon_tile.append(maxlon) #SE_coordinates
        max_lat_tile.append(maxlat) #SE_coordinates
        

        for positive_image in positive_images: #iterate over each image affiliated with a given tile
            #tile and chip names
            chip_name = os.path.splitext(positive_image)[0]
            chip_names.append(chip_name) # The index is a six-digit number like '000023'.
            tile_names_by_chip.append(tile_name)
            #path
            tile_paths_by_chip.append(tile_path)
            tile_urls_by_chip.append(tile_url)

            image_paths.append(os.path.join(positive_image_dir, positive_image))
            xml_paths.append(os.path.join(positive_xml_dir, chip_name +".xml"))
            #row/col indicies 
            y, x = chip_name.split("_")[-2:] #name of tif with the extension removed; y=row;x=col
            y = int(y)
            x = int(x)
            row_indicies.append(y)
            col_indicies.append(x)
            #get the pixel coordinates (indexing starting at 1)
            minx = x*item_dim + 1
            miny = y*item_dim + 1
            maxx = (x+1)*item_dim 
            maxy = (y+1)*item_dim
            if maxx > tile_width:
                maxx = tile_width
            if maxy > tile_height:
                maxy = tile_height
            minx_pixel.append(minx) #NW (max: Top Left) # used for numpy crop
            miny_pixel.append(miny) #NW (max: Top Left) # used for numpy crop
            maxx_pixel.append(maxx) #SE (min: Bottom right) 
            maxy_pixel.append(maxy) #SE (min: Bottom right) 
            #determine the lat/lon
            min_lon, min_lat = transform_point_utm_to_wgs84(utm_proj, utmx[minx-1], utmy[miny-1]) #subtract 1 to get from PASCAL VOC pixel coordinates to python index
            max_lon, max_lat = transform_point_utm_to_wgs84(utm_proj, utmx[maxx-1], utmy[maxy-1]) 
            min_lon_chip.append(min_lon) #NW (max: Top Left) # used for numpy crop
            min_lat_chip.append(min_lat) #NW (max: Top Left) # used for numpy crop
            max_lon_chip.append(max_lon) #SE (min: Bottom right) 
            max_lat_chip.append(max_lat) #SE (min: Bottom right)
    tile_characteristics = pd.DataFrame(data={'tile_name': tile_names_by_tile, 'tile_url': tile_urls_by_tile, 'tile_path': tile_paths_by_tile, 
                        'tile_heights': tile_heights, 'tile_widths': tile_widths, 'tile_bands': tile_depths, 'min_utmx_tile': min_utmx_tile, 'min_utmy_tile': min_utmy_tile, 
                        'max_utmx_tile': max_utmx_tile, 'max_utmy_tile': max_utmy_tile, 'utm_projection': utm_projection,
                        'min_lon_tile': min_lon_tile,'min_lat_tile': min_lat_tile,'max_lon_tile': max_lon_tile,'max_lat_tile': max_lat_tile})

    image_characteristics = pd.DataFrame(data={'chip_name': chip_names, 'image_path': image_paths, 'xml_path': xml_paths,'tile_name': tile_names_by_chip, 
                            'tile_url': tile_urls_by_chip, 'tile_path': tile_paths_by_chip, 'row_indicies': row_indicies, 'col_indicies': col_indicies,
                            'minx_pixel': minx_pixel, 'miny_pixel': miny_pixel, 'maxx_pixel': maxx_pixel,'maxy_pixel': maxy_pixel,
                            'min_lon_chip': min_lon_chip,'min_lat_chip': min_lat_chip,'max_lon_chip': max_lon_chip, 'max_lat_chip': max_lat_chip})
    tile_characteristics.to_csv("tile_characteristics.csv")
    image_characteristics.to_csv("image_characteristics.csv")
    return(tile_characteristics, image_characteristics)
###################################################################################################################
######################################   Create Tile level  XMLs  #################################################
###################################################################################################################                
def create_tile_xml(tile_name, xml_directory, tile_resolution, tile_year, 
                tile_width, tile_height, tile_band):
    tile_name_ext = tile_name + ".tif"
    root = et.Element("annotation")
    folder = et.Element("folder") #add folder to xml
    folder.text = "tiles" #folder
    root.insert(0, folder)
    filename = et.Element("filename") #add filename to xml
    filename.text = tile_name_ext #filename
    root.insert(1, filename)
    path = et.Element("path") #add path to xml
    path.text = os.path.join(xml_directory, tile_name_ext) #path
    root.insert(2, path)
    resolution = et.Element("resolution") #add resolution to xml
    resolution.text = tile_resolution #resolution
    root.insert(3, resolution)
    year = et.Element("year") #add year to xml
    year.text = tile_year #year
    root.insert(4,year)
    source = et.Element("source") #add database to xml
    database = et.Element("database")
    database.text = "Tile Level Annotation" #
    source.insert(0, database)
    root.insert(5,source)
    size = et.Element("size") #add size to xml
    width = et.Element("width")
    width.text = str(tile_width) #width
    size.insert(0, width)
    height = et.Element("height")
    height.text = str(tile_height) #height
    size.insert(1, height)
    depth = et.Element("depth")
    depth.text = str(tile_band) #depth
    size.insert(2, depth)
    root.insert(6,size)
    tree = et.ElementTree(root)
    et.indent(tree, space="\t", level=0)
    #tree.write("filename.xml")
    tree.write(os.path.join(xml_directory, tile_name +".xml"))     
    
def add_objects(xml_directory, tile_name, obj_class, 
                obj_truncated, obj_difficult, obj_chip_name,
                obj_xmin, obj_ymin, obj_xmax, obj_ymax):
    tree = et.parse(os.path.join(xml_directory, tile_name + ".xml"))
    root = tree.getroot() 
    obj = et.Element("object") #add size to xml
    
    name = et.Element("name") #class
    name.text = str(obj_class) 
    obj.insert(0, name)
    
    pose = et.Element("pose") #pose
    pose.text = "Unspecified" 
    obj.insert(1, pose)
    
    truncated = et.Element("truncated")
    truncated.text = str(obj_truncated) #
    obj.insert(2, truncated)

    difficult = et.Element("difficult")
    difficult.text = str(obj_difficult)
    obj.insert(3, difficult)
 
    chip_name = et.Element("chip_name")
    chip_name.text = str(obj_chip_name)
    obj.insert(4, chip_name)

    bndbox = et.Element("bndbox") #bounding box
    xmin = et.Element("xmin") #xmin
    xmin.text = str(obj_xmin) 
    bndbox.insert(0, xmin)
    ymin = et.Element("ymin") #ymin
    ymin.text = str(obj_ymin) 
    bndbox.insert(1, ymin)
    xmax = et.Element("xmax") #xmax
    xmax.text = str(obj_xmax) 
    bndbox.insert(2, xmax)
    ymax = et.Element("ymax") #ymax
    ymax.text = str(obj_ymax) 
    bndbox.insert(3, ymax)
    obj.insert(5, bndbox)
    
    root.append(obj)
    tree = et.ElementTree(root)
    et.indent(tree, space="\t", level=0)
    tree.write(os.path.join(xml_directory, tile_name +".xml"))   
    
def generate_tile_xmls(images_and_xmls_by_tile_path, tiles_dir, tiles_xml_path, item_dim):
    folders_of_images_xmls_by_tile = os.listdir(images_and_xmls_by_tile_path)
    for tile_name in tqdm.tqdm(folders_of_images_xmls_by_tile):
        tile_name_ext = tile_name + ".tif"
        #get tile dimensions ##replace with information from tile characteristics
        da = rioxarray.open_rasterio(os.path.join(tiles_dir, tile_name_ext))
        tile_band, tile_height, tile_width = da.shape[0], da.shape[1], da.shape[2]
        #specify image/xml paths for each tile
        positive_image_dir = os.path.join(images_and_xmls_by_tile_path, tile_name, "chips_positive")
        positive_xml_dir = os.path.join(images_and_xmls_by_tile_path, tile_name, "chips_positive_xml")
        #load a list of images/xmls for each tile
        positive_images = os.listdir(positive_image_dir)
        positive_xmls = os.listdir(positive_xml_dir)
                       
        for index, chip_xml in enumerate(positive_xmls):
            #identify rows and columns
            chip_name = os.path.splitext(chip_xml)[0]
            y, x = chip_name.split("_")[-2:] #name of tif with the extension removed; y=row;x=col
            y = int(y)
            x = int(x)
            minx = x*item_dim
            miny = y*item_dim
            #load each xml
            tree = et.parse(os.path.join(positive_xml_dir, chip_xml))
            root = tree.getroot()
            #create the tile xml
            if index == 0:
                resolution = root.find('resolution').text
                year = root.find('year').text
                create_tile_xml(tile_name, tiles_xml_path, resolution, year, 
                                tile_width, tile_height, tile_band)
            #add the bounding boxes
            for obj in root.iter('object'):
                xmlbox = obj.find('bndbox')
                obj_xmin = str(int(xmlbox.find('xmin').text) + minx)
                obj_xmax = str(int(xmlbox.find('xmax').text) + minx)
                obj_ymin = str(int(xmlbox.find('ymin').text) + miny)
                obj_ymax = str(int(xmlbox.find('ymax').text) + miny)
                add_objects(tiles_xml_path, tile_name, obj.find('name').text, obj.find('truncated').text, 
                            obj.find('difficult').text, chip_name, obj_xmin, obj_ymin, obj_xmax, obj_ymax)
#################################################################################################################
####################################     Correct object names in xmls      ######################################
#################################################################################################################
def reclassify_narrow_closed_roof_and_closed_roof_tanks(xml_path):
    """ Reclassify Narrow Closed Roof and Closed Roof Tanks
    """
    #load each xml
    class_ob = []
    tree = et.parse(xml_path)
    root = tree.getroot()
    for obj in root.iter('object'):
        name = obj.find("name").text 
        xmlbox = obj.find('bndbox') #get the bboxes
        obj_xmin = xmlbox.find('xmin').text
        obj_ymin = xmlbox.find('ymin').text
        obj_xmax = xmlbox.find('xmax').text
        obj_ymax = xmlbox.find('ymax').text
        width = int(obj_xmax) - int(obj_xmin)
        height = int(obj_ymax) - int(obj_ymin)
        if (int(obj.find('difficult').text) == 0) and (int(obj.find('truncated').text) == 0): 
            #if a closed roof tank is less than or equal to the narrow closed roof tank threshold than reclassify as  narrow closed roof tank
            if (name == "closed_roof_tank") and (width <= 15) and (height <= 15): 
                name = "narrow_closed_roof_tank"
            #if a narrow closed roof tank is greater than the closed roof tank threshold than reclassify as closed roof tank
            if (name == "narrow_closed_roof_tank") and (width > 15) and (height > 15):
                name = "closed_roof_tank"
    
    tree.write(os.path.join(xml_path))
    
def correct_inconsistent_labels_xml(xml_dir):
    #Create a list of the possible names that each category may take 
    correctly_formatted_object = ["closed_roof_tank","narrow_closed_roof_tank",
                                  "external_floating_roof_tank","sedimentation_tank",
                                  "water_tower","undefined_object","spherical_tank"] 
    object_dict = {"closed_roof_tank": "closed_roof_tank",
                   "closed_roof_tank ": "closed_roof_tank",
                   "closed roof tank": "closed_roof_tank",
                   "narrow_closed_roof_tank": "narrow_closed_roof_tank",
                   "external_floating_roof_tank": "external_floating_roof_tank",
                   "external floating roof tank": "external_floating_roof_tank",
                   'external_floating_roof_tank ': "external_floating_roof_tank",
                   "water_treatment_tank": "sedimentation_tank",
                   'water_treatment_tank ': "sedimentation_tank",
                   "water_treatment_plant": "sedimentation_tank",
                   "water_treatment_facility": "sedimentation_tank",
                   "water_tower": "water_tower",
                   "water_tower ": "water_tower",
                   'water_towe': "water_tower",
                   "spherical_tank":"spherical_tank",
                   'sphere':"spherical_tank",
                   'spherical tank':"spherical_tank",
                   "undefined_object": "undefined_object",
                   "silo": "undefined_object" }

    #"enumerate each image" This chunk is actually just getting the paths for the images and annotations
    for xml_file in os.listdir(xml_dir):
        # use the parse() function to load and parse an XML file
        tree = et.parse(os.path.join(xml_dir, xml_file))
        root = tree.getroot()         
        
        for obj in root.iter('object'):
            for name in obj.findall('name'):
                if name.text not in correctly_formatted_object:
                    name.text = object_dict[name.text]

            if int(obj.find('difficult').text) == 1:
                obj.find('truncated').text = '1'
                obj.find('difficult').text = '1'
            if int(obj.find('truncated').text) == 1:
                obj.find('truncated').text = '1'
                obj.find('difficult').text = '1'

        tree.write(os.path.join(xml_dir, xml_file))       

###################################################################################################################
####################################     Merge tile level annotations   ###########################################
###################################################################################################################               
#Generate two text boxes a larger one that covers them
def merge_boxes(bbox1, bbox2):
    """ Generate a bounding box that covers two bounding boxes
    Arg:
    bbox1(list): a list of the (ymin, xmin, ymax, xmax) coordinates for box 1 
    bbox2(list): a list of the (ymin, xmin, ymax, xmax) coordinates for box 2
    Returns:
    merged_bbox(list): a list of the (ymin, xmin, ymax, xmax) coordinates for the merged bbox

    """
    return [min(bbox1[0], bbox2[0]), 
         min(bbox1[1], bbox2[1]), 
         max(bbox1[2], bbox2[2]),
         max(bbox1[3], bbox2[3])]

#Computer a Matrix similarity of distances of the text and object
def calc_sim(bbox1, bbox2, dist_limit):
    """Determine the similarity of distances between bboxes to determine whether bboxes should be merged 
    Arg:
    bbox1(list): a list of the (ymin, xmin, ymax, xmax) coordinates for box 1 
    bbox2(list): a list of the (ymin, xmin, ymax, xmax) coordinates for box 2
    dist_list(int): the maximum threshold (pixel distance) to merge bounding boxes
    Returns:
    (bool): to indicate whether the bboxes should be merged 
    """

    # text: ymin, xmin, ymax, xmax
    # obj: ymin, xmin, ymax, xmax
    bbox1_xmin, bbox1_ymin, bbox1_xmax, bbox1_ymax = bbox1
    bbox2_xmin, bbox2_ymin, bbox2_xmax, bbox2_ymax = bbox2
    x_dist = min(abs(bbox2_xmin-bbox1_xmax), abs(bbox2_xmax-bbox1_xmin))
    y_dist = min(abs(bbox2_ymin-bbox1_ymax), abs(bbox2_ymax-bbox1_ymin))
        
    #define distance if one object is inside the other
    if (bbox2_xmin <= bbox1_xmin) and (bbox2_ymin <= bbox1_ymin) and (bbox2_xmax >= bbox1_xmax) and (bbox2_ymax >= bbox1_ymax):
        return(True)
    elif (bbox1_xmin <= bbox2_xmin) and (bbox1_ymin <= bbox2_ymin) and (bbox1_xmax >= bbox2_xmax) and (bbox1_ymax >= bbox2_ymax):
        return(True)
    #determine if the bboxes are suffucuently close to each other 
    elif (x_dist <= dist_limit) and (abs(bbox2_ymin-bbox1_ymin) <= dist_limit*3) and (abs(bbox2_ymax-bbox1_ymax) <= dist_limit*3):
        return(True)
    elif (y_dist <= dist_limit) and (abs(bbox2_xmin-bbox1_xmin) <= dist_limit*3) and (abs(bbox2_xmax-bbox1_xmax) <= dist_limit*3):
        return(True)
    else: 
        return(False)

def merge_algo(characteristics, bboxes, dist_limit):
    merge_bools = [False] * len(characteristics)
    for i, (char1, bbox1) in enumerate(zip(characteristics, bboxes)):
        for j, (char2, bbox2) in enumerate(zip(characteristics, bboxes)):
            if j <= i:
                continue
            # Create a new box if a distances is less than disctance limit defined 
            merge_bool = calc_sim(bbox1, bbox2, dist_limit) 
            if merge_bool == True:
            # Create a new box  
                new_box = merge_boxes(bbox1, bbox2)   
                bboxes[i] = new_box
                #delete previous text boxes
                del bboxes[j]
                
                # Create a new text string
                ##chip_name list
                if char1[0] != char2[0]: #if the chip_names are not the same
                    #make chip_names into an array
                    if type(char1[0]) == str: 
                        chip_names_1 = np.array([char1[0]])
                    if type(char2[0]) == str:
                        chip_names_2 = np.array([char2[0]])
                    chip_names = np.concatenate((chip_names_1, chip_names_2),axis=0)
                    chip_names = np.unique(chip_names).tolist()
                else:
                    chip_names = np.unique(char1[0]).tolist()  #if the chip_names are not the same
                
                #get object type 
                if char1[1] != char2[1]:
                    object_type = 'undefined_object'
                object_type = char1[1]
                
                characteristics[i] = [chip_names, object_type, 'Unspecified', '1', '1']
                #delete previous text 
                del characteristics[j]
                
                #return a new boxes and new text string that are close
                merge_bools[i] = True
    return merge_bools, characteristics, bboxes

def calculate_diameter(bbox, resolution = 0.6):
    obj_xmin, obj_ymin, obj_xmax, obj_ymax = bbox
    obj_width = obj_xmax - obj_xmin
    obj_height = obj_ymax - obj_ymin
    diameter = min(obj_width, obj_height) * resolution #meter
    return(diameter)

def merge_tile_annotations(tile_characteristics, tiles_xml_dir, tiles_xml_list = None, 
                           distance_limit = 5):
    # https://stackoverflow.com/questions/55593506/merge-the-bounding-boxes-near-by-into-one
    #specify tiles_xml_list
    if tiles_xml_list is None: #if tiles_xml_list not provided, specify the tiles xml list
        tiles_xml_list = os.listdir(tiles_xml_dir)
    #lists for geosons/geodatabase
    tile_names = []
    chip_names = []
    object_class = []
    merged_bbox = []
    geometry = []  
    minx_polygon_pixels = []
    miny_polygon_pixels = []
    maxx_polygon_pixels = []
    maxy_polygon_pixels = []
    polygon_vertices_pixels = []
    nw_corner_polygon_lat = []
    nw_corner_polygon_lon = []
    se_corner_polygon_lat = []
    se_corner_polygon_lon = []
    polygon_vertices_lon_lat = []
    
    diameter = []
    for tile_xml in tqdm.tqdm(tiles_xml_list): #iterate over tiles
        #save bboxes and characteristics
        trunc_diff_objs_bboxes = []
        trunc_diff_objs_characteristics = []
        remaining_objs_bboxes = []
        remaining_objs_characteristics = []
        #get tilename/tile xml path
        tile_name = os.path.splitext(tile_xml)[0]
        tile_xml_path = os.path.join(tiles_xml_dir, tile_xml)
        #load tile characteristics 
        tile_characteristics_subset = tile_characteristics[tile_characteristics.loc[:,"tile_name"] == tile_name]
        tile_width = tile_characteristics_subset["tile_widths"].values[0]
        tile_height = tile_characteristics_subset["tile_heights"].values[0]
        tile_utmx_array = np.linspace(tile_characteristics_subset["min_utmx_tile"].values[0], 
                                      tile_characteristics_subset["max_utmx_tile"].values[0],
                                      tile_width)
        
        tile_utmy_array = np.linspace(tile_characteristics_subset["min_utmy_tile"].values[0], 
                                      tile_characteristics_subset["max_utmy_tile"].values[0],
                                      tile_height)
        utm_proj = tile_characteristics_subset["utm_projection"].values[0]
        #load each xml
        tree = et.parse(tile_xml_path)
        root = tree.getroot()
        for obj in root.iter('object'):
            xmlbox = obj.find('bndbox') #get the bboxes
            obj_xmin = xmlbox.find('xmin').text
            obj_ymin = xmlbox.find('ymin').text
            obj_xmax = xmlbox.find('xmax').text
            obj_ymax = xmlbox.find('ymax').text
            if int(obj_xmax) > tile_width:
                obj_xmax = tile_width
            if int(obj_ymax) > tile_height:
                obj_ymax = tile_height
            if (int(obj.find('difficult').text) == 1) or (int(obj.find('truncated').text) == 1): #get truncated bboxes/characteristics
                trunc_diff_objs_bboxes.append([obj_xmin, obj_ymin, obj_xmax, obj_ymax])
                trunc_diff_objs_characteristics.append([obj.find('chip_name').text, obj.find('name').text, obj.find('pose').text, 
                                                        obj.find('truncated').text, obj.find('difficult').text])
            else: #get remaining bboxes/characteristics
                remaining_objs_bboxes.append([obj_xmin, obj_ymin, obj_xmax, obj_ymax])
                remaining_objs_characteristics.append([obj.find('chip_name').text, obj.find('name').text, obj.find('pose').text, 
                                                        obj.find('truncated').text, obj.find('difficult').text])
        
        # Add merge bboxes
        trunc_diff_objs_bboxes = np.array(trunc_diff_objs_bboxes).astype(np.int32)
        trunc_diff_objs_bboxes = trunc_diff_objs_bboxes.tolist()
        merged_bools, merged_characteristics, merged_bboxes =  merge_algo(trunc_diff_objs_characteristics,
                                                                      trunc_diff_objs_bboxes, distance_limit) #merge
        for j, (merged_bool, char, bbox) in enumerate(zip(merged_bools, merged_characteristics, merged_bboxes)):
            tile_names.append(tile_name)
            chip_names.append(char[0])
            object_class.append(char[1])
            #state whether bbox were merged
            merged_bbox.append(merged_bool)
            #pixel coordiantes 
            minx_polygon_pixels.append(bbox[0])
            miny_polygon_pixels.append(bbox[1])
            maxx_polygon_pixels.append(bbox[2])
            maxy_polygon_pixels.append(bbox[3])
            polygon_vertices_pixels.append([(bbox[0],bbox[1]), (bbox[0],bbox[3]), (bbox[2],bbox[3]), (bbox[2],bbox[1])])
            #geospatial data          
            min_lon, min_lat = transform_point_utm_to_wgs84(utm_proj, tile_utmx_array[bbox[0]-1], tile_utmy_array[bbox[1]-1])
            max_lon, max_lat = transform_point_utm_to_wgs84(utm_proj, tile_utmx_array[bbox[2]-1], tile_utmy_array[bbox[3]-1])
            nw_corner_polygon_lon.append(min_lon)
            nw_corner_polygon_lat.append(min_lat)
            se_corner_polygon_lon.append(max_lon)
            se_corner_polygon_lat.append(max_lat)
            polygon_vertices_lon_lat.append([(min_lon,min_lat),(min_lon,max_lat),(max_lon,max_lat),(max_lon,min_lat)])
            geometry.append(Polygon([(min_lon,min_lat),(min_lon,max_lat),(max_lon,max_lat),(max_lon,min_lat)]))
            #calculate diameter
            diameter.append(calculate_diameter(bbox))
            
        #Add remaining bboxes
        remaining_objs_bboxes = np.array(remaining_objs_bboxes).astype(np.int32)
        remaining_objs_bboxes = remaining_objs_bboxes.tolist()
        for j, (char, bbox) in enumerate(zip(remaining_objs_characteristics,remaining_objs_bboxes)):
            tile_names.append(tile_name)
            chip_names.append(char[0])
            object_class.append(char[1])
            #state whether bbox were merged
            merged_bbox.append(merged_bool)
            #pixel coordiantes 
            minx_polygon_pixels.append(bbox[0])
            miny_polygon_pixels.append(bbox[1])
            maxx_polygon_pixels.append(bbox[2])
            maxy_polygon_pixels.append(bbox[3])
            polygon_vertices_pixels.append([(bbox[0],bbox[1]), (bbox[0],bbox[3]), (bbox[2],bbox[3]), (bbox[2],bbox[1])])
            #geospatial data
            min_lon, min_lat = transform_point_utm_to_wgs84(utm_proj, tile_utmx_array[bbox[0]-1], tile_utmy_array[bbox[1]-1])
            max_lon, max_lat = transform_point_utm_to_wgs84(utm_proj, tile_utmx_array[bbox[2]-1], tile_utmy_array[bbox[3]-1])
            nw_corner_polygon_lon.append(min_lon)
            nw_corner_polygon_lat.append(min_lat)
            se_corner_polygon_lon.append(max_lon)
            se_corner_polygon_lat.append(max_lat)
            polygon_vertices_lon_lat.append([(min_lon,min_lat), (min_lon,max_lat), (max_lon,max_lat), (max_lon,min_lat)])
            geometry.append(Polygon([(min_lon,min_lat), (min_lon,max_lat), (max_lon,max_lat), (max_lon,min_lat)]))
            #calculate diameter
            diameter.append(calculate_diameter(bbox))
            
    #create geodatabase
    d = {'tile_name': tile_names,'chip_name': chip_names, 
            "minx_polygon_pixels": minx_polygon_pixels, "miny_polygon_pixels": miny_polygon_pixels, #min lon/lat
            "maxx_polygon_pixels": maxx_polygon_pixels, "maxy_polygon_pixels": maxy_polygon_pixels, #max lat
            "polygon_vertices_pixels": polygon_vertices_pixels, 
            "nw_corner_polygon_lat": nw_corner_polygon_lat, "nw_corner_polygon_lon": nw_corner_polygon_lon,#min lon/lat
            "se_corner_polygon_lat": se_corner_polygon_lat, "se_corner_polygon_lon": se_corner_polygon_lon, #min lon/lat
            "polygon_vertices_lon_lat": polygon_vertices_lon_lat,'geometry': geometry, 
            "object_class": object_class, 'diameter (m)': diameter, 'merged_bbox': merged_bbox}
    gdf = gpd.GeoDataFrame(d)
    return(gdf)  

11

######################################################################################################################################################
######################################               Inundation Values for Tile Database            ##################################################
######################################################################################################################################################
      
def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    return [json.loads(gdf.to_json())['features'][0]['geometry']]

def identify_inundation_for_tanks(gdf, sc_slosh_inundation_map_path):                       
    #identify inundation bounds                               
    category = []
    geometry = []
    for i in range(1,6): #get the bounding box polygons
        sc_slosh_inundation_map_name = "SC_Category" + str(i) + "_MOM_Inundation_HighTide_EPSG4326.tif"
        sc_slosh_inundation_map = rasterio.open(os.path.join(sc_slosh_inundation_map_path, sc_slosh_inundation_map_name))
        min_lon, min_lat, max_lon, max_lat = sc_slosh_inundation_map.bounds
        category.append("SC_Category" + str(i))
        geometry.append(Polygon([(min_lon,min_lat),(min_lon,max_lat),(max_lon,max_lat),(max_lon,min_lat)]))

    #make dataframe of inundation map bounds
    d = {'category': category,'geometry': geometry}
    sc_slosh = gpd.GeoDataFrame(d)

    #idntify useful bounding box
    if sc_slosh["geometry"].nunique() == 1: #all of the bounding boxes for the inundation maps are the same
        sc_inundation_poly = sc_slosh["geometry"].unique()[0] 

    #create dictionary for inundation values for each tank
    inundation_level_for_tank = {}
    for i in range(1,6): 
        inundation_level_for_tank["Category" + str(i)] = np.zeros((len(gdf)))

    #make a list to record whether the inundation level has been recorded
    bbox_within_inundation_bounds = [False] * len(gdf)

    #get inundation values
    for index, poly in enumerate(gdf["geometry"]): #iterate over the polygons
        if sc_inundation_poly.contains(poly): #identify whether the bbox is inside of the inundation map
            bbox_within_inundation_bounds[index] = True #record that the bbox is within the inundation map
            #make a geodatabaframe for each polygon that is 
            geo = gpd.GeoDataFrame({'geometry': poly}, index=[0], crs="EPSG:4326")
            coords = getFeatures(geo)
            for i in range(1,6): #get the bounding box polygons
                sc_slosh_inundation_map_name = "SC_Category" + str(i) + "_MOM_Inundation_HighTide_EPSG4326.tif"
                sc_slosh_inundation_map = rasterio.open(os.path.join(sc_slosh_inundation_map_path, sc_slosh_inundation_map_name))
                out_img, out_transform = rasterio.mask.mask(dataset=sc_slosh_inundation_map, shapes=coords, crop=True)
                if np.all(out_img == 255): #check if all inundation values are equal the no value entry (255)
                    inundation_level_for_tank["Category" + str(i)][index] = 0
                else: 
                    out_img = np.where(out_img >= 255, 0, out_img)
                    inundation_level_for_tank["Category" + str(i)][index] = np.average(out_img)

    #add inundation values to tank database 
    gdf["bbox_within_inundation_bounds"] = bbox_within_inundation_bounds
    for i in range(1,6): 
        gdf["Category" + str(i)] = inundation_level_for_tank["Category" + str(i)]
    return(gdf)
######################################################################################################################################################
######################################                   State Names for Tile Database              ##################################################
######################################################################################################################################################
def identify_state_name_for_each_state(states_gpds_path, gdf):
    #https://gis.stackexchange.com/questions/251812/returning-percentage-of-area-of-polygon-intersecting-another-polygon-using-shape
    states_gpds = gpd.read_file(states_gpds_path)
    states_gds_epsg4326 = states_gpds.to_crs(epsg=4326) #reproject to lat lon

    #get state for each polygon 
    state_list = [None] * len(gdf)
    for tank_index, tank_poly in tqdm.tqdm(enumerate(gdf["geometry"])): #iterate over the tank polygons
        for state_index, state_poly in enumerate(states_gds_epsg4326["geometry"]): #iterate over the state polygons
            if state_poly.intersects(tank_poly) or state_poly.contains(tank_poly): #identify whether the tank bbox is inside of the state polygon
                if state_list[tank_index] == None: 
                    state_list[tank_index] = states_gds_epsg4326.iloc[state_index]["NAME"] #add state name for each tank to list 
                else:
                    index, = np.where(states_gds_epsg4326["NAME"] == state_list[tank_index]) #check percent of tank that intersects with current state 
                    prev_state_poly = states_gds_epsg4326["geometry"][index[0]]
                    prev_state_poly_intersection_area = tank_poly.intersection(prev_state_poly).area/tank_poly.area #check percent of tank that intersects with prev_state_poly
                    
                    proposed_state_poly_intersection_area = tank_poly.intersection(state_poly).area/tank_poly.area #check percent of tank that intersects with proposed state 
                    if proposed_state_poly_intersection_area > prev_state_poly_intersection_area: #change the state if the polygon mainly resides in a different state
                        state_list[tank_index] = states_gds_epsg4326.iloc[state_index]["NAME"]

    #add states to dataframe 
    state_list = np.array(state_list)
    gdf["state"] = state_list
    return(gdf)
######################################################################################################################################################
######################################                      Write Tile Level Annotations            ##################################################
######################################################################################################################################################

def write_gdf(gdf, output_filepath, output_filename = 'tile_level_annotations'):
    gdf.crs = "EPSG:4326" #assign projection

    #save geodatabase as json
    with open(os.path.join(output_filepath, output_filename+".json"), 'w') as file:
        file.write(gdf.to_json()) 

    ##save geodatabase as geojson 
    with open(os.path.join(output_filepath, output_filename+".geojson"), "w") as file:
        file.write(gdf.to_json()) 

    ##save geodatabase as shapefile
    gdf_shapefile = gdf.drop(columns=["chip_name","polygon_vertices_pixels","polygon_vertices_lon_lat"])
    gdf_shapefile.to_file(os.path.join(output_filepath,output_filename+".shp"))
######################################################################################################################################################
###################################### Identify unlabeled images (cut off by previous chipping code ##################################################
######################################################################################################################################################
    
def incorrectly_chipped_image_and_correctly_chipped_names(incorrectly_chipped_images_path, remaining_chips_path, tiles_complete_dataset_path, tile_name):
    """ Load tile of interest; chip the tile using the mxm, chip dimensions where m > n; Gather the previous chip name format, and the new chip name format;
    save all images, record labeled images that contain relevant data (not black images); save images that were not labeled images; 
    Args:
    incorrectly_chipped_images_path(str): path to folder that will contain all of the incorrect named, images chipped from times
    remaining_chips_path(str): path to folder that will contain all of the remaining images that have not been labeled and correspond to tiles that have labeled images
    tiles_complete_dataset_path(str): path to folder containing tiles
    tile_name(str): name of tile without of extension
    Returns:
    ys(list): list of row indices 
    xs(list): list of column indicies
    chip_name_incorrectly_chip_names(np array): the name of the images following the previous format for images that contain relevant data
    chip_name_correct_chip_names(np array): the name of the images following the previous format for images that contain relevant data
    """
    item_dim = int(512)
    tile = cv2.imread(os.path.join(tiles_complete_dataset_path, tile_name + ".tif")) 
    tile_height,  tile_width,  tile_channels = tile.shape #the size of the tile 
    row_index = math.ceil(tile_height/512) 
    col_index = math.ceil(tile_width/512)
    #print(row_index, col_index)

    chip_name_incorrectly_chip_names = []
    chip_name_correct_chip_names = []
    ys = []
    xs = []

    count = 1            
    for y in range(0, row_index): #rows
        for x in range(0, row_index): #cols
            chip_img = tile_to_chip_array(tile, x, y, item_dim)

            #specify the chip names
            chip_name_incorrect_chip_name = tile_name + '_'+ str(count).zfill(6) + '.jpg'
            chip_name_correct_chip_name = tile_name + '_' + f"{y:02}"  + '_' + f"{x:02}" + '.jpg' # row_cols
            if not os.path.exists(os.path.join(incorrectly_chipped_images_path, chip_name_incorrect_chip_name)):
                cv2.imwrite(os.path.join(incorrectly_chipped_images_path, chip_name_incorrect_chip_name), chip_img) #save images       

            #save names of labeled images  
            if (y < col_index) and (x < col_index): #y:rows already annotated #x:(cols) images that contain relevant data (excludes extraneous annotations of black images)
                ys.append(y)#row index
                xs.append(x)#col index
                chip_name_incorrectly_chip_names.append(chip_name_incorrect_chip_name)
                chip_name_correct_chip_names.append(chip_name_correct_chip_name) # The index is a six-digit number like '000023'. 

            #save remaining images 
            if (y >= col_index) and (x < col_index): #y: we started at 0 here and 1 before? (save the remaining rows) #x:do not include extraneous black images
                if not os.path.exists(os.path.join(remaining_chips_path, chip_name_correct_chip_name)):
                    cv2.imwrite(os.path.join(remaining_chips_path, chip_name_correct_chip_name), chip_img)   

            #counter for image pathway
            count += 1  
    
    chip_name_incorrectly_chip_names = np.array(chip_name_incorrectly_chip_names)
    chip_name_correct_chip_names = np.array(chip_name_correct_chip_names)
    return(ys,xs,chip_name_incorrectly_chip_names, chip_name_correct_chip_names)

def reformat_xmls_for_rechipped_images(xml_directory, image_in_tile, correct_xml_name, correct_jpg_name, chips_positive_xml_dir_path):
    """ reformat xml files for rechipped images to include resolution, year, updated filename, and updated path. 
    Args:
    xml_directory(str): directory holding xmls
    image_in_tile(str): path to image
    correct_xml_name: correct name for xml
    correct_jpg_name: correct name for jpgs
    chips_positive_xml_dir_path(str): new path for xml
    
    https://docs.python.org/3/library/xml.etree.elementtree.html
    https://stackoverflow.com/questions/28813876/how-do-i-get-pythons-elementtree-to-pretty-print-to-an-xml-file
    https://stackoverflow.com/questions/28813876/how-do-i-get-pythons-elementtree-to-pretty-print-to-an-xml-file
    """
    #load xml
    formatted_chip_name_wo_ext = os.path.splitext(os.path.basename(image_in_tile))[0]
    tree = et.parse(os.path.join(xml_directory, formatted_chip_name_wo_ext +".xml"))
    root = tree.getroot() 
    
    #add resolution to xml
    resolution = et.Element("resolution")
    resolution.text = formatted_chip_name_wo_ext.split("_")[1] #resolution
    et.indent(tree, space="\t", level=0)
    root.insert(3, resolution)
    
    #add year to xml
    year = et.Element("year")
    year.text = formatted_chip_name_wo_ext.split("_")[2]#year
    et.indent(tree, space="\t", level=0)
    root.insert(4,year)
    
    #correct spacing for source (dataset name)
    et.indent(tree, space="\t", level=0)
    
    #correct filename and path to formatting with row/col coordinates
    for filename in root.iter('filename'):
        filename.text = correct_xml_name
    for path in root.iter('path'):
        path.text = os.path.join(xml_directory, correct_jpg_name)

    tree.write(os.path.join(chips_positive_xml_dir_path, correct_xml_name))       
    
def copy_rename_labeled_images_xmls(xml_directory, images_in_tile, incorrectly_chipped_images_path, chips_positive_dir_path, chips_positive_xml_dir_path,
                                    chip_name_incorrectly_chip_names, chip_name_correct_chip_names, multiple_annotations_images, black_images_with_annotations):
    """ reformat xml files for rechipped images to include resolution, year, updated filename, and updated path. 
    Args:
    xml_directory(str): directory holding xmls
    image_in_tile(str): path to image
    incorrectly_chipped_images_path(str): path to folder that will contain all of the incorrect named, images chipped from times
    chips_positive_dir_path(str): new path for jpg
    chips_positive_xml_dir_path(str): new path for xml
    chip_name_incorrectly_chip_names(np array): the name of the images following the previous format for images that contain relevant data
    chip_name_correct_chip_names(np array): the name of the images following the previous format for images that contain relevant data
    multiple_annotations_images: list of images with multiple annotations in a given folder
    black_images_with_annotations: list of black images with annotations
    Return:
    ultiple_annotations_images, black_images_with_annotations
    """
    for image_in_tile in images_in_tile:
        #get the standard image name
        formatted_image_name = os.path.basename(image_in_tile)
        standard_image_name = formatted_image_name.split("_",4)[-1]
        #get the index of the image array of image names #print(standard_image_name)
        index, = np.where(chip_name_incorrectly_chip_names == standard_image_name)

        if len(index) == 0: #If there there is no matching annotation (black image)
            black_images_with_annotations.append(image_in_tile)
            
        elif len(index) >= 1: #If there is a match
            #make sure the image in the folder is correct   
            gray_labeled_image = cv2.cvtColor(cv2.imread(image_in_tile), cv2.COLOR_BGR2GRAY) #image that had been labeled
            incorrectly_chipped_image_path = os.path.join(incorrectly_chipped_images_path, chip_name_incorrectly_chip_names[index[0]])
            gray_known_image = cv2.cvtColor(cv2.imread(incorrectly_chipped_image_path), cv2.COLOR_BGR2GRAY) #image that has been chipped from tile
            (score, diff) = compare_ssim(gray_labeled_image, gray_known_image, full=True)

            if score >= 0.90: #If the labeled image is correct
                #chip_name_incorrectly_chip_names[index]
                correct_xml_name = os.path.splitext(chip_name_correct_chip_names[index[0]])[0] + ".xml"
                correct_jpg_name = os.path.splitext(chip_name_correct_chip_names[index[0]])[0] + ".jpg"
                #copy image
                shutil.copyfile(incorrectly_chipped_image_path, os.path.join(chips_positive_dir_path, correct_jpg_name))
                #create renamed xml
                reformat_xmls_for_rechipped_images(xml_directory, image_in_tile, correct_xml_name, correct_jpg_name, chips_positive_xml_dir_path)
                
        if len(index) > 1: #record images with multiple annotations
            multiple_annotations_images.append(image_in_tile)
    return(multiple_annotations_images, black_images_with_annotations)
    
    
def img_anno_paths_to_corrected_names_for_labeled_images_and_remaining_images(img_paths_anno_paths, correct_directory, incorrectly_chipped_images_path, 
                                                                              remaining_chips_path, tiles_complete_dataset_path):
    """ iterate over all the image and xml paths in directory of annotated images; identify tiles and the corresonding images/xmls in each folder;
    match name of previous naming convention and row/col naming convention;for labeled images and xmls, create folder to store, 
    identify correct images, copy, and rename; identify and save remaining images; 
    Args: 
    img_paths_anno_paths(np array): n x 2 array of jpg and xml paths
    correct_directory(str): path to directory containing xmls and images with correct names
    incorrectly_chipped_images_path(str): path to folder that will contain all of the incorrect named, images chipped from times
    remaining_chips_path(str): path to folder that will contain all of the remaining images that have not been labeled and correspond to tiles that have labeled images
    tiles_complete_dataset_path(str): path to folder containing tiles
    
    Returns:
    multiple_annotations_images: list of images with multiple annotations in a given folder
    black_images_with_annotations: list of black images with annotations
    """
    multiple_annotations_images = []
    black_images_with_annotations = []
    for directory in tqdm.tqdm(img_paths_anno_paths):
        #get all the image and xml paths in directory of annotated images
        print(directory)
        remove_thumbs(directory[0])
        image_paths = glob(directory[0] + "/*.jpg", recursive = True)
        xml_paths = glob(directory[1] + "/*.xml", recursive = True)
        #print(len(image_paths),len(xml_paths))

        #identify tiles in each folder
        tiles = []
        for image in image_paths:
            image_name = os.path.splitext(os.path.basename(image))[0]
            tile_name = image_name.split("_",4)[-1].rsplit("_",1)[0]
            tiles.append(tile_name)
        tiles = np.unique(tiles)

        #identify the images/xmls that correspond with each tile in folder
        for tile_name in tiles:        
            images_in_tile = [string for string in image_paths if tile_name in string]          
            xmls_in_tile = [string for string in xml_paths if tile_name in string]  
            assert len(images_in_tile) == len(xmls_in_tile), "The same number of images and xmls"
            #print(tile_name, len(images_in_tile))

            #create folder to store corrected chips/xmls
            tile_dir_path = os.path.join(correct_directory, tile_name) #sub folder for each tile 
            chips_positive_dir_path = os.path.join(tile_dir_path,"chips_positive") #images path
            chips_positive_xml_dir_path = os.path.join(tile_dir_path,"chips_positive_xml") #xmls paths

            tile_dir = os.makedirs(tile_dir_path, exist_ok=True)
            chips_positive_dir = os.makedirs(chips_positive_dir_path, exist_ok=True)
            chips_positive_xml_dir = os.makedirs(chips_positive_xml_dir_path, exist_ok=True)

            #identify and save remaining images; match name of previous naming convention and row/col naming convention
            ys, xs, chip_name_incorrectly_chip_names, chip_name_correct_chip_names =  incorrectly_chipped_image_and_correctly_chipped_names(incorrectly_chipped_images_path, remaining_chips_path,                                                                                        tiles_complete_dataset_path, tile_name)

            #identify labeled images that are correct; copy and rename correct images and xmls
            multiple_annotations_images, black_images_with_annotations = copy_rename_labeled_images_xmls(directory[1], images_in_tile, incorrectly_chipped_images_path,
                                                                                                            chips_positive_dir_path, chips_positive_xml_dir_path,
                                                                                                            chip_name_incorrectly_chip_names, chip_name_correct_chip_names,
                                                                                                            multiple_annotations_images, black_images_with_annotations)
        #remaining images
        print("remaining images", len(os.listdir(remaining_chips_path)))
        
        
###########################################################################################################
############################# Identify incorrect/correct images ###########################################
def identify_correct_images(tile_dir, tiles_in_directory, 
                            images_in_directory, images_in_directory_array,
                            image_directories):
    """
    Find images that do not align with the tiles chipped ####flipped rows and columns####
    Confirm that the standard tile name in the chip and the contents of the chip match
    """
    #index over the tiles with corresponding images in the given directory
    tile_names = []
    correct_chip_names = []
    correct_chip_paths = []
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
        for y in range(0, col_index): #rows
            for x in range(0, row_index): #cols
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
                        if score >= 0.90:
                            tile_names.append(tile_name)
                            xs.append(x)
                            ys.append(y)
                            correct_chip_names.append(confirmed_chip)
                            correct_chip_paths.append(os.path.join(image_directory,confirmed_chip))
                        if (score < 0.90) and (score >= 0.80):                           
                            fig, (ax1, ax2) = plt.subplots(1, 2)
                            ax1.set_title('correct_image')
                            ax1.imshow(correct_image)
                            ax2.set_title('labeled_chip_array')
                            ax2.imshow(image_in_directory_array)
                            plt.show() 

    return(tile_names, xs, ys, correct_chip_names, correct_chip_paths)


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

####################### Identify Duplicates#####################################################################
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

def positive_images_to_array_correctly_labeled(images_dir_path, incorrect_labeled_chip_names_by_subfolder):
    """
    For every image in the given subdirectory, the name and image contents have been verified and incorrect images have been identified
    This function identifies the images that are correctly labeled 
    get image array for correctly labeled images
    
    Args:
    images_dir_path(str)
    incorrect_labeled_chip_names_by_subfolder(list)
    """
    subfolders_files = os.listdir(images_dir_path)
    for chip in incorrect_labeled_chip_names_by_subfolder["incorrect_chip_names"].tolist():
        if chip in subfolders_files:
            subfolders_files.remove(chip)
        
    correctly_labeled_images = np.array(subfolders_files) #image names 
    correctly_labeled_image_array = np.zeros((len(correctly_labeled_images),512,512, 3), dtype='uint8') #image contents 
    correctly_labeled_image_directory = np.array([images_dir_path] * len(correctly_labeled_images)) #image directory 
    for num in range(len(correctly_labeled_images)):    
        correctly_labeled_image = cv2.imread(os.path.join(images_dir_path, correctly_labeled_images[num])) #open image
        correctly_labeled_image_array[num,:,:,:] = correctly_labeled_image
        
    return(correctly_labeled_images, correctly_labeled_image_array, correctly_labeled_image_directory)      
        
        
        
        
########################## Long way round ############################################
################### Keep for posterity ##############################################
def correct_images_from_chipped_tile_for_positive_images(tile_dir, tile_names, gray_incorrect_labeled_chip_image_array):#incorrect_labeled_chip_names_by_subfolder, incorrect_labeled_chip_image_array):
    """
    Find images that do not align with the tile chip
    """

    #index over the tiles with corresponding images in the given directory
    correct_images = np.zeros((len(gray_incorrect_labeled_chip_image_array), 512, 512, 3),dtype='uint8')
    correct_standard_chip_paths =  np.zeros((len(gray_incorrect_labeled_chip_image_array)))
    nums = list(range(len(gray_incorrect_labeled_chip_image_array))) 
    #tiles_paths = glob(tile_dir + "/*.tif", recursive = True)
    for tile_name in tqdm.tqdm(tile_names): 
        #tile_name = os.path.basename(tile_path) 
        #file_name, ext = os.path.splitext(tile_name) # File name
        file_name = tile_name
        #get tile shape
        item_dim = int(512)   
        tile_path = os.path.join(tile_dir, tile_name + ".tif")
        print(tile_path)
        tile = cv2.imread(tile_path) 
        tile_height,  tile_width,  tile_channels = tile.shape #the size of the tile #determine tile dimensions
        row_index = math.ceil(tile_height/512) #divide the tile into 512 by 512 chips (rounding up)
        col_index = math.ceil(tile_width/512)

        count = 1 #image names start at 1 
        
        for y in range(0, col_index): #row actually
            for x in range(0, row_index): #column actually
                chip_name = file_name + '_' + str(count).zfill(6) + '.jpg'
                
                #create a numpy array of each correctly chipped images 
                correct_image = tile_to_chip_array(tile, x, y, item_dim)
                count += 1  
                for num in nums:
                    #incorrect_image = incorrect_labeled_chip_image_array[num,:,:,:]
                    ##https://pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/
                    #https://pyimagesearch.com/2014/09/15/python-compare-two-images/
                    #gray_incorrect_image = cv2.cvtColor(incorrect_image, cv2.COLOR_BGR2GRAY)
                    gray_correct_image = cv2.cvtColor(correct_image, cv2.COLOR_BGR2GRAY)
                    (score, diff) = compare_ssim(gray_incorrect_labeled_chip_image_array[num], gray_correct_image, full=True)
                    diff = (diff * 255).astype("uint8")

                    if score >= 0.92: #Save the same images
                        print("same image")
                        correct_images[num,:,:,:] = correct_image
                        correct_standard_chip_names[num](chip_name)
                        nums.remove(num) #remove index of matched image
                        print("SSIM: {}".format(score))
                    if (score < 0.92) & (score > 0.9) :
                        fig, (ax1, ax2) = plt.subplots(1, 2)
                        ax1.set_title('correct_image')
                        ax1.imshow(correct_image)
                        ax2.set_title('labeled_chip_array')
                        ax2.imshow(labeled_chip_array)
                        plt.show() 
        if len(nums) == 0:
            break
    return(correct_images, correct_standard_chip_names)

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
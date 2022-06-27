"""
Module containing azure functions, and label work distribution and image processing functions.
"""

"""
Load Packages
"""
# Standard modules
import tempfile
import warnings
import urllib
import urllib.request
#import shutils
import shutil 
import os
import os.path
from pathlib import Path
import sys
from zipfile import ZipFile
import pickle
import math
from contextlib import suppress
from glob import glob

# Less standard, but still pip- or conda-installable
import pandas as pd
import numpy as np
import progressbar # pip install progressbar2, not progressbar
from tqdm import tqdm
import cv2

# Image processing files
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import matplotlib.image as mpimg
#import rasterio
#from rasterio.windows import Window #
import re
import rtree
import shapely
from geopy.geocoders import Nominatim
import PIL
#print('PIL',PIL.__version__)
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
#Parsing/Modifying XML
from lxml.etree import Element,SubElement,tostring
import xml.dom.minidom
from xml.dom.minidom import parseString
import xml.etree.ElementTree as et
from xml.dom import minidom
import xml
"""
Azure Functions
"""

class DownloadProgressBar():
    """
    A progressbar to show the completed percentage and download speed for each image downloaded using urlretrieve.

    https://stackoverflow.com/questions/37748105/how-to-use-progressbar-module-with-urlretrieve
    """
    
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(max_value=total_size)
            self.pbar.start()
            
        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()
                 
class NAIPTileIndex:
    """
    Utility class for performing NAIP tile lookups by location.
    """
    
    tile_rtree = None
    tile_index = None
    base_path = None

    
    def __init__(self, base_path=None):
        blob_root = 'https://naipeuwest.blob.core.windows.net/naip'
        index_files = ["tile_index.dat", "tile_index.idx", "tiles.p"]
        index_blob_root = re.sub('/naip$','/naip-index/rtree/', blob_root) 
        
        if base_path is None:
            
            base_path = os.path.join(tempfile.gettempdir(),'naip')
            os.makedirs(base_path,exist_ok=True)
            
            for file_path in index_files:
                download_url_no_destination_folder(index_blob_root + file_path, base_path + '/' + file_path,
                             progress_updater=DownloadProgressBar())
                
        self.base_path = base_path
        self.tile_rtree = rtree.index.Index(base_path + "/tile_index")
        self.tile_index = pickle.load(open(base_path  + "/tiles.p", "rb"))
      
    
    def lookup_tile(self, lat, lon):
        """"
        Given a lat/lon coordinate pair, return the list of NAIP tiles that contain
        that location.

        Returns a list of COG file paths.
        """

        point = shapely.geometry.Point(float(lon),float(lat))
        intersected_indices = list(self.tile_rtree.intersection(point.bounds))

        intersected_files = []
        tile_intersection = False

        for idx in intersected_indices:

            intersected_file = self.tile_index[idx][0]
            intersected_geom = self.tile_index[idx][1]
            if intersected_geom.contains(point):
                tile_intersection = True
                intersected_files.append(intersected_file)

        if not tile_intersection and len(intersected_indices) > 0:
            print('''Error: there are overlaps with tile index, 
                      but no tile completely contains selection''')   
            return None
        elif len(intersected_files) <= 0:
            print("No tile intersections")
            return None
        else:
            return intersected_files
    
def download_url_no_destination_folder(url, destination_filename=None, progress_updater=None, force_download=False):
    """
    Download a URL to a temporary file
    # This is not intended to guarantee uniqueness, we just know it happens to guarantee
    # uniqueness for this application.
    """
    temp_dir = os.path.join(tempfile.gettempdir(),'naip')
    os.makedirs(temp_dir,exist_ok=True)
    
    if destination_filename is None:
        url_as_filename = url.replace('://', '_').replace('/', '_')    
        destination_filename = \
            os.path.join(temp_dir,url_as_filename)
    if (not force_download) and (os.path.isfile(destination_filename)):
        print('Bypassing download of already-downloaded file {}'.format(os.path.basename(url)))
        return destination_filename
    print('Downloading file {} to {}'.format(os.path.basename(url),destination_filename),end='')
    urllib.request.urlretrieve(url, destination_filename, progress_updater)  
    assert(os.path.isfile(destination_filename))
    nBytes = os.path.getsize(destination_filename)
    print('...done, {} bytes.'.format(nBytes))
    return destination_filename

def download_url(url, destination_folder, destination_filename=None, progress_updater=None, force_download=False):
    """
    Download a URL to a a file
    Args:
    url(str): url to download
    destination_folder(str): directory to download folder
    destination_filename(str): the name for each of files to download
    return:
    destination_filename
    """
    
    # This is not intended to guarantee uniqueness, we just know it happens to guarantee
    # uniqueness for this application.
    if destination_filename is not None:
        destination_filename = os.path.join(destination_folder, destination_filename)
    if destination_filename is None:
        url_as_filename = url.replace('://', '_').replace('/', '_') 
        destination_filename = os.path.join(destination_folder, url_as_filename)
    if os.path.isfile(destination_filename):
        print('Bypassing download of already-downloaded file {}'.format(os.path.basename(url)))
        return destination_filename
  #  print('Downloading file {} to {}'.format(os.path.basename(url),destination_filename),end='')
    urllib.request.urlretrieve(url, destination_filename, progress_updater)  
    assert(os.path.isfile(destination_filename))
    nBytes = os.path.getsize(destination_filename)
    print('...done, {} bytes.'.format(nBytes))

    return destination_filename

def display_naip_tile(filename):
    """
    Display a NAIP tile using rasterio.
    """
    dsfactor = 10
    
    with rasterio.open(filename) as raster:
        # NAIP imagery has four channels: R, G, B, IR
        # Stack RGB channels into an image; we won't try to render the IR channel
        # rasterio uses 1-based indexing for channels.
        h = int(raster.height/dsfactor)
        w = int(raster.width/dsfactor)
        print('Resampling to {},{}'.format(h,w))
        r = raster.read(1, out_shape=(1, h, w))
        g = raster.read(2, out_shape=(1, h, w))
        b = raster.read(3, out_shape=(1, h, w))        
    
    rgb = np.dstack((r,g,b))
    fig = plt.figure(figsize=(7.5, 7.5), dpi=100, edgecolor='k')
    plt.imshow(rgb)
    raster.close()
    
    
def get_coordinates_from_address(address):
    """
    Look up the lat/lon coordinates for an address.
    """
    
    geolocator = Nominatim(user_agent="NAIP")
    location = geolocator.geocode(address)
    print('Retrieving location for address:\n{}'.format(location.address))
    return location.latitude, location.longitude

"""
# Functions to retrieve filepathways from EIA HFID datasources 
"""

# %%      
def lons_lat_to_filepaths(lons, lats, index):
    """
    Calculate file paths given lat and lat
    """
    all_paths = np.empty(shape=(1,8))
    for i in tqdm(range(len(lons))):
        naip_file_pathways = index.lookup_tile(lats[i], lons[i])
        if naip_file_pathways != None:
            select_path = []
            for ii in range(len(naip_file_pathways)):
                tmp = naip_file_pathways[ii].split('/')
                tmp = np.hstack((tmp, naip_file_pathways[ii].split('/')[3].split("_")[1]))
                iii = iter(tmp[5].split("_",4))
                tmp = np.hstack((tmp, list((map("_".join,zip(*[iii]*4)) ))))
                select_path.append(tmp)
            select_path = np.array(select_path)
            select_path = select_path[select_path[:,2] >= "2018"] #filter out years to get the most recent data that will include the highest resolution data

            select_path = select_path[(select_path[:,6] == "60cm") | (select_path[:,6] == "060cm")] #select only pathways with 60cm
                        
            all_paths = np.vstack((all_paths, select_path)) #add to the rest of the paths
            
    file_pathways = np.delete(all_paths, 0, axis=0)
    
    file_pathways = np.unique(file_pathways, axis=0) #select unique values
    return file_pathways


def filepaths_to_tile_name_tile_url(file_pathways):
    """
    Determine the tile name and url for a given file pathway
    """
    
    tile_name = []
    tile_url = []

    #blob_root = 'https://naipblobs.blob.core.windows.net/naip'
    blob_root = 'https://naipeuwest.blob.core.windows.net/naip'


    for i in range(len(file_pathways)):
        tile_name.append(file_pathways[i,5])
        
        # Tiles are stored at: [blob root]/v002/[state]/[year]/[state]_[resolution]_[year]/[quadrangle]/filename
        tile_url.append(blob_root + '/v002/' + file_pathways[i,1] + '/'+ file_pathways[i,2] + '/' \
                        + file_pathways[i,3] +'/'+ file_pathways[i,4] +'/'+ file_pathways[i,5] )
                        
    return (tile_name, tile_url)


"""
Function to retrieve file pathways from Group identified ASTs
"""
def collected_quads_to_tile_name_tile_url(quads):
    """
    Read in a excel sheet which includes the quadrangle 
    """
    
    tile_name = []
    tile_url = []
    file_name_index = {'m': 0, 'qqname': 1, 'direction': 2,'YY': 3, 'resolution': 4,'capture_date': 5,'version_date': 5}
    blob_root = 'https://naipblobs.blob.core.windows.net/naip'
    two_digit_state_resolution = ["al","ak","az","ar","ca", "co","ct","de","fl","ga",
                                  "hi","id","il","in","ia", "ks","ky","la","me","md",
                                  "ma","mi","mn","ms","mo", "mt","ne","nv","nh","nj",
                                  "nm","ny","nc","nd","oh", "ok","or","pa","ri","sc",
                                  "sd","tn","tx","ut","vt", "va", "wa","wv","wi","wy"]

    for i in range(len(quads)):
        file_name = quads.iloc[i,3].split('_') #filename
        state = quads.iloc[i,6].lower() #state
        year = quads.iloc[i,5] # YYYY
        if state in two_digit_state_resolution:
            resolution = file_name[file_name_index["resolution"]][1:3]+"cm"
        else:
            resolution = file_name[file_name_index["resolution"]]+"cm"
        quadrangle = file_name[file_name_index["qqname"]][0:5] #qqname
        
        tile_name.append(quads.iloc[i,3] +'.tif')
        tile_url.append(blob_root + '/v002/' + state + '/' + str(year)+ '/' + state + '_' + resolution \
                    + '_' + str(year) + '/' + str(quadrangle) + '/' + tile_name[i])
        # Tiles are stored at: [blob root]/v002/[state]/[year]/[state]_[resolution]_[year]/[quadrangle]/filename
    return (tile_name, tile_url)


def tile_characeteristics(tile_name_tile_url_eia_hfid_thirty_ports):
    """tabulates the tile characteristics (the states, year resolution ranges), 
      returns the tile charcateristics 
       (quadrange names, the filenames,the states, year resolution ranges)

    Args:
        file_loc (str): The file location of the spreadsheet
        print_cols (bool): A flag used to print the columns to the console
            (default is False)

    Returns:
        list: a list of strings representing the header columns
    """
    
    state_array = np.empty((len(tile_name_tile_url_eia_hfid_thirty_ports), 1), dtype = object)
    year_array = np.empty((len(tile_name_tile_url_eia_hfid_thirty_ports), 1))
    quad_array = np.empty((len(tile_name_tile_url_eia_hfid_thirty_ports), 1))
    resolution_array = np.empty((len(tile_name_tile_url_eia_hfid_thirty_ports), 1), dtype = object)
    filename_array = np.empty((len(tile_name_tile_url_eia_hfid_thirty_ports), 1), dtype = object)

    for i in range(len(tile_name_tile_url_eia_hfid_thirty_ports)):
        state_array[i] = tile_name_tile_url_eia_hfid_thirty_ports[i,1].split('/')[5]
        year_array[i] = tile_name_tile_url_eia_hfid_thirty_ports[i,1].split('/')[6]
        quad_array[i] = tile_name_tile_url_eia_hfid_thirty_ports[i,1].split('/')[8]
        filename_array[i] = tile_name_tile_url_eia_hfid_thirty_ports[i,1].split('/')[9]
        resolution_array[i] = tile_name_tile_url_eia_hfid_thirty_ports[i,1].split('/')[-3].split('_')[1]
  
    num_states = len(np.unique(state_array))
    state_abbreviations = np.unique(state_array)
    years = np.unique(year_array)
    resolutions = np.unique(resolution_array)
    
    print("the number of tiles includes", len(tile_name_tile_url_eia_hfid_thirty_ports))
    print("The number of states included", num_states)
    print("Postal abriviations of the states included", state_abbreviations)
    print("The years in which the images were collected", years)
    print("The resolutions of the images", resolutions)
    
    return num_states, state_abbreviations, years, resolutions, quad_array, filename_array

"""
Tile distribution functions
"""
class annotator:
    def __init__(self, sub_directory):
        self.sub_directory = sub_directory
        
    def state_dcc_directory(self, dcc_directory):
        self.dcc_directory = dcc_directory    
        
    def number_of_tiles(self, num_tiles):
        self.num_tiles = num_tiles
        
    def get_tile_urls(self, tile_name_tile_url_unlabeled):
        """
        self.tile_name_tile_url_unlabeled: npy array of the initial tiles that have not been labeled
        self.tile_name_tile_url_tiles_for_annotators: npy array of the tiles to be allocated to the annotator 
        """
        
        self.tile_name_tile_url_unlabeled = np.load(tile_name_tile_url_unlabeled) #the tiles that have not yet been labeled to date
        print("Unlabeled Tiles",self.tile_name_tile_url_unlabeled.shape)
        self.tile_name_tile_url_tiles_for_annotators = self.tile_name_tile_url_unlabeled[range(self.num_tiles),:] #create an array of the tiles that will be allocated to this annotator
        
        self.tile_url = self.tile_name_tile_url_tiles_for_annotators[:,1] #get the urls of the tiles that will allocated to the annotator
    
    def track_tile_annotations(self, tile_name_tile_url_labeled):
        """
        self.tile_name_tile_url_remaining: npy array of the remaining tiles to be annotated; this will then be passed in the next iteration
        self.tile_name_tile_url_labeled: npy array of the tiles labeled
        """
        
        self.tile_name_tile_url_labeled = np.load(tile_name_tile_url_labeled) #the tiles that have not yet been labeled to date
        self.tile_name_tile_url_labeled = np.concatenate((self.tile_name_tile_url_labeled, self.tile_name_tile_url_tiles_for_annotators), axis=0)
        print("Labeled Tiles", self.tile_name_tile_url_labeled.shape)

        self.tile_name_tile_url_remaining = np.delete(self.tile_name_tile_url_unlabeled, range(self.num_tiles), 0) #the numpy array of the remaining tiles 
                                                                                  #(remove the tiles that the annotator is labeling)
        print(self.tile_name_tile_url_remaining.shape)
   
        if len(self.tile_name_tile_url_tiles_for_annotators) + len(self.tile_name_tile_url_remaining) != len(self.tile_name_tile_url_unlabeled):
            raise Exception("The number of remaining tiles and the tiles allocated to annotaters is less \
                             than the number of tiles passed through this function")
    
    def make_subdirectories(self):
        self.new_dir = self.dcc_directory + "/" + self.sub_directory
        
        os.makedirs(self.new_dir, exist_ok = True)

        self.tiles_dir = os.path.join(self.new_dir,'tiles') #directory for the naip data 
        os.makedirs(self.tiles_dir,exist_ok=True)

        self.chips_dir = os.path.join(self.new_dir,'chips') #directory to hold chips that are clipped from naip tiles
        os.makedirs(self.chips_dir,exist_ok=True)

        self.chips_positive_dir = os.path.join(self.new_dir,'chips_positive') #directory to hold chips with tanks
        os.makedirs(self.chips_positive_dir,exist_ok=True)

        self.chips_negative_dir = os.path.join(self.new_dir,'chips_negative') #directory to hold chips with tanks
        os.makedirs(self.chips_negative_dir,exist_ok=True)

        self.chips_xml_dir = os.path.join(self.new_dir,'chips_positive_xml') #directory to hold xml files
        os.makedirs(self.chips_xml_dir,exist_ok=True)
        
        #Make directory to store all xml after correction
        self.chips_positive_corrected_xml_dir = os.path.join(self.new_dir,"chips_positive_corrected_xml")
        os.makedirs(self.chips_positive_corrected_xml_dir, exist_ok = True)
    
    def download_images(self):
        destination_of_filenames = [] #use so that we can index over the file names for processing later
        for i in range(self.num_tiles):
            print(i)
            destination_of_filenames.append(download_url(self.tile_url[i], self.tiles_dir,
                                                         progress_updater=DownloadProgressBar()))
        return destination_of_filenames
    
    def tile_rename(self):
        """Rename all the tiles into the standard format outlined in repo readme 
        """

        self.tile_names = os.listdir(self.tiles_dir) #get a list of all of the tiles in tiles directory
        print(self.tile_names)
        
        for tile_name in self.tile_names: 
            tile_name_split = tile_name.split('_')
            old_tile_path = os.path.join(self.tiles_dir, tile_name)
            new_tile_path = os.path.join(self.tiles_dir, tile_name_split[6]+'_'+tile_name_split[7]+'_'+tile_name_split[8]+'_'+tile_name_split[9]+'_'+  \
                                                    tile_name_split[10]+'_'+tile_name_split[11]+'_'+tile_name_split[12]+'_'+tile_name_split[13]+'_'+  \
                                                    tile_name_split[14]+'_'+tile_name_split[15].split(".")[0]+".tif")

            if os.path.isfile(new_tile_path):
                print('Bypassing download of already-downloaded file {}'.format(os.path.basename(new_tile_path)))
            
            else:
                os.rename(old_tile_path, new_tile_path)
                
    def tile_rename_standard(self):
        """Rename all the tiles into the standard format outlined in repo readme 
        """

        self.tile_names = os.listdir(self.tiles_dir) #get a list of all of the tiles in tiles directory
        print(self.tile_names)
        
        for tile_name in self.tile_names: 
            tile_name_split = tile_name.split('_')
            old_tile_path = os.path.join(self.tiles_dir, tile_name)
            new_tile_path = os.path.join(self.tiles_dir, tile_name_split[6]+'_'+tile_name_split[7]+'_'+tile_name_split[8]+'_'+tile_name_split[9]+'_'+  \
                                                    tile_name_split[10]+'_'+tile_name_split[11]+'_'+tile_name_split[12]+'_'+tile_name_split[13]+'_'+  \
                                                    tile_name_split[14]+'_'+tile_name_split[15].split(".")[0]+".tif")

            if os.path.isfile(new_tile_path):
                print('Bypassing download of already-downloaded file {}'.format(os.path.basename(new_tile_path)))
            
            else:
                os.rename(old_tile_path, new_tile_path)
    def chip_tiles(self):
        """Segment tiles into 512 x 512 pixel chips, preserving resolution
        """
        print("chip tiles")
        self.tile_names = os.listdir(self.tiles_dir) #get a list of all of the tiles in tiles directory
        for tile_name in self.tile_names: #index over the tiles in the tiles_dir 
            file_name, ext = os.path.splitext(tile_name) # File name
            print(tile_name)
            item_dim = int(512)
            count = 1            
            tile = cv2.imread(os.path.join(self.tiles_dir, tile_name)) 
            tile_height,  tile_width,  tile_channels = tile.shape #the size of the tile 

            #divide the tile into 512 by 512 chips (rounding up)
            row_index = math.ceil(tile_height/512) 
            col_index = math.ceil(tile_width/512)
            #print(row_index, col_index)

            for y in range(0, row_index):
                for x in range(0, col_index):
                    #https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python
                    chip_img = tile[y*item_dim:y*item_dim+item_dim, x*(item_dim):x*(item_dim)+item_dim]

                    #specify the path to save the image
                    chip_name_correct_chip_name = file_name + '_' + f"{y:02}"  + '_' + f"{x:02}" + '.jpg' #

                    chips_save_path = os.path.join(self.chips_dir, chip_name_correct_chip_name) # row_col.jpg

                    #add in back space if it is the edge of an image
                    if (chip_img.shape[0] != 512) & (chip_img.shape[1] != 512): #width
                        #print("Incorrect Width")
                        chip = np.zeros((512,512,3))
                        chip[0:chip_img.shape[0], 0:chip_img.shape[1]] = chip_img
                        chip_img = chip
                    if chip_img.shape[0] != 512:  #Height
                        #print("Incorrect Height")
                        black_height = 512  - chip_img.shape[0] #Height
                        black_width = 512 
                        black_img = np.zeros((black_height,black_width,3), np.uint8)
                        chip_img = np.concatenate([chip_img, black_img])
                    if chip_img.shape[1] != 512: #width
                        #print("Incorrect Width")
                        black_height = 512 
                        black_width = 512 - chip_img.shape[1] #width
                        black_img = np.zeros((black_height,black_width,3), np.uint8)
                        chip_img = np.concatenate([chip_img, black_img],1)
                    #save image
                    cv2.imwrite(os.path.join(chips_save_path), chip_img)    
                    #counter for image pathway
                    count += 1  
            print(count)

    def copy_positive_images(self):
        """seperate out positive chips into specific directory.
        """
        
        # Input .xml files' names
        print("it ran")
        for annotation in os.listdir(self.chips_xml_dir): #iterate through the annotations
            annotation_filename = os.path.splitext(annotation)[0]
            
            for image in os.listdir(self.chips_dir): #iterate through the images
                image_filename = os.path.splitext(image)[0]
                if image_filename == annotation_filename: 
                    shutil.copy(os.path.join(self.chips_dir, image), self.chips_positive_dir) # copy images with matching .xml files in the "chips_tank" folder
        print("it finished")
        
    def copy_negative_images(self):
        """seperate out negative chips into specific directory.
        """
        
        print("it ran")
        for image in os.listdir(self.chips_dir):
            shutil.copy(os.path.join(self.chips_dir, image), self.chips_negative_dir)  # copy all chips into negative folder

        for annotation in os.listdir(self.chips_xml_dir):
            annotation_filename = os.path.splitext(annotation)[0]

            for image in os.listdir(self.chips_dir):
                image_filename = os.path.splitext(image)[0]
                if image_filename == annotation_filename: 
                    os.remove(os.path.join(self.chips_negative_dir, image)) #delete positive images according to the .xml files
        print("it finished")
        
    def correct_inconsistent_labels_xml(self):
        #Define lists of positive images and xml files
        self.chips_positive_list = glob(self.chips_positive_dir + '/*.jpg') #os.listdir(img_path) 
        self.chips_xml_list =  os.listdir(self.chips_xml_dir)
        
        #calculate the number of images
        number_of_images = len(self.chips_xml_list)

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
        #calculate the number of images
        number_of_images = len(self.chips_xml_list)
        
        #"enumerate each image" This chunk is actually just getting the paths for the images and annotations
        for i in range(len(self.chips_xml_list)):
            xml_file = self.chips_xml_list[i]
            # use the parse() function to load and parse an XML file
            tree = et.parse(os.path.join(self.chips_xml_dir, xml_file))
            root = tree.getroot()         

            for obj in root.iter('object'):
                for name in obj.findall('name'):
                    if name.text not in correctly_formatted_object:
                        name.text = object_dict[name.text]
                        
                if int(obj.find('difficult').text) == 1:
                    obj.find('truncated').text = '1'
                    obj.find('difficult').text = '1'
                    
            tree.write(os.path.join(self.chips_positive_corrected_xml_dir, xml_file))       
            
            
    def move_images_annotations_to_complete_dataset(self, include_tiles = False, original = True):
        """seperate out all of the positive chips, annotations, and conditionally tiles from one directory into a new folder.
        Args:
            file_loc (str): The file location of the spreadsheet
            include_tiles (bool; default = False): Specifies whether the full tiles should be moved
            original  include_tiles (bool; default = True): Specifies whether the original annotation in chips positive or the corrected 
                                                          annotation in chips_positive_xml should be used

        Returns:
            len(annotations): number of annotations
            len(images): number of images
        """
        #make a complete dataset
        self.complete_dataset_xml_dir = os.path.join(self.dcc_directory,"complete_dataset",'chips_positive_xml') 
        os.makedirs(self.complete_dataset_xml_dir, exist_ok=True) #directory to hold entire dataset annotations
        self.complete_dataset_chips_dir = os.path.join(self.dcc_directory,"complete_dataset","chips_positive") 
        os.makedirs(self.complete_dataset_chips_dir, exist_ok=True) #directory to hold xml files
       
        #Move annotations
        if original:
            annotations_path = self.chips_xml_dir
            annotations = os.listdir(annotations_path)  
            
        if not original:
            annotations_path = self.chips_positive_corrected_xml_dir
            annotations = os.listdir(annotations_path)
        
        for a in annotations:
            #copy annotations 
            shutil.copy(os.path.join(annotations_path, a), self.complete_dataset_xml_dir)
        
        #Move images
        images = os.listdir(self.chips_positive_dir)
        for i in images:
            #move images
            shutil.copy(os.path.join(self.chips_positive_dir, i), self.complete_dataset_chips_dir)
        
        #Move tiles 
        if include_tiles:
            self.complete_dataset_tile_dir = os.path.join(self.dcc_directory,"complete_dataset","tiles") 
            os.makedirs(self.complete_dataset_chips_dir, exist_ok=True) #directory to hold xml files
                    
            tiles = os.listdir(self.tiles_dir)
            for t in tiles:
                #move images
                shutil.copy(os.path.join(self.tiles_dir, t), self.complete_dataset_tile_dir)
        
        print(len(annotations),len(images))
        return len(annotations), len(images)
"""
Find file paths
"""
def list_of_sub_directories(path_to_images):
    """
    Define a function to create a list of the directories in the storage space containing images
    Find the subdirectories containing images
    """
    
    sub_directories = [] #initialize list
    
    for folder in os.listdir(path_to_images): #identifies the subfolders
        d = path_to_images + "/"+ folder #creates the complete path for each subfolder
        if os.path.isdir(d):
            sub_directories.append(d) #adds the subfolder to the list

    return sub_directories

def img_path_anno_path(sub_directories):
    """
    ### Define a function to create a list of the annotation and positive_chip paths for each of the subdirectories 
        "Create an array of the paths to the folders containing the images and annotation given a subdirectory"
    Only create paths for subdirectories that have these paths and for subdirectories that are correctly formated (Qianyu's thesis, etc.)
    """

    img_path = []
    anno_path = []
            
    for i in range(len(sub_directories)):
        if "chips" in os.listdir(sub_directories[i]):
            img_path.append(sub_directories[i] + "/" +  "chips_positive")
            anno_path.append(sub_directories[i] + "/" + "chips_positive_xml")
        elif "chips_positive" in os.listdir(sub_directories[i]):
            img_path.append(sub_directories[i] + "/" +  "chips_positive")
            anno_path.append(sub_directories[i] + "/" + "chips_positive_xml")
        else:
            for ii in range(len(os.listdir(sub_directories[i]))):
                img_path.append(sub_directories[i] + "/" + os.listdir(sub_directories[i])[ii] + "/" + "chips_positive")
                anno_path.append(sub_directories[i] + "/" + os.listdir(sub_directories[i])[ii] + "/" + "chips_positive_xml")
    
    img_annotation_path = np.empty((1,2)) #form a numpy array
    for i in range(len(img_path)):
        if os.path.isdir(img_path[i]) == True:
            img_annotation_path = np.vstack((img_annotation_path,
                                             [img_path[i], anno_path[i]]))
    img_annotation_path = np.delete(img_annotation_path, 0, axis=0) #0 removes empty row
    return img_annotation_path



"""
Check and track annotations 
"""
def check_for_missing_images_annotations(img_annotation_path):
    """
    Check if images are missing annotations
    """
    
    for i in range(len(img_annotation_path)): #index over each folder
        path = Path(img_annotation_path[i,0]) #get root path
        parent_dir = path.parent.absolute()
        
        img_files = os.listdir(img_annotation_path[i,0]) #pull the files in the img folder
        anno_files = os.listdir(img_annotation_path[i,1]) #pull the files in the annotation folder
        
        anno_files_no_ext = []
        for annotation in anno_files: #iterate through the annotations
            anno_files_no_ext.append(os.path.splitext(annotation)[0])
            
        img_files_no_ext = []
        for image in img_files: #iterate through the images
            if image.endswith(".jpg"):
                img_files_no_ext.append(os.path.splitext(image)[0])
        
        for image in img_files_no_ext:
            if image not in anno_files_no_ext:
                print(parent_dir)
                print(image)
                   
        for anno in anno_files_no_ext:
            if anno not in img_files_no_ext:
                print(parent_dir)
                print(anno)

"""
Tracking and Verification 
"""
def reference_image_annotation_file_with_annotator(img_annotation_path, 
                                                   tracker_file_path = 'outputs/tile_img_annotation_annotator.npy'):
    """
    Track image annotations
    """
    if os.path.isfile(tracker_file_path): #check if the tracking file exists
        print("Initializing annotation tracking array; add new annotations to tracking array")
        tile_img_annotation_annotator = np.load(tracker_file_path) #load existing 
    else:
        print("Create new tracking array")
        tile_img_annotation_annotator = np.empty((1,8)) #form a numpy array

    for i in range(len(img_annotation_path)): #index over each folder
        print(img_annotation_path[i,0])
        #img files + image_file_pathways
        img_files = [] #pull the files in the img folder
        img_file_pathways = [] #pull the files in the img folder
        for image in os.listdir(img_annotation_path[i,0]): #iterate through the images
            if image.endswith(".jpg"):
                img_files.append(image)
                img_file_pathways.append(os.path.join(img_annotation_path[i,0]))
        
        #sort so that the paths/file names match
        img_file_pathways = sorted(img_file_pathways)
        img_files = sorted(img_files)
        num_img_files = len(img_files)
              
        #tiles
        tiles = [] # create a list of the tile names
        for image in img_files: #iterate through the images
            tiles.append(image.rsplit("_",1)[0])

        #annotation files
        anno_files = sorted(os.listdir(img_annotation_path[i,1])) #pull the files in the annotation folder

        #annotator
        path = Path(img_annotation_path[i,0]).parent.absolute() #get root path of chips postive/chips postive xml folder
        annotator = str(path).rsplit('\\')[-2] #get the annotator name from the root path
        annotator_list = [annotator] * len(anno_files)
                
        #annotator - verify coverage 
        annotator_verify_coverage = [""] * num_img_files

        #annotator - verify coverage 
        annotator_verify_quality = [""] * num_img_files

        #annotator - verify coverage 
        annotator_verify_classes = [""] * num_img_files
        
        tile_img_annotation_annotator = np.vstack((tile_img_annotation_annotator, 
                                                   np.column_stack([tiles, img_files, img_file_pathways, anno_files,annotator_list,
                                                                    annotator_verify_coverage, annotator_verify_quality,
                                                                    annotator_verify_classes]) ))  
    
    if not os.path.isfile(tracker_file_path): #if the file does not exist; remove the initalizing dummy array
        tile_img_annotation_annotator = np.delete(tile_img_annotation_annotator, 0, axis=0) #0 removes empty row
        
    return tile_img_annotation_annotator

def update_path(path, tracker_file_path):
    """
    If the verfification has not yet been completed, update the image/xml path
    """
    img_annotation_path = img_path_anno_path(list_of_sub_directories(path))
    
    #get the correct img files + image_file_pathways
    img_files = [] #pull the files in the img folder
    img_file_pathways = [] #pull the files in the img folder
    for i in range(len(img_annotation_path)): #index over each folder         
        for image in os.listdir(img_annotation_path[i,0]): #iterate through the images
            if image.endswith(".jpg"):
                img_file_pathways.append(os.path.join(img_annotation_path[i,0].rsplit("/",1)[0]))
                img_files.append(image)
    imgs_and_pathways = np.array(list(zip(img_file_pathways, img_files)))

    #replace incorrect pathways
    tile_img_annotation_annotator = np.load(tracker_file_path)
    for i in range(len(tile_img_annotation_annotator)): #i - index for tracker .npy
        for ii in np.where(imgs_and_pathways[:,1] == tile_img_annotation_annotator[i,1])[0]: #find the same images, (ii -index for img and pathway array)
            if imgs_and_pathways[ii,0] != tile_img_annotation_annotator[i,2]:
                tile_img_annotation_annotator[i,2] = imgs_and_pathways[ii,0]

    np.save('outputs/tile_img_annotation_annotator.npy', tile_img_annotation_annotator)
    column_names = ["tile_name", "chip_name", "chip pathway", "xml annotation", 
                    "annotator - draw","annotator - verify coverage",
                    "annotator - verify quality", "annotator - verify classes"]
    tile_img_annotation_annotator_df = pd.DataFrame(data = tile_img_annotation_annotator, 
                                                   index = tile_img_annotation_annotator[:,1], 
                                                   columns = column_names)
    tile_img_annotation_annotator_df.to_csv('outputs/tile_img_annotation_annotator_df.csv')
    return tile_img_annotation_annotator 

def verification_folders(home_directory, folder_name, annotator_allocation, set_number):
    """
    Create folder for workers to verify images
    Args:
    """
    #create verification folder 
    verification_dir = os.path.join(home_directory,'verification_set'+set_number) 
    os.makedirs(verification_dir, exist_ok=True) 

    #pair folder name with annotors 
    print(folder_name[0])
    ##create verification subfolder for each group
    os.makedirs(os.path.join(verification_dir, "verify_" + folder_name[0]+ "_" + set_number), exist_ok = True) #verification folder for each group
    os.makedirs(os.path.join(verification_dir, "verify_" + folder_name[0]+ "_" + set_number, "chips_positive"), exist_ok = True) #image sub folder             
    os.makedirs(os.path.join(verification_dir, "verify_" + folder_name[0]+ "_" + set_number, "chips_positive_xml"), exist_ok = True) #xml sub folder
    folder_annotator_list = [folder_name[0], annotator_allocation]
    return(folder_annotator_list, verification_dir)

def seperate_images_for_verification_update_tracking(folder_annotator_list, verification_dir, set_number, tile_img_annotation_annotator):
    """
    Move images to verifcation folder
    """        
    print("folder",folder_annotator_list[0]) #the current folder
    count = 0
    for i in range(len(folder_annotator_list[1])): #iterate over annotator
        print("annotator",folder_annotator_list[1][i]) #the current annotator
        for ii in np.where(tile_img_annotation_annotator[:, 4] == folder_annotator_list[1][i])[0]:
                if len(tile_img_annotation_annotator[ii,5]) == 0:
                    tile_img_annotation_annotator[ii,5] = folder_annotator_list[0].split("_")[0].capitalize()#coverage
                    tile_img_annotation_annotator[ii,6] = folder_annotator_list[0].split("_")[1].capitalize()#quality
                    tile_img_annotation_annotator[ii,7] = folder_annotator_list[0].split("_")[2].capitalize()#class

                    shutil.copy(os.path.join(tile_img_annotation_annotator[ii, 2],"chips_positive", tile_img_annotation_annotator[ii, 1]), 
                                os.path.join(verification_dir, "verify_" + folder_annotator_list[0] + "_" + set_number, "chips_positive")) #copy images

                    shutil.copy(os.path.join(tile_img_annotation_annotator[ii, 2], "chips_positive_xml",
                                             tile_img_annotation_annotator[ii, 3]), 
                                os.path.join(verification_dir, "verify_" + folder_annotator_list[0] + "_" + set_number, "chips_positive_xml")) #copy annotations

                    count += 1 #count the files allocated to each 
        print(count)
    return tile_img_annotation_annotator


## Old Functions

def verification_folders_specify_in_function(home_directory, set_number):
    """
    Create folder for workers to verify images 
    """
    verification_dir = os.path.join(home_directory,'verification_set2') #create verification folder 
    os.makedirs(verification_dir, exist_ok=True) 

    folder_names = ['josh_james_amadu',
                    'jaewon_james_josh',
                    'jaewon_james_amadu',
                    'josh_jaewon_amadu']
    annotator_allocation = [['Jaewon','Jamila','Jonathan','Mia','Faiz','Alex'],
                            ['Amadu','Aidan', 'Sunny'],
                            ['Josh', 'Jackson'],
                            ['James','Qianyu', 'Connor', 'Celine', 'group_unreviewed_unverfied_images']]

    folder_annotator_list = []


    for i in range(len(folder_names)):    
        print(folder_names[i])
        #create verification subfolder for each group
        os.makedirs(os.path.join(verification_dir, "verify_" + folder_names[i]+ "_" + set_number), exist_ok = True) #verification folder for each group
        os.makedirs(os.path.join(verification_dir, "verify_" + folder_names[i]+ "_" + set_number, "chips"), exist_ok = True) #image sub folder                 
        os.makedirs(os.path.join(verification_dir, "verify_" + folder_names[i]+ "_" + set_number, "chips_xml"), exist_ok = True) #xml sub folder

        folder_annotator_list.append([folder_names[i],annotator_allocation[i]])

    return(folder_annotator_list, verification_dir)

def seperate_images_for_verification_update_tracking_specify_in_function(folder_annotator_list, verification_dir, set_number, tile_img_annotation_annotator):
    """
    Move images to verifcation folder
    """
    for i in range(len(folder_annotator_list)): #iterate over folders
        print("folder",folder_annotator_list[i][0]) #the current folder
        count = 0
        for ii in range(len(folder_annotator_list[i][1])): #iterate over annotator
            #print("annotator",folder_annotator_list[i][1][ii]) #the current annotator
            for iii in np.where(tile_img_annotation_annotator[:, 4] == folder_annotator_list[i][1][ii])[0]:

                if len(tile_img_annotation_annotator[iii,5]) == 0:
                    tile_img_annotation_annotator[iii,5] = folder_annotator_list[i][0].split("_")[0].capitalize()#coverage
                    tile_img_annotation_annotator[iii,6] = folder_annotator_list[i][0].split("_")[1].capitalize()#quality
                    tile_img_annotation_annotator[iii,7] = folder_annotator_list[i][0].split("_")[2].capitalize()#class
                    
                    shutil.copy(tile_img_annotation_annotator[iii, 2], 
                                os.path.join(verification_dir, "verify_" + folder_annotator_list[i][0] + "_" + set_number, "chips")) #copy images

                    shutil.copy(os.path.join(tile_img_annotation_annotator[iii, 2].rsplit("/",1)[0],"chips_positive_xml",
                                             tile_img_annotation_annotator[iii, 3]), 
                                os.path.join(verification_dir, "verify_" + folder_annotator_list[i][0] + "_" + set_number, "chips_xml")) #copy annotations

                    count += 1 #count the files allocated to each 
        print(count)
    return tile_img_annotation_annotator

"""
Review Characteristics 
"""
def summary_of_dataset(img_path, anno_path):
    ### Define function to count the number of objects in each category

    """Get summary of the whole dataset

    Args: 
        img_path (str): The path of the folder containing original images
        anno_path (str): The path of the folder containing original annotation files

    Returns: 
        summary_table (pandas df): A dataframe summary table of the number of objects in each class
        unknown_object_name (array): An array of the labels ascribes to objects that are not counted in the other existing categories 
        number_of_images (int): the number of images in the summary table
    """
    #Define lists
    print(anno_path)
    img_list = glob(img_path + '/*.jpg') #os.listdir(img_path) 
    anno_list =  os.listdir(anno_path)
    
    #calculate the number of images
    number_of_images = len(img_list)
    
    #Initial variables to count the number of objects in each category (set to zero)
    all_objects_count = 0 #all objects
    closed_roof_tank_count = 0 #closed_roof_tank
    narrow_closed_roof_tank_count = 0 #narrow_closed_roof_tank
    external_floating_roof_tank_count = 0 #external_floating_roof_tank
    spherical_tank_count = 0 #spherical_tank
    sedimentation_tank_count = 0 #water_treatment_tank
    water_tower_count = 0 #water_tower
    undefined_object_count = 0 #undefined_object

    #Create an list to save unknown object names
    unknown_object_name = []
    #"enumerate each image" This chunk is actually just getting the paths for the images and annotations
    for i in range(len(img_list)):
        img_file = img_list[i]
        anno_file = anno_list[i]
        
    #read .xml file
        dom_tree = xml.dom.minidom.parse(anno_path + "/" + anno_file)
        annotation = dom_tree.documentElement
        file_name_list = annotation.getElementsByTagName('filename') #[<DOM Element: filename at 0x381f788>]
        file_name = file_name_list[0].childNodes[0].data
        object_list = annotation.getElementsByTagName('object')

        for objects in object_list:
            # print objects
            all_objects_count += 1
            namelist = objects.getElementsByTagName('name')
            object_name = namelist[0].childNodes[0].data
            if object_name == "closed_roof_tank":
                closed_roof_tank_count += 1
            elif object_name == "narrow_closed_roof_tank":
                narrow_closed_roof_tank_count += 1
            elif object_name == "external_floating_roof_tank":
                external_floating_roof_tank_count += 1
            elif object_name == "spherical_tank": 
                spherical_tank_count += 1
            elif object_name == "sedimentation_tank":
                sedimentation_tank_count += 1
            elif object_name == "water_tower":
                water_tower_count += 1
            elif object_name == "undefined_object":
                undefined_object_count += 1
            else:
                unknown_object_name.append(object_name)
                
    summary_table = pd.DataFrame({"categories":["all_objects_count","closed_roof_tank_count", "narrow_closed_roof_tank_count", 
                                                "external_floating_roof_tank_count", "spherical_tank_count", "sedimentation_tank_count", 
                                                "water_tower_count", "undefined_object"],
                                 "values": [all_objects_count, closed_roof_tank_count, narrow_closed_roof_tank_count, external_floating_roof_tank_count,
                                            spherical_tank_count, sedimentation_tank_count, water_tower_count, undefined_object_count]})
    summary_table.set_index('categories', inplace = True)
    unknown_object_name = np.unique(unknown_object_name)
    return summary_table, unknown_object_name, number_of_images

def dataset_summary_assessment(img_annotation_path, multiple = True):
    """
    #### Iterate over the list of paths an create summary tables for each of the folders
    """
    summary_table = pd.DataFrame({"categories":["all_objects_count","closed_roof_tank_count", "narrow_closed_roof_tank_count", 
                                                "external_floating_roof_tank_count", "spherical_tank_count", "sedimentation_tank_count", 
                                                "water_tower_count", "undefined_object"],
                                     "values": [0]*8})
    summary_table.set_index('categories', inplace = True)

    unknown_object = []
    number_of_images = 0
    if multiple == True:
        for i in range(len(img_annotation_path)):
            summary_table_temp, unknown_object_array_temp, number_of_images_temp = summary_of_dataset(img_annotation_path[i,0],
                                                                                                      img_annotation_path[i,1])
            summary_table = summary_table.add(summary_table_temp)
            unknown_object.append(unknown_object_array_temp)
            number_of_images += number_of_images_temp
    if multiple == False:
        summary_table_temp, unknown_object_array_temp, number_of_images_temp = summary_of_dataset(img_annotation_path[0],
                                                                                                             img_annotation_path[1])
        summary_table = summary_table.add(summary_table_temp)
        unknown_object.append(unknown_object_array_temp)
        number_of_images += number_of_images_temp
    summary_table.to_csv('outputs/summary_table.csv')
    print("Array unknown objects", unknown_object)
    print("The number of clipped images included in the assessment", number_of_images)

def tile_progress(tiles_completed, tiles_remaining):
    print("tiles done")
    tiles_completed = np.load(tiles_completed)
    print(tiles_completed.shape)


    print("tiles left to be done")
    tile_remaining = np.load(tiles_remaining)
    print(tile_remaining.shape)

    print("Percent completed")
    print(tiles_completed.shape[0]/(tile_remaining.shape[0] + tiles_completed.shape[0]))
    
    
#### Convert VOC to YOLO
def get_classes(class_text_file):
    with open(class_text_file, "r") as class_text_file:
        lines = class_text_file.readlines()
        classes = []

        for l in lines:
            classes.append(l.replace("\n", ""))
    return classes
def get_images_in_dir(dir_path):
    image_list = []
    for filename in glob(dir_path + '/*.jpg'):
        image_list.append(filename)
    return image_list
def get_annotations_in_dir(dir_path):
    anno_list = []
    for filename in glob(dir_path + '/*.txt'):
        anno_list.append(filename)
    return anno_list

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(dir_path, xml_path, output_path, image_path, classes):
    basename = os.path.basename(image_path)
    basename_no_ext = os.path.splitext(basename)[0]

    in_file = open(xml_path + '/' + basename_no_ext + '.xml')
    out_file = open(output_path + '/' + basename_no_ext + '.txt', 'w')
    tree = et.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

## data augmentation
def convert_yolo_to_cv_format(yolo_annotation, H, W):
    f = open(yolo_annotation, 'r')
    data = f.readlines()
    bboxes = []
    for d in data:
        c, x, y, w, h = map(float, d.split(' ')) #bounding box, class, x,y, cords, width, height
        #l = int((x - w / 2) * W)
        #r = int((x + w / 2) * W)
        #t = int((y - h / 2) * H)
        #b = int((y + h / 2) * H)
        l = ((x - w / 2) * W)
        r = ((x + w / 2) * W)
        t = ((y - h / 2) * H)
        b = ((y + h / 2) * H)
        if l < 0:
            l = 0
        if r > W - 1:
            r = W - 1
        if t < 0:
            t = 0
        if b > H - 1:
            b = H - 1
        bboxes.append([l,t,r,b,c])
    bboxes = np.array(bboxes)
    return(bboxes)

def convert_cv_to_yolo_format(directory, output_filename, bboxes, H, W):
    #Convert from opencv format to yolo format
    # H,W is the image height and width
    bboxes_str = []
    for i in range(len(bboxes)):
        bbox_W = round(((bboxes[i,2] - bboxes[i,0]) / W), 10)
        bbox_H = round(((bboxes[i,3] - bboxes[i,1]) / H), 10)
        center_bbox_x = round(((bboxes[i,0] + bboxes[i,2]) / (2 * W)), 10)
        center_bbox_y = round(((bboxes[i,1] + bboxes[i,3]) / (2 * H)), 10)
        bbox = [int(bboxes[i,4]),
                center_bbox_x, center_bbox_y,
                bbox_W, bbox_H, " \n"]
        bbox = ' '.join(map(str,bbox))
        bboxes_str.append(bbox)
    print(bboxes_str)
    f = open(directory + output_filename +'.txt', 'w')
    f.writelines(bboxes_str)
    f.close() #to change file access modes
    
#In progress
def identify_incorrectly_formated_files(img_annotation_path):
    
    for i in range(len(img_annotation_path)): #index over each folder
        path = Path(img_annotation_path[i,0])
        parent_dir = path.parent.absolute()

        img_files = os.listdir(img_annotation_path[i,0]) #pull the files in the img folder
        anno_files = os.listdir(img_annotation_path[i,1]) #pull the files in the annotation folder

        incorrectly_formatted_files = []  #intialize list to store location of incorrectly formated files

        for ii in range(len(anno_files)): #index over each anno files 
            
            if anno_files[ii].endswith(".xml") != True: #create a list of the incorrectly formated files
                incorrectly_formatted_files.append(anno_files[ii]) 
          
        print(len(incorrectly_formatted_files))

        if len(incorrectly_formatted_files) > 0:
            os.mkdir(os.path.join(parent_dir,"chips_incorrect_format"))
            os.mkdir(os.path.join(parent_dir, "chips_xml_incorrect_format"))

            x = 0
            
            """         
            chips_incorrect_format = os.mkdir(path.join(parent_dir,"chips_incorrect_format"))
            chips_xml_incorrect_format = os.mkdir(os.path.join(parent_dir, "chips_xml_incorrect_format"))

            for anno_file in incorrectly_formatted_files: #get the file names in the annotation folder (without the extension)
                anno_filename = os.path.splitext(anno_file)[0]
                # copy images having same names with .xml files to "chips_tank" folder

                for img_file in img_files:
                    img_filename = os.path.splitext(img_file)[0]
                    
                    if anno_filename == img_filename:  
                        x += 1
                        #shutil.move(os.path.join(img_annotation_path[i,0], img_file), os.path.join(parent_dir, "chips_incorrect_format"))
                        #shutil.move(os.path.join(img_annotation_path[i,1], anno_file), os.path.join(parent_dir, "chips_xml_incorrect_format"))
            """
            print(x)
       
    
    
    
    
# Random Functions
def resize_256x256_to_515x512():
    #Edit annotations 
    #Annotation resize
    #read original annotations and write new xml files. 

    """Resize annotation.

    Args: 
        ImgPath (str): The path of the folder containing original images
        Annopath (str): The path of the folder containing original annotation files
        ProcessedPath (str): The path of folder to save new annotation files
        imagelist (list): a list of original images
        image_pre (str): The file name of the image
        ext (str): The extension name of the image
        imgfile (str): The path of the image
        xmlfile (str): The path of the xml file of the image
        DomTree, annotation, filenamelist, filename, objectlist are nodes in the xml file
        xmins, xmaxs, ymins, ymaxs (int): locations of bounding box of tanks
        names (str): label names
        num (int): number of labels
        filename_fill, filename_jpg (str): image name
        dealpath (str): path to save new xml file
        imagpath (str): path of the image


    Returns: 
        create new xml files
    """


    ImgPath = 'C:/Users/vcm/Desktop/NAIP/512_tank/' 
    AnnoPath = 'C:/Users/vcm/Desktop/NAIP/512_xml/'
    ProcessedPath = 'C:/Users/vcm/Desktop/NAIP/512_xml_rename/'
    os.makedirs(ProcessedPath, exist_ok=True)


    imagelist = os.listdir(ImgPath)

    for image in imagelist:
        print('a new image:', image)
        image_pre, ext = os.path.splitext(image)
        imgfile = ImgPath + image 
        xmlfile = AnnoPath + image_pre + '.xml'

        # Read original xml file content
        DomTree = xml.dom.minidom.parse(xmlfile)
        annotation = DomTree.documentElement

        filenamelist = annotation.getElementsByTagName('filename') #[<DOM Element: filename at 0x381f788>]
        filename = filenamelist[0].childNodes[0].data
        objectlist = annotation.getElementsByTagName('object')

        count = 0
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        names = []
        for objects in objectlist:
            # print objects
            count = count + 1
            namelist = objects.getElementsByTagName('name')
            # print 'namelist:',namelist


        # change label name
            objectname = namelist[0].childNodes[0].data
            if objectname == "closed roof tank" or objectname == "silo":
                names.append("closed_roof_tank")    
            elif objectname == "external floating roof tank":
                names.append("external_floating_roof_tank")  
            elif objectname == "sphere" or objectname == "spherical tank":
                names.append("spherical_tank")
            elif objectname == "water_treatment_facility" or objectname == "water treatment tank":
                names.append("water_treatment_tank") 
            else:
                names.append(objectname)

        # write locations of bounding boxes
            bndbox = objects.getElementsByTagName('bndbox')
            cropboxes = []

            for box in bndbox:
                try:
                    x1_list = box.getElementsByTagName('xmin')
                    x1 = int(x1_list[0].childNodes[0].data)
                    y1_list = box.getElementsByTagName('ymin')
                    y1 = int(y1_list[0].childNodes[0].data)
                    x2_list = box.getElementsByTagName('xmax')
                    x2 = int(x2_list[0].childNodes[0].data)
                    y2_list = box.getElementsByTagName('ymax')
                    y2 = int(y2_list[0].childNodes[0].data)

                    x1_1 = x1
                    y1_1 = y1
                    x2_1 = x2
                    y2_1 = y2

                    img = Image.open(imgfile)
                    width,height = img.size

                    xmins.append(x1_1)
                    ymins.append(y1_1)
                    xmaxs.append(x2_1)
                    ymaxs.append(y2_1)

                except Exception as e:
                    print(e)
        num = count
        print(num)
        print(names)
        filename_fill = image_pre
        filename_jpg = filename_fill + ".jpg"
        dealpath=ProcessedPath+ filename_fill +".xml"
        with open(dealpath, 'w') as f:
            height, width = (256, 256)
            writexml(dealpath,filename_jpg,num,xmins,ymins,xmaxs,ymaxs,names, height, width)
            
def positive_images(path): #found in random script, compare the two functions to see if this one is needed
    """Save positive images' names.
    Args: 
        path (string): The path to where positive images are stored.
        image_list (list): A list of images' names without extensions
        image_name (list): A list of images' names with extensions

    Returns: 
        Create a .npy file in the path where you store this code.

    """
    image_list = []
    image_name = os.listdir(path)
    for image in image_name:
        filename=os.path.splitext(image)[0]
        image_list.append(filename)
    np.save("positive_image_list", image_list)
    return

#old functions
def summary_of_dataset_inconsistent_data(img_path, anno_path):
    ### Define function to count the number of objects in each category

    """Get summary of the whole dataset

    Args: 
        img_path (str): The path of the folder containing original images
        anno_path (str): The path of the folder containing original annotation files

    Returns: 
        summary_table (pandas df): A dataframe summary table of the number of objects in each class
        unknown_object_name (array): An array of the labels ascribes to objects that are not counted in the other existing categories 
        number_of_images (int): the number of images in the summary table
    """
    #Define lists
    print(img_path)
    img_list = glob(img_path + '/*.jpg') #os.listdir(img_path) 
    anno_list =  os.listdir(anno_path)
    
    #calculate the number of images
    number_of_images = len(img_list)
    
    #Initial variables to count the number of objects in each category (set to zero)
    all_objects_count = 0 #all objects
    closed_roof_tank_count = 0 #closed_roof_tank
    narrow_closed_roof_tank_count = 0 #narrow_closed_roof_tank
    water_tower_count = 0 #water_tower
    external_floating_roof_tank_count = 0 #external_floating_roof_tank
    spherical_tank_count = 0 #spherical_tank
    water_treatment_tank_count = 0 #water_treatment_tank
    water_tower_count = 0 #water_tower
    undefined_object_count = 0 #undefined_object
    
    #Create a list of the possible names that each category may take 
    closed_roof_tank_label_list = ["closed_roof_tank", "'closed roof tank'"]
    narrow_closed_roof_tank_label_list = ["narrow_closed_roof_tank"]
    external_floating_roof_tank_label_list = ["external_floating_roof_tank", 'external floating roof tank']
    spherical_tank_label_list = ["spherical_tank", 'sphere', 'spherical tank']
    water_treatment_tank_label_list = ["water_treatment_tank", 'water_treatment_plant', 'water_treatment_facility']
    water_tower_label_list = ["water_tower"]
    undefined_object_label_list = ["undefined_object", 'silo']

    #Create an list to save unknown object names
    unknown_object_name = []
    #"enumerate each image" This chunk is actually just getting the paths for the images and annotations
    for i in range(len(img_list)):
        img_file = img_list[i]
        #print(img_file)
        anno_file = anno_list[i]
        
    #read .xml file
        dom_tree = xml.dom.minidom.parse(anno_path + "/" + anno_file)
        annotation = dom_tree.documentElement
        file_name_list = annotation.getElementsByTagName('filename') #[<DOM Element: filename at 0x381f788>]
        file_name = file_name_list[0].childNodes[0].data
        object_list = annotation.getElementsByTagName('object')

        for objects in object_list:
            # print objects
            all_objects_count += 1
            namelist = objects.getElementsByTagName('name')
            object_name = namelist[0].childNodes[0].data
            if object_name in closed_roof_tank_label_list:
                closed_roof_tank_count += 1
            elif object_name in narrow_closed_roof_tank_label_list:
                narrow_closed_roof_tank_count += 1
            elif object_name in external_floating_roof_tank_label_list:
                external_floating_roof_tank_count += 1
            elif object_name in spherical_tank_label_list: 
                spherical_tank_count += 1
            elif object_name in water_treatment_tank_label_list:
                water_treatment_tank_count += 1
            elif object_name in water_tower_label_list:
                water_tower_count += 1
            elif object_name in undefined_object_label_list:
                undefined_object_count += 1
            else:
                unknown_object_name.append(object_name)
                
    summary_table = pd.DataFrame({"categories":["all_objects_count","closed_roof_tank_count", "narrow_closed_roof_tank_count", 
                                                "external_floating_roof_tank_count", "spherical_tank_count", "water_treatment_tank_count", 
                                                "water_tower_count", "undefined_object"],
                                 "values": [all_objects_count, closed_roof_tank_count, narrow_closed_roof_tank_count, external_floating_roof_tank_count,
                                            spherical_tank_count, water_treatment_tank_count, water_tower_count, undefined_object_count]})
    summary_table.set_index('categories', inplace = True)
    unknown_object_name = np.unique(unknown_object_name)
    return summary_table, unknown_object_name, number_of_images

def img_path_anno_path_inconsistent_data(sub_directories):
    """
    ### Define a function to create a list of the annotation and positive_chip paths for each of the subdirectories 
        "Create an array of the paths to the folders containing the images and annotation given a subdirectory"
    Only create paths for subdirectories that have these paths and for subdirectories that are correctly formated (Qianyu's thesis, etc.)
    """

    img_path = []
    anno_path = []
                
    for i in range(len(sub_directories)):
        if "chips" in os.listdir(sub_directories[i]):
            img_path.append(sub_directories[i] + "/" +  "chips_positive")
            anno_path.append(sub_directories[i] + "/" + "chips_positive_xml")

        else:
            for ii in range(len(os.listdir(sub_directories[i]))):
                print(sub_directories[i],"/", os.listdir(sub_directories[i])[ii])
                img_path.append(sub_directories[i] + "/" + os.listdir(sub_directories[i])[ii] + "/" + "chips_positive")
                anno_path.append(sub_directories[i] + "/" + os.listdir(sub_directories[i])[ii] + "/" + "chips_positive_xml")
    
    img_annotation_path = np.empty((1,2)) #form a numpy array
    for i in range(len(img_path)):
        if os.path.isdir(img_path[i]) == True:
            img_annotation_path = np.vstack((img_annotation_path,
                                             [img_path[i], anno_path[i]]))
    img_annotation_path = np.delete(img_annotation_path, 0, axis=0) #0 removes empty row
    return img_annotation_path
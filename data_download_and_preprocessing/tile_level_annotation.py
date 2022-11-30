"""
Create pandas csv of image characteristics  
"""

"""
Import Packages
"""
import os
import argparse
import cv2
import math
from glob import glob
# Standard packages
import tempfile
import warnings
import urllib
import shutil
import pickle
# import requests
from PIL import Image
from io import BytesIO
import tqdm
from tqdm.notebook import tqdm_notebook
from skimage.metrics import structural_similarity as compare_ssim
import random
import numpy as np
import fiona  # must be import before geopandas
import geopandas as gpd
import rasterio
import rioxarray
import re
import rtree
import pyproj
import shapely
from shapely.geometry import Polygon, Point
from shapely.ops import transform
# from cartopy import crs
import collections
# Less standard, but still pip- or conda-installable
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# Parsing/Modifying XML
from lxml.etree import Element, SubElement, tostring
import xml.dom.minidom
from xml.dom.minidom import parseString
import xml.etree.ElementTree as et
from xml.dom import minidom

import data_eng.az_proc as ap
import data_eng.form_calcs as fc


def get_args_parse():
    parser = argparse.ArgumentParser(
        description='This script adds a subdirectory of xmls to correct possible inconsistent labels')
    parser.add_argument('--parent_dir', type=str, default=None,
                        help='path to parent directory, holding the img/annotation sub directories.')
    parser.add_argument('--annotation_dir', type=str, default="chips_positive_corrected_xml",
                        help="name of folder in complete dataset directory that contains annotations")
    parser.add_argument('--tile_dir', type=str, default=None,
                        help='path to directory holding tiles.')
    parser.add_argument('--tile_level_annotation_dir', type=str, default=None,
                        help='path to directory which holds tile level annotation and related files.')
    parser.add_argument('--tile_level_annotation_dataset_filename', type=str, default="tile_level_annotation",
                        help='File name of tile level annotation')
    parser.add_argument('--item_dim', type=int, default=int(512),
                        help='Dimensions of image (assumed sq)')
    parser.add_argument('--distance_limit', type=int, default=int(5),
                        help='The maximum pixel distance between bbox adjacent images, to merge')
    parser.add_argument('--state_gpd_path', type=str, default=None,
                        help='Path to dataset of state boundaries')
    parser.add_argument('--xml_folder_name', type=str, default="chips_positive_corrected_xml",
                        help="name of folder in complete dataset directory that contains annotations")
    args = parser.parse_args()
    return args


def main(args):

    # Generate table of characteristics for tiles/images
    # change where they are written
    tile_characteristics, image_characteristics = fc.image_tile_characteristics(args.parent_dir, args.tile_dir,
                                                                                args.xml_folder_name)

    # Generate tile level XMLs
    tiles_xml_dir = os.path.join(args.tile_level_annotation_dir, "tiles_xml")
    os.makedirs(tiles_xml_dir, exist_ok=True)
    fc.generate_tile_xmls(args.parent_dir, args.tile_dir, tiles_xml_dir, args.item_dim)

    # Merge neighboring bounding boxes within each tile
    # References:
    # https: // answers.opencv.org / question / 231263 / merging - nearby - rectanglesedited /
    # https: // stackoverflow.com / questions / 55593506 / merge - the - bounding - boxes - near - by - into - one
    tile_database = fc.merge_tile_annotations(tile_characteristics, tiles_xml_dir, distance_limit=args.distance_limit)

    # Add in state
    tile_database = fc.identify_state_name_for_each_state(args.state_gpd_path, tile_database)
    # check issues in state list
    #print(len(state_list[state_list==None]))
    # state_list = state_list[state_list==None]
    # np.unique(state_list)

    # Save tile dabasebase
    fc.write_gdf(tile_database, args.tile_level_annotation_dir, args.tile_level_annotation_dataset_filename)


if __name__ == '__main__':
    args = get_args_parse()
    main(args)

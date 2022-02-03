#Make into module, function specify annotator, directory as args
#URG: updae tile_undone_npy

#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import os
import os.path
import urllib.request
import progressbar # pip install progressbar2, not progressbar

import os
import shutil

import argparse

import tempfile
import urllib
import shutil
import os
import os.path
import sys

import PIL
from PIL import Image

import math
import numpy as np
import pandas as pd
import rtree
import pickle

import progressbar # pip install progressbar2, not progressbar

from geopy.geocoders import Nominatim

from contextlib import suppress

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

import warnings
from zipfile import ZipFile

import data_eng.az_proc as ap



if __name__ == '__main__':
    annotation_directory, parent_directory, number_of_tiles, tiles_remaining, tiles_labeled = main()
    
    dist = ap.annotator(annotation_directory) #create the processing class
    dist.state_dcc_directory(parent_directory)
    
    dist.number_of_tiles(number_of_tiles)
    
    dist.get_tile_urls(tiles_remaining)

    dist.make_subdirectories()
    dist.download_images()
    dist.tile_rename()
    dist.chip_tiles()
    
    dist.track_tile_annotations(tiles_labeled)
    np.save('tile_name_tile_url_remaining_expanded', dist.tile_name_tile_url_remaining)
    np.save('tile_name_tile_url_labeled', dist.tile_name_tile_url_labeled)
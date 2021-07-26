"""
Data Summary of Labeled Images (To - Date)
"""

"""
Import Packages
"""
import argparse

import numpy as np
import pandas as pd
from pathlib import Path
import os
from glob import glob
import data_eng.az_proc as ap

def main():
    parser = argparse.ArgumentParser(
        description='This script tracks which annotator drew the original annotations')
    parser.add_argument('--parent_directory', type=str, default=None,
                        help='path to parent directory, holding the annotation directories.')
    args = parser.parse_args()
    return args.parent_directory

    
if __name__ == '__main__':
    parent_directory= main()  
    img_anno = ap.img_path_anno_path(ap.list_of_sub_directories(parent_directory))
    print(img_anno)
    #load existing and update
    tile_img_annotation_annotator = ap.reference_image_annotation_file_with_annotator(img_anno, args.tracker_file_path)
    column_names = ["tile_name", "chip_name", "chip pathway", "xml annotation", 
                    "annotator - draw","annotator - verify coverage",
                    "annotator - verify quality", "annotator - verify classes"]
    np.save('outputs/tile_img_annotation_annotator.npy', tile_img_annotation_annotator)
    tile_img_annotation_annotator_df = pd.DataFrame(data = tile_img_annotation_annotator, 
                                                   index = tile_img_annotation_annotator[:,1], 
                                                   columns = column_names)
    tile_img_annotation_annotator_df.to_csv('outputs/tile_img_annotation_annotator_df.csv')

    
        


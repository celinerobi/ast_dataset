"""
Correct inconsistent labels and reclassify
"""

"""
Import Packages
"""
import shutil
import xml.etree.ElementTree as et
import argparse
import tqdm
import os
import sys
from PIL import Image
import numpy as np
from glob import glob
import data_eng.form_calcs as fc

def get_args_parse():
    parser = argparse.ArgumentParser(
        description='This script adds a subdirectory of corrected xmls to correct possible inconsistent labels and '
                    'reclassify tanks based on tanks size')
    parser.add_argument('--parent_dir', type=str, default=None,
                        help='path to parent directory, holding the img/annotation sub directories.')
    parser.add_argument('--orig_xml_folder_name', type=str, default="chips_positive_xml",
                        help="name of folder in complete dataset directory that contains annotations")
    parser.add_argument('--corrected_xml_folder_name', type=str, default="chips_positive_corrected_xml",
                        help="name of folder in complete dataset directory that contains annotations")
    args = parser.parse_args()
    return args

def main(args):
    tile_names = os.listdir(args.parent_dir)
    # Correct inconsistent labfolders_of_images_xmls_by_tileels
    for tile_name in tqdm.tqdm(tile_names):  # iterate over tile folders
        # get original and corrected xml directories
        orig_xml_dir = os.path.join(args.parent_dir, tile_name, args.orig_xml_folder_name)
        corrected_xml_dir = os.path.join(args.parent_dir, tile_name, args.corrected_xml_folder_name)

        for xml in os.listdir(orig_xml_dir):
            # get original and corrected xml paths
            orig_xml_path = os.path.join(orig_xml_dir, xml)
            corrected_xml_path = os.path.join(corrected_xml_dir, xml)
            # reformat xmls to add resolution/date/filename/path
            fc.reformat_xml_for_compiled_dataset(orig_xml_path)
            # correct inconsistent labels
            fc.correct_inconsistent_labels_xml(orig_xml_path, corrected_xml_path)
            # reclassify narrow/closed roof tanks based on size
            fc.reclassify_narrow_closed_roof_and_closed_roof_tanks(corrected_xml_path)

if __name__ == '__main__':
    ### Get the arguments 
    args = get_args_parse()
    main(args)


def reformat_xmls_for_rechipped_images(xml_directory, image_in_tile, correct_xml_name,
                                       correct_jpg_name, chips_positive_xml_dir_path):
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
    # load xml
    formatted_chip_name_wo_ext = os.path.splitext(os.path.basename(image_in_tile))[0]
    tree = et.parse(os.path.join(xml_directory, formatted_chip_name_wo_ext + ".xml"))
    root = tree.getroot()

    # add resolution to xml
    resolution = et.Element("resolution")
    resolution.text = formatted_chip_name_wo_ext.split("_")[1]  # resolution
    et.indent(tree, space="\t", level=0)
    root.insert(3, resolution)

    # add year to xml
    year = et.Element("year")
    year.text = formatted_chip_name_wo_ext.split("_")[2]  # year
    et.indent(tree, space="\t", level=0)
    root.insert(4, year)

    # correct spacing for source (dataset name)
    et.indent(tree, space="\t", level=0)

    # correct filename and path to formatting with row/col coordinates
    for filename in root.iter('filename'):
        filename.text = correct_xml_name
    for path in root.iter('path'):
        path.text = os.path.join(xml_directory, correct_jpg_name)

    tree.write(os.path.join(chips_positive_xml_dir_path, correct_xml_name))
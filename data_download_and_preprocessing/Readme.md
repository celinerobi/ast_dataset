# NAIP Data Download and Processing 
> This repository contains the necessary python files to 1) download and process NAIP tiles; 2) chip tiles in 512 x 512 images while conserving resolution, 3) allocate images to annotators; 4) seperate postive and negatives; 5) standardize object labels; 6) allocate images for verfication; 7) compile a data summary. 

## Table of Contents
* [General Info](#general-information)
* [Overview](#technologies-used)
    * [0. Tile Identification](#tile-identification )
    * [1. Data Download](#data-download)
        * [Naming Convention](#naming-convention)
            * [Tile Naming Convention](#tile-naming-convention)
            * [Chip Naming Convention](#chip-naming-convention)
    * [2. Seperate Positive and Negative Images](#seperate-positive-and-negative-images)
    * [3. Record Annotator](#record-annotator)
    * [4. Verification and Tracking](#verification-and-tracking)
    * [5. Create Complete Dataset](#create-complete-dataset)
    * [6. Standardize Object Labels](#standardize-object-labels)
    * [7. Identify Missing Annotations](#identify-missing-annotations)
    * [8. Data Summary](#data-summary)
## General Information
- Provide general information about your project here.
- What problem does it (intend to) solve?
- What is the purpose of your project?
- Why did you undertake it?
- Data Annotation: 
    - The Annotation tool is in labelImg.zip.
    - Data annotations are created using the LabelImg graphical image annotation tool. The tool has been developed in Python using Qt as its graphical interface. The annotations were created as XML files in the PASCAL VOC format, the format used by ImageNet. For flexibility, a script has been included to convert the annotations to COCO format.

The installation infromation and guide can be found at https://github.com/tzutalin/labelImg. 
## 0. Tile Identification 
NAIP data is accessed using the Microsoft Planetary Computer STAC API. To download tiles of interest, the file pathway have been identified and collected using the EIA, HFID, and other datasources. The naip_pathways.ipynb jupyter notebook processess these datasources and creates/saves a numpy array of the tile names and tile urls.
## 1. Data Download
Labelwork_distribution_dcc.py distributes tiles to annotators.

python cred/AST_dataset/data_download_and_preprocessing/labelwork_distribution_dcc.py --number_of_tiles number
                                            --annotation_directory annotator_name
                                            --parent_directory \dir_containing_chips_and_annotations
                                            --tiles_remaining path_to_numpy_array
                                            --tiles_labeled path_to_numpy_array
                                            
Example:
python labelwork_distribution_dcc.py --number_of_tiles 8 --annotation_directory Kang_10 --parent_directory \\oit-nas-fe13dc.oit.duke.edu\\data_commons-borsuk\\labelwork --tiles_remaining tile_name_tile_url_remaining_expanded.npy --tiles_labeled tile_name_tile_url_labeled.npy

python labelwork_distribution_dcc.py --number_of_tiles 30 --annotation_directory Poonacha_10 --parent_directory C:\chip_allocation --tiles_remaining tile_name_tile_url_remaining_expanded.npy --tiles_labeled tile_name_tile_url_labeled.npy

### Naming Convention:
#### Tile Naming Convention:
All tiles' names include information: state_resolution_year_quadrangle_filename
**State:**
Two-letter state code (eg. NC means North Carolina). 
**Resolution:**
String specification of image resolution, which has varied throughout NAIP’s history. Depending on year and state, this may be “050cm”, “060cm”, or “100cm”. (“060cm” in this dataset).
**Year:**
Four-digit year. Images are collected in each state every 3-5 years, with any given year containing some (but not all) states. For example, Alabama has data in 2011 and 2013, but not in 2012, while California has data in 2012, but not 2011 or 2013. Esri provides information about NAIP coverage in their interactive NAIP annual coverage map. In this dataset, tiles are acquired in 2018 or 2019. 
**Quadrangle:** 
USGS quadrangle identifier, specifying a 7.5 minute x 7.5 minute area.
The filename component of the path (m_3008601_ne_16_1_20150804 in this example) is preserved from USDA’s original archive to allow consistent referencing across different copies of NAIP. Minor variation in file naming exists, but filenames are generally formatted as: 
m_[quadrangle]_[quarter-quad]_[utm zone]_[resolution]_[capture date].tif
For example, the above file is in USGS quadrangle 30086, in the NE quarter-quad, which is in UTM zone 16, with 1m resolution, and was first captured on 8/4/2015. 
#### Chip Naming Convention:
All chips' names include information: state_resolution_year_quadrangle_filename_index
The two npy files record the tiles have been labeled (tile_name_tile_url_labeled) and have not been labeled(tile_name_tile_url_remaining).
Each tile 
**Index**
The index of a 512x512 chips which is clipped from a tile. 
**Data clipping**
After renaming, then copy 3_labelwork_tile_clip_dcc.py to the same path as 1_labelwork_distribution_dcc.py
Use 3_labelwork_tile_clip_dcc.py to clip the tiles into chips that will be stored in folder "chips".
## 2. Seperate Positive and Negative Images
After the annotators have annotated images, the positive (the images containing objects) and negative (the images that do not contain objects) must be placed in seperate folders.
This can be completed by running the following script in the command line.

cred/AST_dataset/data_download_and_preprocessing/seperate_positive_negative_images.py

python cred\AST_dataset\data_download_and_preprocessing\seperate_positive_negative_images.py  --annotation_directory annotator_name
                                             --parent_directory \dir_containing_chips_and_annotations
                                             
Example:
python seperate_positive_negative_images.py  --annotation_directory unverified_images_not_reviewed_by_student6_Feinberg --parent_directory \\oit-nas-fe13dc.oit.duke.edu\\data_commons-borsuk\\labelwork

python seperate_positive_negative_images.py  --annotation_directory unverified_images_not_reviewed_by_student10_Poonacha --parent_directory C:\chip_allocation

## 3. Record Annotator
After the annotators have reviewed their images to fix any small errors, the organizer relocates their images into the *Unverified* folder. This folder is organized by annotator, by annotation set. To record which annotations have been recorded by which annotator in a centralized location, the following script is run. This produces two outputs, a npy array and a csv which indicate the tile, chip, xml, and annotator. 

python track_annotator_draw.py  --parent_directory \dir_containing_all_chips_and_annotations
                                             
Example:
python track_annotator_draw.py --parent_directory C:\chip_allocation

python track_annotator_draw.py --parent_directory \\oit-nas-fe13dc.oit.duke.edu\\data_commons-borsuk\\verification_set3\unverified_images\student_reviewed_unverified_images

## 4. Verification and Tracking
After the annotators have reviewed their images to fix any small errors, the organizer relocates their images into the *Unverified* folder. This folder is organized by annotator, by annotation set. To record which annotations have been recorded by which annotator in a centralized location, the following script is run. This produces two outputs, a npy array and a csv which indicate the tile, chip, xml, and annotator. 

python verification_and_tracking.py     --tracker_file_path path_to_tracker_numpy_array
                                        --set_number the_set_number
                                        --home_directory path_to_home_directory
  
Example:
python verification_and_tracking.py --tracker_file_path outputs/tile_img_annotation_annotator.npy --set_number 1 --home_directory D:/
## 5. Create Complete Dataset
python make_complete_dataset.py  --parent_directory \dir_containing_all_chips_and_annotations
                                             
Example:
python make_complete_dataset.py --parent_directory \\oit-nas-fe13dc.oit.duke.edu\\data_commons-borsuk\\verification_set1\unverified_images\student_reviewed_unverified_images
## 6. Standardize Object Labels 
python correct_incon_labels.py  --parent_directory \dir_containing_all_annotator_folders
                                             
Example:
python correct_incon_labels.py --parent_directory D:\Unverified_images\student_reviewed_unverified_images

python correct_incon_labels.py --complete_dataset_directory \\oit-nas-fe13dc.oit.duke.edu\\data_commons-borsuk\\complete_dataset
### 7. Identify Missing Annotations 
path_to_images = "D:/Unverified_images/student_reviewed_unverified_images"
sub_directories = ap.list_of_sub_directories(path_to_images)
img_anno = ap.img_path_anno_path(sub_directories)
#check_for_missing_images_annotations(img_anno)

## 8. Data Summary
Creates a table of the number of closed_roof_tanks, water_treatment_tank, spherical_tank, external_floating_roof_tank, water_tower for all of the images. 

python data_clean_descrip.py  --parent_directory \dir_containing_all_annotator_folders 
                              --tiles_remaining path_to_numpy_array
                              --tiles_labeled path_to_numpy_array
                                             
Example for complete dataset:
python data_clean_descrip.py --complete_dataset_directory \\oit-nas-fe13dc.oit.duke.edu\\data_commons-borsuk\\complete_dataset --annotation_directory chips_positive_corrected_xml --tiles_remaining tile_name_tile_url_remaining_expanded.npy --tiles_labeled tile_name_tile_url_labeled.npy 


    

def compare_images(t_2_chip, labeled_img, scores):  
    gray_t_2_chip = cv2.cvtColor(t_2_chip.astype(np.uint8), cv2.COLOR_BGR2GRAY) # make gray
    gray_labeled_image = cv2.cvtColor(labeled_img.astype(np.uint8), cv2.COLOR_BGR2GRAY) #image that has been chipped from tile
    
    score = compare_ssim(gray_t_2_chip, gray_labeled_image, win_size = 3) #set window size so that is works on the edge peices 
    scores.append(score)
    if score >= 0.95: #If the labeled image is correct
        #chip_name_incorrectly_chip_names[index]
        return(True, scores)
    else: #if it is incorrect
        ## move incorrectly named image if it one of the same name has not already been moved
        return(False,scores)
def copy_and_replace_images_xml(img_name, img_path, xml_path, copy_dir):                  
    ####    
    new_img_path = os.path.join(copy_dir, "chips_positive", img_name + ".jpg")
    shutil.copy(img_path, new_img_path)
        
    new_xml_path = os.path.join(copy_dir, "chips_positive_xml", img_name + ".xml")
    shutil.copy(xml_path, new_xml_path) #destination
    
def compare_imgs_wo_blk_pxls_state_yr_std_from_6_digit_xy_idxs(correct_img_wo_black_sq, correct_img_wo_black_sq_path, 
                                                                            compile_dir, state_year_six_digit_idx_list, 
                                                                            state_year_img_paths, state_year_xml_paths,
                                                                            yx_list, standard_img_paths, standard_xml_paths):
    #process correct img (wo black sq) info
    correct_img_name = os.path.splitext(os.path.basename(correct_img_wo_black_sq_path))[0] #get correct img name
    print(correct_img_name)
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
        scores = []
        for idx in state_idxs:
            #get verified img/xml path
            img_path = state_year_img_paths[idx]
            xml_path = state_year_xml_paths[idx]
            img = cv2.imread(img_path)
            img = img[0:row_dim, 0:col_dim]
            #if (np.sum(img) != 0) & (compare_images(correct_img_wo_black_sq, img)): #only move images if they are not all black and they match the correct image
            if (np.sum(img) != 0):
                match, scores = compare_images(correct_img_wo_black_sq, img, scores) #only move images if they are not all black and they match the correct image
                if match:
                    copy_and_replace_images_xml(standard_quad_img_name_wo_ext, img_path, xml_path, by_tile_dir) #use standard name and copy to compiled directory 
        if len(scores)>0:
            print(max(scores))

        #identify imgs/xmls that match the chip position (standard imgs)
        for idx in standard_idxs:
            img_path = standard_img_paths[idx]
            xml_path = standard_xml_paths[idx]
            img = cv2.imread(img_path)
            img = img[0:row_dim, 0:col_dim]
            #if (np.sum(img) != 0) & (compare_images(correct_img_wo_black_sq, img)):
                #copy_and_replace_images_xml(standard_quad_img_name_wo_ext, img_path, xml_path, by_tile_dir) #use standard name and copy to compiled directory
                #print("match", correct_img_path, img_path)
            if (np.sum(img) != 0):
                match, scores = compare_images(correct_img_wo_black_sq, img, scores) #only move images if they are not all black and they match the correct image
                if match:
                    copy_and_replace_images_xml(standard_quad_img_name_wo_ext, img_path, xml_path, by_tile_dir) #use standard name and copy to compiled directory       
        if len(scores)>0:
            print(max(scores))

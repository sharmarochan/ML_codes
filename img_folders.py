# -*- coding: utf-8 -*-
"""
Spyder Editor


"""

import os
import random
import glob
import shutil
def create_test_train_dir(old_base,new_base):
    folder_names=os.listdir(old_base)
    for i in ['train','test']:
        train_path=os.path.join(new_base,i)
        if not os.path.isdir(train_path):
            os.mkdir(train_path)
        for folder in folder_names:
            folder_path=os.path.join(new_base,i,folder)
            if not os.path.isdir(folder_path):
                os.mkdir(folder_path)
    return folder_names

def move_files(folder_names,new_base,old_base):
    file_names={}
    for folder in folder_names:
        dir_path=os.path.join(old_base,folder)
        os.chdir(dir_path)
        file_names[folder]=glob.glob("*.jpg")
        file_names[folder].extend(glob.glob("*.jpeg"))
        file_names[folder].extend(glob.glob("*.png"))
        file_names[folder].extend(glob.glob("*.bmp"))
        file_names[folder].extend(glob.glob("*.tiff"))
    for f in file_names:
        sample_size=int(0.70*len(file_names[f]))
        train_imgs=random.sample(file_names[f],sample_size)
        test_imgs=[x for x in file_names[f] if x not in train_imgs]
        for tr_im in train_imgs:
            dest=os.path.join(new_base,"train",f,tr_im)
            source=os.path.join(old_base,f,tr_im)
            shutil.copyfile(source,dest)
        print("Done copying train images for class {}".format(f))
        for te_im in test_imgs:
            dest=os.path.join(new_base,"test",f,te_im)
            source=os.path.join(old_base,f,te_im)
            shutil.copyfile(source,dest)
        print("Done copying test images for class {}".format(f))
    
old_base="D:\\data\\kaggle\\flowers-recognition\\flowers"
new_base= "D:\\data/kaggle\\flowers-recognition\\pytorch_data"       
folder_names=create_test_train_dir(old_base,new_base)    
move_files(folder_names,new_base,old_base)    
    
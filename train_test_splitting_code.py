# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 13:59:46 2022

@author: shivaramakrishna.kv
"""

#train-test-split

import os
import pandas as pd
import shutil
import sklearn
from sklearn import model_selection
from tqdm import tqdm

#%%

root_dir = r"D:\Gramener\NHA_PoC\image_quality_check_19_07\train"
class_list = ["good", "bad"]

'''
Create a k-fold cross-validation set, 
with kfold column.
'''
file_name_list = []
target_gt = []

aug_flag = True
if aug_flag:
    for class_ in class_list:
        file_name_list.extend(os.listdir(os.path.join(root_dir, class_+'_aug')))
        target_gt.extend(len(os.listdir(os.path.join(root_dir, class_+'_aug')))*[class_])
else:
    for class_ in class_list:
        file_name_list.extend(os.listdir(os.path.join(root_dir, class_)))
        target_gt.extend(len(os.listdir(os.path.join(root_dir, class_)))*[class_])
new_df = pd.DataFrame({'file_name': file_name_list, 
                       'target': target_gt})

new_df.to_csv(os.path.join(root_dir, 'train_aug.csv'), index=False)

# Training data is in a CSV file called train.csv
df = pd.read_csv(os.path.join(root_dir, 'train_aug.csv'))
 # we create a new column called kfold and fill it with -1
df["kfold"] = -1
# the next step is to randomize the rows of the data
df = df.sample(frac=1).reset_index(drop=True)
# initiate the kfold class from model_selection module
y = df.target.values
kf = model_selection.StratifiedKFold(n_splits=5)
# fill the new kfold column
for fold, (trn_, val_) in enumerate(kf.split(X=df, y=y)):
    df.loc[val_, 'kfold'] = fold
    # save the new csv with kfold column 
df.to_csv(os.path.join(root_dir, "train_folds_aug.csv"), index=False)


#%%
if not aug_flag:
    for index, row in tqdm(df.iterrows(), total=len(df), desc="processing images"):
        image_name = row["file_name"]
        class_ = row["target"]
        if row["kfold"] != 4:
            image_path = os.path.join(root_dir, class_, image_name)
            target_path = os.path.join(root_dir, "train", class_, image_name)
            shutil.copy(image_path, target_path)
        else:
            image_path = os.path.join(root_dir, class_, image_name)
            target_path = os.path.join(root_dir, "test", class_, image_name)
            shutil.copy(image_path, target_path)
        
        
    



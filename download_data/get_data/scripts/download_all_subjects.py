# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 13:00:49 2020

@author: ning

This script is to download all the data to your local computer
"""

import os

import pandas as pd
import seaborn as sns
import utils

sns.set_style('white')
sns.set_context('poster')

if __name__ == '__main__':
    # download the data if not
    dataframe_dir   = '../data'
    df              = pd.read_csv(os.path.join(dataframe_dir,'available_subjects.csv'))
    EEG_dir         = '../EEG'
    annotation_dir  = '../annotations'
    for f in [EEG_dir,annotation_dir]:
        if not os.path.exists(f):
            os.mkdir(f)
    
    for (sub,day),row in df.groupby(['sub','day']):
        
        url_eeg         = row['link'].values[0]
        url_vmrk        = row['link'].values[1]
        url_vhdr        = row['link'].values[2]
        url_annotation  = row['annotation_file_link'].values[0]
        
        if len(os.listdir(EEG_dir)) < 1:
            for url in [url_eeg,url_vmrk,url_vhdr]:
                utils.download_url(url,
                             os.path.join(EEG_dir,url.split('/')[-1],)
                             )
        utils.download_url(url_annotation,
                     os.path.join(annotation_dir,
                                  f'suj{sub}_day{day}_annotations.txt'))

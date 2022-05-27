'''
Created on Thu May 27 02:00:00 2022

@author: joakim
'''
import os
import utils

from tqdm import tqdm

import numpy  as np
import pandas as pd

if __name__ == "__main__":
    EEG_dir             = './EEG'
    annotation_dir      = './annotations'
    epochs_dir          = './epoch_data'
    
    if not os.path.exists(epochs_dir):
        os.mkdir(epochs_dir)

    info_for_all_subjects_dir = './get_data/data'
    df                        = pd.read_csv(os.path.join(info_for_all_subjects_dir, 'available_subjects.csv'))
    # iterate through all experiment participants' EEGs
    for (suj,day),df_sub in df.groupby(['sub','day']):

        if not os.path.exists(os.path.join(epochs_dir,f'suj{suj}day{day}')):
            os.mkdir(os.path.join(epochs_dir,f'suj{suj}day{day}'))

        raw_file_name = os.path.join(EEG_dir, 'suj11_l5nap_day1.vhdr')
        annotation_file_name = os.path.join(annotation_dir, 'suj11_day1_annotations.txt')

        # Downsize the sampling from 1000 to 128 to reduce file sizes while maintaining data integrity
        resample_size = 128

        # 11-16 Hz bandpass for spindle
        # 1-10 Hz bandpass for k-complex
        for target_event in ['spindle', 'kcomplex']:
            if target_event == 'spindle':
                l_freq_desired = 11
                h_freq_desired = 16
            elif target_event == 'kcomplex':
                l_freq_desired = 1
                h_freq_desired = 10

            # Use Ning's Filter_based_and_thresholding class in utils.py
            FBT = utils.Filter_based_and_thresholding()

            FBT.get_raw(raw_file_name)
            FBT.get_annotation(annotation_file_name)
            FBT.make_events_windows(time_steps = .25, window_size=1)
            events = FBT.event_dictionary[target_event].Onset.values * FBT.raw.info['sfreq']
            spind_events = FBT.label_segments(events, event_type=target_event)
            FBT.get_epochs(spind_events, resample=resample_size)

            # Prepare the information to save the epoch numpy arrays
            array_is = FBT.epochs.get_data()
            labels = FBT.epochs.events[:,-1]
            windows = FBT.windows
            for condition in [f'{target_event}', f'no_{target_event}']:
                if not os.path.exists(os.path.join(epochs_dir,
                                                   f'suj{suj}day{day}',
                                                   condition)):
                    os.mkdir(os.path.join(epochs_dir,
                                          f'suj{suj}day{day}',
                                          condition))
            for array, label, window in tqdm(zip(array_is, labels, windows)):
                if label == 1:
                    np.save(os.path.join(epochs_dir,
                                         f'suj{suj}day{day}',
                                         f'{target_event}',
                                         f'{window[0]}_{window[1]}.npy'),
                            array)
                else:
                    np.save(os.path.join(epochs_dir,
                                         f'suj{suj}day{day}',
                                         f'no_{target_event}',
                                         f'{window[0]}_{window[1]}.npy'),
                            array)                    

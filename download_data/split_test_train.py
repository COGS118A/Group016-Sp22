import os
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd

if __name__ == "__main__":
    EEG_dir                   = './EEG'
    annotations_dir           = './annotations'
    epochs_dir                = './epoch_data'

    if not os.path.exists(epochs_dir):
        raise NotADirectoryError('Run download_all_epochs.py first.')

    info_for_all_subjects_dir = './get_data/data'
    df                        = pd.read_csv(os.path.join(info_for_all_subjects_dir, 'available_subjects.csv'))

    comp = np.array([]).reshape(0, 128)
    spind = np.array([]).reshape(0, 128)
    neither = np.array([]).reshape(0, 128)
    data = np.array([]).reshape(0,128)
    for (suj,day),df_sub in df.groupby(['sub','day']):

        subday = glob.glob(os.path.join(epochs_dir, f'suj{suj}day{day}/*'))
        num_of_files = [len(os.listdir(event)) for event in subday if any(subday)]
        if num_of_files:
            sample_cap = min(num_of_files)
        for event in tqdm(subday):
            event_str = event.split('\\')[-1]
            print(suj, day, event_str)
            i = 0
            while i < sample_cap:
                f = os.listdir(event)
                if event_str == 'kcomplex':
                    comp = np.concatenate((comp, np.load(os.path.join(event,f[i]))), axis=0)
                elif event_str == 'spindle':
                    spind = np.concatenate((spind, np.load(os.path.join(event,f[i]))), axis=0)
                elif event_str == 'no_kcomplex' or event_str == 'no_spindle':
                    neither = np.concatenate((neither, np.load(os.path.join(event,f[i]))), axis=0)
                i += 1
                
    c = np.ones((comp.shape[0], 1))
    s = np.full((spind.shape[0], 1), -1)
    n = np.zeros((neither.shape[0], 1))

    data = np.concatenate((spind, comp, neither), axis=0)
    labels = np.concatenate((s, c, n), axis=0)

    np.save('./dataset_1-30.npy', data)
    np.save('./dataset_labels_1-30.npy', labels)

    # For train/test arrays, use something like:
    # X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=42)
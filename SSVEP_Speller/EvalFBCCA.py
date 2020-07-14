from sklearn.cross_decomposition import CCA
import scipy.io as sio
import scipy.signal as spsignal
from lib import utils
import numpy as np

from lib.utils import ITR

import os


# ================== data information ==============
target_file = os.path.join('data', 'Freq_Phase.mat')
target_data = sio.loadmat(target_file)
freqs = target_data['freqs'].ravel()

data_file = os.path.join('data', 'S{}.mat')
nb_subjects = 35
channels = ['PZ', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'O1', 'OZ', 'O2']
channel_indices = np.array([48, 54, 55, 56, 57, 58, 61, 62, 63])-1

fs = 250  # Hz
time_gaze_s = 1.25  # s
time_delay_s = 0.13  # s
time_cue_s = 0.5  # s
time_all_s = time_gaze_s + time_delay_s

gaze_length = round(time_gaze_s * fs)
delay_length = round(time_delay_s * fs)
cue_length = round(time_cue_s * fs)
cue_length = round(time_cue_s * fs)

# ================== cca_reference ===================
nb_harms = 5
fb_a, fb_b = 1.25, 0.25   # for FBCCA
tx = np.arange(1, gaze_length+1, 1) / fs
y_ref = []
for freq in freqs:
    temp_ref = []
    for harm_i in range(nb_harms):
        temp_ref.append(np.sin(2*np.pi*tx*(harm_i+1)*freq))
        temp_ref.append(np.cos(2*np.pi*tx*(harm_i+1)*freq))
    y_ref.append(np.array(temp_ref))
y_ref = np.array(y_ref)

# ================== filtering and FBCCA ====================
filter_low_cutoff = 7.  # Hz
filter_high_cutoff = 90.  # Hz
b, a = spsignal.butter(4, [filter_low_cutoff/(fs/2.), filter_high_cutoff/(fs/2.)], 'bandpass')
cca = CCA(n_components=1, scale=False)

accs = []
itrs = []
selected_subjects = [3, 4, 12, 22, 25, 26, 32, 34]
for s in selected_subjects:
    print('subject: {}'.format(s))
    data_path = data_file.format(s)
    data = sio.loadmat(data_path)['data']
    data = np.transpose(data, axes=[3, 2, 0, 1])
    subject_accs = []
    subject_itrs = []
    for block_id in range(len(data)):
        eeg = data[block_id][:, channel_indices, cue_length+delay_length:cue_length+delay_length+gaze_length]
        eeg = spsignal.filtfilt(b, a, eeg, axis=-1)
        nb_correct = 0.
        for label, eeg_epoch in enumerate(eeg):
            rho_list = []
            eeg_epoch = eeg_epoch.T
            for ref in y_ref:
                ref = ref.T
                rho = 0
                for band_id in range(10):
                    band_eeg_epoch = utils.filterband(eeg_epoch, band_id, fs, axis=0)
                    x_, y_ = cca.fit_transform(band_eeg_epoch, ref)
                    band_rho = np.abs(np.matmul(x_.T, y_)/np.linalg.norm(x_, ord=2)/np.linalg.norm(y_, ord=2))
                    w = (band_id+1.)**(-fb_a)+fb_b
                    rho += w*band_rho**2
                rho_list.append(rho)
            if np.argmax(rho_list) == label:
                nb_correct += 1.

        acc = nb_correct / len(eeg)
        itr = ITR(len(freqs), acc, time_all_s)
        print('block:{}, acc:{}, itr:{}'.format(block_id, acc, itr))
        subject_accs.append(acc)
        subject_itrs.append(itr)
    accs.append(subject_accs)
    itrs.append(subject_itrs)
np.savez('results/fb_clean_eval.npz', acc=np.array(accs), itr=np.array(itrs))


from sklearn.cross_decomposition import CCA
import scipy.io as sio
from lib import utils
import scipy.signal as spsignal
import numpy as np

from lib.utils import ITR

import os

# ============= adversarial noise =================
adv_file = os.path.join('template', 'pert_S{}_25.npz')

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

# ================== cca_reference ===================
nb_harms = 5
fb_a, fb_b = 1.25, 0.25
tx = np.arange(1, gaze_length+1, 1) / fs
y_ref = []
for freq in freqs:
    temp_ref = []
    for harm_i in range(nb_harms):
        temp_ref.append(np.sin(2*np.pi*tx*(harm_i+1)*freq))
        temp_ref.append(np.cos(2*np.pi*tx*(harm_i+1)*freq))
    y_ref.append(np.array(temp_ref))
y_ref = np.array(y_ref)

# ================== filtering and CCA ====================
filter_low_cutoff = 7.  # Hz
filter_high_cutoff = 90.  # Hz
b, a = spsignal.butter(4, [filter_low_cutoff/(fs/2.), filter_high_cutoff/(fs/2.)], 'bandpass')
cca = CCA(n_components=1, scale=False, max_iter=1000)

selected_subjects = [3, 4, 12, 22, 25, 26, 32, 34]
for s in selected_subjects:
    data_path = data_file.format(s)
    data = sio.loadmat(data_path)['data']
    data = np.transpose(data, axes=[3, 2, 0, 1])
    adv_noises = np.load(adv_file.format(s))['template']
    user_accs_list = []
    user_itrs_list = []
    attacker_accs_list = []
    attacker_itrs_list = []
    attacker_sprs_list = []
    for target_id in range(len(freqs)):
    # for target_id in [0]:
        show_flag = True
        user_accs = []
        user_itrs = []
        attacker_accs = []
        attacker_itrs = []
        attacker_sprs = []
        for block_id in range(len(data)):
            eeg = data[block_id][:, channel_indices, cue_length+delay_length:cue_length+delay_length+gaze_length]
            attacker_correct = 0.
            user_correct = 0.
            mean_spr = 0.
            for label, eeg_epoch_original in enumerate(eeg):
                rho_list = []
                adv_eeg_epoch = eeg_epoch_original + adv_noises[target_id]
                clean_eeg_epoch = spsignal.filtfilt(b, a, eeg_epoch_original, axis=-1)
                adv_eeg_epoch = spsignal.filtfilt(b, a, adv_eeg_epoch, axis=-1)

                noise = adv_eeg_epoch - clean_eeg_epoch
                spr = 10 * np.log10(np.sum(np.square(clean_eeg_epoch)) / np.sum(np.square(noise)))
                mean_spr += spr
                adv_eeg_epoch = adv_eeg_epoch.T
                for ref in y_ref:
                    ref = ref.T
                    rho = 0
                    for band_id in range(10):
                        band_eeg_epoch = utils.filterband(adv_eeg_epoch, band_id, fs, axis=0)
                        x_, y_ = cca.fit_transform(band_eeg_epoch, ref)
                        band_rho = np.abs(np.matmul(x_.T, y_) / np.linalg.norm(x_, ord=2) / np.linalg.norm(y_, ord=2))
                        w = (band_id + 1.) ** (-fb_a) + fb_b
                        rho += w * band_rho ** 2
                    rho_list.append(rho)
                if np.argmax(rho_list) == target_id:
                    attacker_correct += 1.
                if np.argmax(rho_list) == label:
                    user_correct += 1.
            attacker_acc = attacker_correct / len(eeg)
            user_acc = user_correct / len(eeg)
            mean_spr = mean_spr / len(eeg)
            user_itr = ITR(len(freqs), user_acc, time_all_s)
            attacker_itr = ITR(len(freqs), attacker_acc, time_all_s)
            print('subject:{}, target:{}, block:{}, attacker_acc:{}, attacker_itr:{}, spr:{}'.format(s, target_id, block_id, attacker_acc, attacker_itr, mean_spr))
            user_accs.append(user_acc)
            user_itrs.append(user_itr)
            attacker_accs.append(attacker_acc)
            attacker_itrs.append(attacker_itr)
            attacker_sprs.append(mean_spr)
        user_accs_list.append(user_accs)
        user_itrs_list.append(user_itrs)
        attacker_accs_list.append(attacker_accs)
        attacker_itrs_list.append(attacker_itrs)
        attacker_sprs_list.append(attacker_sprs)
        print(np.array(attacker_accs_list).shape)
    np.savez('results/FB_attack_result_S{}_25.npz'.format(s),
             user_acc=np.array(user_accs_list),
             user_itr=np.array(user_itrs_list),
             attacker_acc=np.array(attacker_accs_list),
             attacker_itr=np.array(attacker_itrs_list),
             attacker_spr=np.array(attacker_sprs_list))
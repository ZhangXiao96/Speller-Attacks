from sklearn.cross_decomposition import CCA
import scipy.io as sio
import scipy.signal as spsignal
import numpy as np

from lib.TraceCCA import TraceCCA

import tensorflow as tf

import os


# attack parameters
epochs = 20000

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
cca = CCA(n_components=1, scale=False)

selected_subjects = [3, 4, 12, 22, 25, 26, 32, 34]
allowed_spr = [25, 25, 25, 25, 25, 25, 25, 25]

for s_id, s in enumerate(selected_subjects):
    data_path = data_file.format(s)
    data = sio.loadmat(data_path)['data']
    data = np.transpose(data, axes=[3, 2, 0, 1])
    eeg = data[0][:, channel_indices, cue_length+delay_length:cue_length+delay_length+gaze_length]
    eeg = spsignal.filtfilt(b, a, eeg, axis=-1)

    # ======== energy ==========
    energy = np.mean(np.sum(np.square(eeg), axis=(-1, -2)))
    # ==========================

    templates = []
    for target_id in range(len(freqs)):
        with tf.Session() as sess:
            temp_ref = y_ref[target_id][np.newaxis, :, :]
            ref = np.repeat(temp_ref, repeats=eeg.shape[0], axis=0)

            # generate adversarial examples
            universal_noise_root = tf.Variable(np.zeros(shape=(1, len(channels), gaze_length)), dtype=tf.float32)
            # ========================  filtering =========================
            eeg_fft = tf.signal.rfft(universal_noise_root)
            masks = np.ones(shape=(1, len(channels), int(round(gaze_length/2)+1)))
            masks[:, :, 0:11] = 0
            masks[:, :, 141:] = 0
            eeg_fft_filtered = masks * eeg_fft
            universal_noise = tf.signal.irfft(eeg_fft_filtered)
            # =============================================================
            x_pd = tf.placeholder(dtype=tf.float32, shape=(None, len(channels), gaze_length))
            y_pd = tf.placeholder(dtype=tf.float32, shape=(None, nb_harms*2, gaze_length))
            x_perturbed = x_pd + universal_noise
            tf_cca = TraceCCA(x_perturbed, y_pd)

            rho = tf.reduce_mean(tf_cca.rho)
            loss = -rho + 0.5 * tf.reduce_mean(tf.square(universal_noise))

            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
            train_op = optimizer.minimize(loss, var_list=[universal_noise_root])
            sess.run(tf.global_variables_initializer())
            for epoch_id in range(epochs):
                sess.run(train_op, feed_dict={x_pd: eeg, y_pd: ref})
                pert_ = sess.run(universal_noise)
                spr_ = -10*np.log10(np.sum(np.square(pert_))/energy)
                if spr_ < allowed_spr[s_id]+0.2:
                    break
                if epoch_id % 200 == 0:
                    print('subject:{}, target:{}, epoch:{}, spr:{}'.format(s, target_id, epoch_id, spr_))

            templates.append(np.squeeze(sess.run(universal_noise)))
        tf.reset_default_graph()
    np.savez('template/pert_S{}.npz'.format(s), template=templates)
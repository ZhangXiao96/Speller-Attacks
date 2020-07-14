import scipy.io as io
import numpy as np
import os

_CHAR_MATRIX = np.array(
            [list('abcdef'),
             list('ghijkl'),
             list('mnopqr'),
             list('stuvwx'),
             list('yz1234'),
             list('56789_')]
        )

data_path = os.path.join('Data', 'A0{}.mat')
save_path = os.path.join('Data', 'processed_data', '{}_{}.mat')
for subject in range(1, 9, 1):
    data = io.loadmat(data_path.format(subject))['data'][0][0]

    original_signal = data[1].astype(np.float64)
    original_stimulusType = data[2].astype(np.float64).ravel()
    original_stimulusCode = data[3].astype(np.float64).ravel()
    trialStart = data[4].ravel()

    signal = []
    stimulusType = []
    stimulusCode = []
    chars = []
    flash = []
    for i in trialStart:
        signal.append(original_signal[i:i+8500].transpose(-1, 0))
        trial_stimulusType = original_stimulusType[i:i+8500]
        trial_stimulusCode = original_stimulusCode[i:i+8500]

        trial_flash = np.concatenate([[0], trial_stimulusType[:-1]], axis=0)
        trial_flash = np.where(trial_stimulusType-trial_flash > 0, 1, 0)
        flash.append(trial_flash)
        stimulusType.append(trial_flash * (trial_stimulusType-1))
        stimulusCode.append(trial_flash * trial_stimulusCode)
        targets = np.unique(trial_flash * trial_stimulusCode * (trial_stimulusType-1))
        for target in targets:
            if 0 < target <= 6:
                column = int(target - 1)
            elif 6 < target <= 12:
                row = int(target - 7)
        chars.append(_CHAR_MATRIX[row, column])
    signal = np.array(signal)
    stimulusType = np.array(stimulusType)
    stimulusCode = np.array(stimulusCode)
    chars = np.array(chars)
    flash = np.array(flash)

    io.savemat(save_path.format(subject, 'train'),
               {'flashing': flash[:21], 'signal': signal[:21], 'stimuli': stimulusCode[:21],
                'label': stimulusType[:21], 'char': chars[:21]})

    io.savemat(save_path.format(subject, 'test'),
               {'flashing': flash[21:], 'signal': signal[21:], 'stimuli': stimulusCode[21:],
                'label': stimulusType[21:], 'char': chars[21:]})

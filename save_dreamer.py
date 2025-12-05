import scipy.io as sio
import pandas as pd
import numpy as np

path = "DREAMER.mat"
raw = sio.loadmat(path)
participant =0
video = 0
electrode = 0
B, S = [], []
basl_eeg = (raw["DREAMER"][0, 0]["Data"][0, participant]["EEG"][0, 0]["baseline"][0, 0][video, 0][:, electrode])
basl_ecg= (raw["DREAMER"][0, 0]["Data"][0, participant]["ECG"][0, 0]["baseline"][0, 0][video, 0][:, electrode])
stim_eeg = (raw["DREAMER"][0, 0]["Data"][0, participant]["EEG"][0, 0]["stimuli"][0, 0][0, 0][:, ])
stim_ecg = (raw["DREAMER"][0, 0]["Data"][0, participant]["ECG"][0, 0]["stimuli"][0, 0][0, 0][:, ])
print(basl_eeg.shape)
print(basl_ecg.shape)
print(stim_eeg.shape)
print(stim_ecg.shape)


def feat_extract_EEG_ECG(raw):
    EEG_tmp = np.zeros((23, 18, 14, 7680))
    ECG_tmp = np.zeros((23, 18, 2, 15360))

    for participant in range(0, 23):
        for video in range(0, 18):
            for electrode in range(0, 14):
                stim_eeg = raw["DREAMER"][0, 0]["Data"][0, participant]["EEG"][0, 0]["stimuli"][0, 0][video, 0][:,
                           electrode]
                EEG_tmp[participant, video, electrode, :] = stim_eeg[:7680]

            for channel in range(0, 2):
                stim_ecg = raw["DREAMER"][0, 0]["Data"][0, participant]["ECG"][0, 0]["stimuli"][0, 0][video, 0][:,
                           channel]
                ECG_tmp[participant, video, channel, :] = stim_ecg[:15360]

    eeg_cols = []
    for electrode in range(1, 15):
        for i in range(0, 7680):
            eeg_cols.append(f"EEG_Electrode{electrode}_{i + 1}")

    ecg_cols = []
    for channel in range(1, 3):
        for i in range(0, 15360):
            ecg_cols.append(f"ECG_Channel{channel}_{i + 1}")

    EEG_flat = EEG_tmp.reshape(23 * 18, -1)
    ECG_flat = ECG_tmp.reshape(23 * 18, -1)

    data_EEG = pd.DataFrame(EEG_flat, columns=eeg_cols)
    data_ECG = pd.DataFrame(ECG_flat, columns=ecg_cols)

    return data_EEG, data_ECG


def participant_affective(raw):
    a = np.zeros((23, 18, 4), dtype=object)
    for participant in range(0, 23):
        for video in range(0, 18):
            a[participant, video, 0] = ["calmness", "surprise", "amusement",
                                        "fear", "excitement", "disgust",
                                        "happiness", "anger", "sadness",
                                        "disgust", "calmness", "amusement",
                                        "happiness", "anger", "fear",
                                        "excitement", "sadness",
                                        "surprise"][video]
            a[participant, video, 1] = int(raw["DREAMER"][0, 0]["Data"]
                                           [0, participant]["ScoreValence"]
                                           [0, 0][video, 0])
            a[participant, video, 2] = int(raw["DREAMER"][0, 0]["Data"]
                                           [0, participant]["ScoreArousal"]
                                           [0, 0][video, 0])
            a[participant, video, 3] = int(raw["DREAMER"][0, 0]["Data"]
                                           [0, participant]["ScoreDominance"]
                                           [0, 0][video, 0])
    b = pd.DataFrame(a.reshape((23 * 18, a.shape[2])),
                     columns=["target_emotion", "valence", "arousal", "dominance"])
    return b


df_EEG, df_ECG = feat_extract_EEG_ECG(raw)
df_participant_affective = participant_affective(raw)

df = pd.concat([df_EEG, df_ECG, df_participant_affective], axis=1)

num_rows = 18
total_iterations = 23

for i in range(total_iterations):
    start_row = i * num_rows
    end_row = (i + 1) * num_rows

    eeg_data = []
    for electrode in range(1, 15):
        selected_columns = [col for col in df.columns if col.startswith(f'EEG_Electrode{electrode}_')]
        result_df = df[selected_columns].iloc[start_row:end_row]
        eeg_data.append(result_df.values)

    eeg_data = np.array(eeg_data)
    eeg_data = np.transpose(eeg_data, (1, 0, 2))

    ecg_data = []
    for channel in range(1, 3):
        selected_columns = [col for col in df.columns if col.startswith(f'ECG_Channel{channel}_')]
        result_df = df[selected_columns].iloc[start_row:end_row]
        ecg_data.append(result_df.values)

    ecg_data = np.array(ecg_data)
    ecg_data = np.transpose(ecg_data, (1, 0, 2))

    label_columns = [col for col in df.columns if col in df_participant_affective.columns and col != 'target_emotion']
    labels = df[label_columns].iloc[start_row:end_row].values

    combined_data = {
        'eeg_data': eeg_data,
        'ecg_data': ecg_data,
        'labels': labels
    }

    excel_filename = f'Your_Save_path/dreamer{str(i)}.npy'
    np.save(excel_filename, combined_data)
    print(f'The extracted data has been saved as {excel_filename}')
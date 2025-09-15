import os
import numpy as np
import pandas as pd
import neurokit2 as nk

####### Parameters
clean_signal = True
sampling_rate = 500
task_length_secs = 60  # seconds
num_samples = sampling_rate * task_length_secs

# sample index from where to extract [num_samples]
sample_seek = {
    "Baseline": 0 * sampling_rate,
    "Relax": 80 * sampling_rate,
    "Reading": 0 * sampling_rate,
    "Speaking": 0 * sampling_rate,
    "Math": 0 * sampling_rate,
    "Stroop": 0 * sampling_rate,
    "Counting1": 0 * sampling_rate,
    "Counting2": 0 * sampling_rate,
    "Counting3": 0 * sampling_rate,
    "Video1": 114 * sampling_rate,  # 1:54
    "Video2": 58 * sampling_rate,  # 0:58
    "Breathing": 110 * sampling_rate,
}

####### LOAD DATA
path_data = "../../../stressid-dataset/Physiological/"

filelist = [f for f_sub in [f_names for root, d_names, f_names in os.walk(path_data)] for f in f_sub]
dirlist = [d for d_sub in [d_names for root, d_names, f_names in os.walk(path_data)] for d in d_sub]

data_ecg = dict()
data_eda = dict()
data_rsp = dict()

filelist.sort()

for i in filelist:
    if not i.startswith("."):
        path = os.path.join(path_data, i.split("_")[0] + "/")
        file = pd.read_csv(path + i, sep=",")
        if file.isnull().sum().sum() != 0:
            print("There are ", file.isnull().sum().sum(), " nan values in the recording", i)

        file_ecg = np.array(file["ECG"])
        file_eda = np.array(file["EDA"])
        file_rsp = np.array(file["RR"])

        # Normalization
        data_ecg[i.split(".")[0]] = (file_ecg - file_ecg.mean()) / file_ecg.std()
        data_eda[i.split(".")[0]] = (file_eda - file_eda.mean()) / file_eda.std()
        data_rsp[i.split(".")[0]] = (file_rsp - file_rsp.mean()) / file_rsp.std()


del data_eda["r5s8_Counting3"]


####### CLEAN USING NK
ecg_clean = data_ecg.copy()
eda_clean = data_eda.copy()
rsp_clean = data_rsp.copy()

for ecg, eda, rsp in zip(data_ecg.items(), data_eda.items(), data_rsp.items()):
    task = ecg[0].split("_")[1]
    seek = sample_seek[task]
    ecg_clean[ecg[0]] = nk.ecg_clean(ecg[1], sampling_rate=500, method="biosppy").iloc[seek, num_samples]
    task = eda[0].split("_")[1]
    seek = sample_seek[task]
    eda_clean[eda[0]] = nk.eda_clean(eda[1], sampling_rate=500, method="biosppy").iloc[seek, num_samples]
    task = rsp[0].split("_")[1]
    seek = sample_seek[task]
    rsp_clean[rsp[0]] = nk.rsp_clean(rsp[1], sampling_rate=500, method="biosppy").iloc[seek, num_samples]

######## MERGE
df = pd.concat([ecg_clean, rsp_clean], axis=1).merge(eda_clean, left_index=True, right_index=True)

####### EXPORT
eda_clean.to_csv(f"{path_data}/eda_windowed.csv", sep=",", index=True)
rsp_clean.to_csv(f"{path_data}/rsp_windowed.csv", sep=",", index=True)
ecg_clean.to_csv(f"{path_data}/ecg_windowed.csv", sep=",", index=True)

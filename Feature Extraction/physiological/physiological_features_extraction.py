import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

import os
from pathlib import Path

import scipy.stats as stats
import neurokit2 as nk

from ecg_features import *
from eda_features import *
from respiration_features import *


####### LOAD DATA
path_data = "../../../stressid-dataset/Physiological/"

filelist = [f for f_sub in [f_names for root, d_names, f_names in os.walk(path_data)] for f in f_sub]
filelist.sort()

STD_BY_SUBJ = True  # Else, is Standarization(z-score) by Task (by sample)

subjects = {}
for f in filelist:
    if not f.startswith("."):
        subj = f[:4]
        filename = f.split(".")[0]
        subjects.setdefault(subj, []).append(filename)

data_ecg = dict()
data_eda = dict()
data_rsp = dict()

for i in filelist:
    if not i.startswith("."):
        path = os.path.join(path_data, i.split("_")[0] + "/")
        file = pd.read_csv(path + i, sep=",")
        if file.isnull().sum().sum() != 0:
            print("There are ", file.isnull().sum().sum(), " nan values in the recording", i)

        filename = i.split(".")[0]
        user_id = filename.split("_")[0]

        ecg = np.array(file["ECG"])
        eda = np.array(file["EDA"])
        rsp = np.array(file["RR"])

        data_ecg[filename] = ecg if STD_BY_SUBJ else (ecg - ecg.mean()) / ecg.std()
        data_eda[filename] = eda if STD_BY_SUBJ else (eda - eda.mean()) / eda.std()
        data_rsp[filename] = rsp if STD_BY_SUBJ else (rsp - rsp.mean()) / rsp.std()


del data_eda["r5s8_Counting3"]

if STD_BY_SUBJ:
    for subj, subj_files in subjects.items():
        print(f"Standarizing subject {subj}")

        all_ecg = []
        all_eda = []
        all_rsp = []

        for filename in subj_files:
            all_ecg.extend(data_ecg[filename])
            all_rsp.extend(data_rsp[filename])
            if filename == "r5s8_Counting3":
                continue
            all_eda.extend(data_eda[filename])

        np_ecg = np.array(all_ecg)
        np_eda = np.array(all_eda)
        np_rsp = np.array(all_rsp)

        subj_stats = {
            "EDA": {
                "mean": np_eda.mean(),
                "std": np_eda.std(),
            },
            "ECG": {
                "mean": np_ecg.mean(),
                "std": np_ecg.std(),
            },
            "RSP": {
                "mean": np_rsp.mean(),
                "std": np_rsp.std(),
            },
        }

        for filename in subj_files:
            data_ecg[filename] = (data_ecg[filename] - subj_stats["ECG"]["mean"]) / subj_stats["ECG"]["std"]
            data_rsp[filename] = (data_rsp[filename] - subj_stats["RSP"]["mean"]) / subj_stats["RSP"]["std"]
            if filename == "r5s8_Counting3":
                continue
            data_eda[filename] = (data_eda[filename] - subj_stats["EDA"]["mean"]) / subj_stats["EDA"]["std"]


####### CLEAN USING NK
ecg_clean = data_ecg.copy()
eda_clean = data_eda.copy()
rsp_clean = data_rsp.copy()

for ecg, eda, rsp in zip(data_ecg.items(), data_eda.items(), data_rsp.items()):
    ecg_clean[ecg[0]] = nk.ecg_clean(ecg[1], sampling_rate=500, method="biosppy")
    eda_clean[eda[0]] = nk.eda_clean(eda[1], sampling_rate=500, method="biosppy")
    # rsp_clean[rsp[0]] = nk.rsp_clean(rsp[1], sampling_rate=500, method="biosppy")


######## EXTRACT FEATURES BY MODALITY
df_eda_features = get_eda_features(eda_clean, 500)
print("EDA: {0:2d} trials and {1:2d} features".format(df_eda_features.shape[0], df_eda_features.shape[1]))

# df_rsp_features = get_resp_features(rsp_clean, 500)
# print("Respiration : {0:2d} trials and {1:2d} features".format(df_rsp_features.shape[0], df_rsp_features.shape[1]))

df_ecg_features = get_ecg_features(ecg_clean, 500)
print("ECG : {0:2d} trials and {1:2d} features".format(df_ecg_features.shape[0], df_ecg_features.shape[1]))


######## MERGE
df_features = df_ecg_features.merge(df_eda_features, left_index=True, right_index=True)


####### EXPORT
df_eda_features.to_csv("../../../Reprod-Features-StdSubject/eda_features.csv", sep=",", index=True)
# df_rsp_features.to_csv("../../../Reprod-Features-StdSubject/resp_features.csv", sep=",", index=True)
df_ecg_features.to_csv("../../../Reprod-Features-StdSubject/ecg_features.csv", sep=",", index=True)
df_features.to_csv("../../../Reprod-Features-StdSubject/ecg_eda_features.csv", sep=",", index=True)

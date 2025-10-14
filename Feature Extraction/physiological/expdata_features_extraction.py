# pyright: reportAny=false
import numpy as np
import pandas as pd
import glob

import neurokit2 as nk

from ecg_features import get_ecg_features
from eda_features import get_eda_features

TASKS = ["Baseline", "AmusementClip", "StressClip", "EmoReset", "FormL", "FormM", "Debriefing"]
STD_BY_SUBJ = True  # Else, is Standarization(z-score) by Task (by sample)
EXPECTED_NUM_FILES = 21
SAMP_RATE = 51.2
dataset_path = "../../../experiment-data"

type FeatureDict = dict[str, np.ndarray]

####### LOAD DATA
filelist = glob.glob(f"{dataset_path}/*.Annotated.csv")
if len(filelist) != EXPECTED_NUM_FILES:
    raise ValueError(f"Expected {EXPECTED_NUM_FILES} files, found: {len(filelist)}")

data_ppg: FeatureDict = dict()
data_eda: FeatureDict = dict()

filelist.sort()
subjects: dict[str, list[str]] = {}


def split_by_label(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    output: list[tuple[str, pd.DataFrame]] = []
    start_idx = end_idx = 0
    for label in TASKS:
        if label == "FormL":
            if "FormLRead" in df["Event"].values:
                start_idx = df.index.get_loc(df[df["Event"] == "FormLRead"].index[0])
                end_idx = df.index.get_loc(df[df["Event"] == "L15"].index[-1])
            else:
                start_idx = df.index.get_loc(df[df["Event"] == "FormL"].index[0])
                end_idx = df.index.get_loc(df[df["Event"] == "FormL"].index[-1])
        elif label == "FormM":
            if "FormMRead" in df["Event"].values:
                start_idx = df.index.get_loc(df[df["Event"] == "FormMRead"].index[0])
                end_idx = df.index.get_loc(df[df["Event"] == "M15"].index[-1])
            else:
                start_idx = df.index.get_loc(df[df["Event"] == "FormM"].index[0])
                end_idx = df.index.get_loc(df[df["Event"] == "FormM"].index[-1])
        else:
            start_idx = df.index.get_loc(df[df["Event"] == label].index[0])
            end_idx = df.index.get_loc(df[df["Event"] == label].index[-1])
        output.append((label, df[start_idx:end_idx]))
    return output


for item in filelist:
    file: pd.DataFrame = pd.read_csv(
        item,
        delimiter=";",
        parse_dates=["Datetime", "Timestamp"],
        index_col=[
            "Datetime",
        ],
        dtype={
            "Timestamp": float,
            "Event": str,
            "ExtraEvent": str,
            "AccelLN_X": float,
            "AccelLN_Y": float,
            "AccelLN_Z": float,
            "Battery": float,
            "GSR_Range": int,
            "Skin_Conductance": float,
            "Skin_Resistance": float,
            "Gyro_X": float,
            "Gyro_Y": float,
            "Gyro_Z": float,
            "PPG": float,
            "Pressure": float,
            "Temperature": float,
            "AccelLN_X_Uncal": int,
            "AccelLN_Y_Uncal": int,
            "AccelLN_Z_Uncal": int,
            "Skin_Conductance_Uncal": int,
            "PPG_Uncal": int,
        },
    )
    filename = item.split("/")[-1]
    participant_id = filename.split("-")[1]

    for label, event_df in split_by_label(file):
        sample_name = f"{participant_id}-{label}"
        subjects.setdefault(participant_id, []).append(sample_name)
        file_ppg = np.array(event_df["PPG_Uncal"])
        file_eda = np.array(event_df["Skin_Conductance_Uncal"])

        data_ppg[sample_name] = file_ppg if STD_BY_SUBJ else (file_ppg - file_ppg.mean()) / file_ppg.std()
        data_eda[sample_name] = file_eda if STD_BY_SUBJ else (file_eda - file_eda.mean()) / file_eda.std()


if STD_BY_SUBJ:
    for subj, subj_files in subjects.items():
        print(f"Standarizing subject {subj}")

        all_ppg = []
        all_eda = []

        for filename in subj_files:
            all_ppg.extend(data_ppg[filename])
            all_eda.extend(data_eda[filename])

        np_ppg = np.array(all_ppg)
        np_eda = np.array(all_eda)

        subj_stats = {
            "EDA": {
                "mean": np_eda.mean(),
                "std": np_eda.std(),
            },
            "PPG": {
                "mean": np_ppg.mean(),
                "std": np_ppg.std(),
            },
        }

        for filename in subj_files:
            data_ppg[filename] = (data_ppg[filename] - subj_stats["PPG"]["mean"]) / subj_stats["PPG"]["std"]
            data_eda[filename] = (data_eda[filename] - subj_stats["EDA"]["mean"]) / subj_stats["EDA"]["std"]

####### CLEAN USING NK
ppg_clean = data_ppg.copy()
eda_clean = data_eda.copy()

for ppg, eda in zip(data_ppg.items(), data_eda.items()):
    ppg_clean[ppg[0]] = nk.ppg_clean(ppg[1], sampling_rate=SAMP_RATE)
    eda_clean[eda[0]] = nk.eda_clean(eda[1], sampling_rate=SAMP_RATE, method="neurokit")


######## EXTRACT FEATURES BY MODALITY
df_eda_features = get_eda_features(eda_clean, SAMP_RATE)
print("EDA: {0:2d} trials and {1:2d} features".format(df_eda_features.shape[0], df_eda_features.shape[1]))

df_ppg_features = get_ecg_features(ppg_clean, SAMP_RATE)
print("PPG : {0:2d} trials and {1:2d} features".format(df_ppg_features.shape[0], df_ppg_features.shape[1]))


######## MERGE
df_features = pd.concat([df_ppg_features, df_eda_features], axis=1)


####### EXPORT
std_suffix = "stdsubj." if STD_BY_SUBJ else "stdtask."
df_eda_features.to_csv(f"{dataset_path}/extracted-features/eda_features.{std_suffix}csv", sep=";", index=True)
df_ppg_features.to_csv(f"{dataset_path}/extracted-features/ppg_features.{std_suffix}csv", sep=";", index=True)
df_features.to_csv(f"{dataset_path}/extracted-features/all_features.{std_suffix}csv", sep=";", index=True)

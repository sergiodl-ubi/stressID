# pyright: reportAny=false
import numpy as np
import pandas as pd
import glob

import neurokit2 as nk

from ecg_features import get_ecg_features
from eda_features import get_eda_features

EXPECTED_NUM_FILES = 21
dataset_path = "../../../experiment-data"

type FeatureDict = dict[str, np.ndarray]

####### LOAD DATA
filelist = glob.glob(f"{dataset_path}/*.Annotated.csv")
if len(filelist) != EXPECTED_NUM_FILES:
    raise ValueError(f"Expected {EXPECTED_NUM_FILES} files, found: {len(filelist)}")

data_ecg: FeatureDict = dict()
data_eda: FeatureDict = dict()

filelist.sort()

def split_by_label(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    output: list[tuple[str, pd.DataFrame]] = []
    labels = ["Baseline", "AmusementClip", "StressClip", "EmoReset", "FormL", "FormM", "Debriefing"]
    start_idx = end_idx = 0
    for label in labels:
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
        index_col=["Datetime",],
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
            "PPG_Uncal": int,}
        )
    filename = item.split("/")[-1]
    participant_id = filename.split("-")[1]

    for label, event_df in split_by_label(file):
        sample_name = f"{participant_id}-{label}"
        file_ecg = np.array(event_df['PPG_Uncal'])
        file_eda = np.array(event_df['Skin_Conductance_Uncal'])

        data_ecg[sample_name] = (file_ecg - file_ecg.mean())/file_ecg.std()
        data_eda[sample_name] = (file_eda - file_eda.mean())/file_eda.std()


####### CLEAN USING NK
ecg_clean = data_ecg.copy()
eda_clean = data_eda.copy()

for ecg,eda in zip(data_ecg.items(), data_eda.items()):
    ecg_clean[ecg[0]] = nk.ecg_clean(ecg[1], sampling_rate=51.2, method="neurokit")
    eda_clean[eda[0]] = nk.eda_clean(eda[1], sampling_rate=51.2, method='neurokit')


    
######## EXTRACT FEATURES BY MODALITY    
df_eda_features = get_eda_features(eda_clean, 51.2)
print('EDA: {0:2d} trials and {1:2d} features'.format(df_eda_features.shape[0], df_eda_features.shape[1]))

df_ecg_features = get_ecg_features(ecg_clean, 51.2)
print('ECG : {0:2d} trials and {1:2d} features'.format(df_ecg_features.shape[0], df_ecg_features.shape[1]))


######## MERGE
df_features = pd.concat([df_ecg_features, df_eda_features], axis=1)


####### EXPORT
df_eda_features.to_csv(f'{dataset_path}/extracted-features/eda_features.csv', sep=";", index=True)
df_ecg_features.to_csv(f'{dataset_path}/extracted-features/ecg_features.csv', sep=";", index=True)
df_features.to_csv(f'{dataset_path}/extracted-features/all_features.csv', sep=";", index=True)


        

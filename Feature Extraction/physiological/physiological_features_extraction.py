# pyright: reportAny=false
import numpy as np
import pandas as pd
import glob

import neurokit2 as nk

from ecg_features import get_ecg_features
from eda_features import get_eda_features

EXPECTED_NUM_FILES = 21
dataset_path = "../../../experiment-data"

####### LOAD DATA
filelist = glob.glob(f"{dataset_path}/*.Annotated.csv")
if len(filelist) != EXPECTED_NUM_FILES:
    raise ValueError(f"Expected {EXPECTED_NUM_FILES} files, found: {len(filelist)}")

data_ecg = dict()
data_eda = dict()

filelist.sort()

def split_by_label(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    output: list[tuple[str, pd.DataFrame]] = []
    labels = ["Baseline", "AmusementClip", "StressClip", "EmoReset", "FormL", "FormM", "Debriefing"]
    for label in labels:
        if label == "FormL":
            start_idx = df.index.get_loc(df[df["Event"] == "FormLRead"].index[0])
            end_idx = df.index.get_loc(df[df["Event"] == "L15"].index[-1])
        elif label == "FormM":
            start_idx = df.index.get_loc(df[df["Event"] == "FormMRead"].index[0])
            end_idx = df.index.get_loc(df[df["Event"] == "M15"].index[-1])
        else:
            start_idx = df.index.get_loc(df[df["Event"] == label].index[0])
            end_idx = df.index.get_loc(df[df["Event"] == label].index[-1])
        output.append((label, df[start_idx:end_idx]))
    return output

for item in filelist:
    file = pd.read_csv(
        item,
        delimiter=";",
        parse_dates=["Datetime",],
        index_col=["Datetime",])
    print(file.shape)
    participant_id = item.split("-")[1]
    if file.isnull().sum().sum() != 0:
        print('There are ', file.isnull().sum().sum(), ' nan values in the recording', item)

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
    ecg_clean[ecg[0]] = nk.ecg_clean(ecg[1], sampling_rate=51, method="biosppy")
    eda_clean[eda[0]] = nk.eda_clean(eda[1], sampling_rate=51, method='biosppy')

    
    
######## EXTRACT FEATURES BY MODALITY    
df_eda_features = get_eda_features(eda_clean, 51)
print('EDA: {0:2d} trials and {1:2d} features'.format(df_eda_features.shape[0], df_eda_features.shape[1]))

df_ecg_features = get_ecg_features(ecg_clean, 51)
print('ECG : {0:2d} trials and {1:2d} features'.format(df_ecg_features.shape[0], df_ecg_features.shape[1]))


######## MERGE
df_features = pd.concat([df_ecg_features, df_eda_features], axis=1)


####### EXPORT
df_eda_features.to_csv(f'{dataset_path}/extracted-features/eda_features.csv', sep=";", index=True)
df_ecg_features.to_csv(f'{dataset_path}/extracted-features/ecg_features.csv', sep=";", index=True)
df_features.to_csv(f'{dataset_path}/extracted-features/all_features.csv', sep=";", index=True)


        

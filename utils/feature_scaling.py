import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from joblib import dump


# min max scaling
def min_max_scale(df):
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    dump(scaler, "./models/output/scaler.joblib")
    return df_scaled


# robust scaling
def robust_scale(df):
    scaler = RobustScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    dump(scaler, "./models/output/scaler.joblib")
    return df_scaled


# standard scaling
def standard_scale(df):
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    dump(scaler, "./models/output/scaler.joblib")
    return df_scaled

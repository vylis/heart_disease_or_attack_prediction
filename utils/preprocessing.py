from .feature_engineering import (
    invert_wrong_binary_features,
    classify_bmi,
    classify_genhealth,
    classify_physical_health,
    classify_mental_health,
    risk_score_health_data,
)
from .feature_scaling import min_max_scale, robust_scale, standard_scale


def process_data(df, scaling_method="None"):
    # cleaning
    columns_to_drop = [
        "Education",
        "Income",
    ]
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

    # feature engineering
    df = invert_wrong_binary_features(df)
    df = classify_bmi(df)
    df = classify_genhealth(df)
    df = classify_physical_health(df)
    df = classify_mental_health(df)
    df = risk_score_health_data(df)

    # feature scaling
    if scaling_method == "min_max":
        df = min_max_scale(df)
    elif scaling_method == "robust":
        df = robust_scale(df)
    elif scaling_method == "standard":
        df = standard_scale(df)
    else:
        df = df
    return df

# select high correlation features
def select_high_corr_features_general(df, threshold):
    corr_matrix = df.corr()
    high_corr_features = (corr_matrix.abs() > threshold) & (corr_matrix < 1.0)
    selected_features = corr_matrix.loc[high_corr_features.any()].index.tolist()
    return selected_features


# select high correlation features based on the target
def select_high_corr_features_per_heart(df, threshold):
    corr_matrix = df.corr()
    high_corr_features = corr_matrix["HeartDiseaseorAttack"].abs() > threshold
    selected_features = corr_matrix.loc[
        high_corr_features, "HeartDiseaseorAttack"
    ].index.tolist()
    return selected_features

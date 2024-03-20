import pandas as pd
import numpy as np


# invert healthy lifestyle features because 1 is supposed to be high risk
def invert_wrong_binary_features(df):
    for feature in ["PhysActivity", "Fruits", "Veggies", "CholCheck", "AnyHeatlhcare"]:
        if feature in df.columns:
            df[feature] = 1 - df[feature]
    return df


# cut education into 4 categories
def classify_educa(df):
    bins = [0, 3, 4, 5, 6]
    labels = [0, 1, 2, 3]
    df["Education"] = pd.cut(df["Education"], bins, labels=labels)
    return df


# cut income into 5 categories
def classify_income(df):
    bins = [0, 2, 4, 5, 6, 8]
    labels = [0, 1, 2, 3, 4]
    df["Income"] = pd.cut(df["Income"], bins, labels=labels)
    return df


# cut bmi into 4 categories
def classify_bmi(df):
    bins = [0, 18.5, 25, 30, 100]
    # underweight, normal, overweight, obese
    labels = [1, 2, 3, 4]
    df["BMI"] = pd.cut(df["BMI"], bins, labels=labels)
    return df


# cut genhealth into 2 categories
def classify_genhealth(df):
    bins = [0, 3, 5]
    labels = [0, 1]
    df["GenHlth"] = pd.cut(df["GenHlth"], bins, labels=labels)
    return df


# cut physical health into 3 categories
def classify_physical_health(df):
    bins = [-np.inf, 0, 13, np.inf]
    labels = [0, 1, 2]
    df["PhysHlth"] = pd.cut(df["PhysHlth"], bins, labels=labels)
    return df


# cut mental health into 3 categories
def classify_mental_health(df):
    bins = [-np.inf, 0, 13, np.inf]
    labels = [0, 1, 2]
    df["MentHlth"] = pd.cut(df["MentHlth"], bins, labels=labels)
    return df


# risk score based health data
def risk_score_health_data(df):
    df["RiskScoreHealthData"] = (
        df["HighBP"] + df["HighChol"] + df["Stroke"] + df["Diabetes"]
    )
    return df


# risk score based on lifestyle data
def risk_score_lifestyle_data(df):
    df["RiskScoreLifestyleData"] = (
        df["Smoker"]
        + df["HvyAlcoholConsump"]
        + df["PhysActivity"]
        + df["Fruits"]
        + df["Veggies"]
    )
    return df


# bad health with physhlth and menthlth
def bad_health(df):
    df["badHlth"] = df["PhysHlth"] + df["MentHlth"]
    return df


# real age
def real_age(df):
    age_ranges = {
        1: (18, 24),
        2: (25, 29),
        3: (30, 34),
        4: (35, 39),
        5: (40, 44),
        6: (45, 49),
        7: (50, 54),
        8: (55, 59),
        9: (60, 64),
        10: (65, 69),
        11: (70, 74),
        12: (75, 79),
        13: (80, None),
    }
    df["Age"] = df["Age"].map(age_ranges)
    return df


# real income
def real_income(df):
    income_ranges = {
        1: 10000,
        2: 20000,
        3: 30000,
        4: 40000,
        5: 50000,
        6: 60000,
        7: 70000,
        8: 80000,
    }
    df["Income"] = df["Income"].map(income_ranges)
    return df


# income per sex
def income_per_sex(df):
    df["IncomePerSex"] = df.groupby("Sex")["Income"].transform("mean")
    return df


# income per age
def income_per_age(df):
    df["IncomePerAge"] = df.groupby("Age")["Income"].transform("mean")
    return df


# income per education
def income_per_education(df):
    df["IncomePerEducation"] = df.groupby("Education")["Income"].transform("mean")
    return df


# healthcare per income
def healthcare_per_income(df):
    df["HealthcarePerIncome"] = df.groupby("Income")["AnyHealthcare"].transform("mean")
    return df


# no doctor consultation bcs cost per income
def no_doctor_consultation_per_income(df):
    df["NoDocConsultationPerIncome"] = df.groupby("Income")["NoDocbcCost"].transform(
        "mean"
    )
    return df

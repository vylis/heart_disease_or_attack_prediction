from fastapi import FastAPI, HTTPException, Form, Depends

import pandas as pd
from joblib import load
import tensorflow as tf

from models.model import build_model
from utils.preprocessing import process_data
from utils.categorize_prediction import categorize_prediction

app = FastAPI(
    swagger_ui_parameters=[
        {"name": "validatorUrl", "value": None},
    ]
)

# load model and scaler
model = tf.keras.models.load_model("./models/output/model.keras")
# model = load("./models/output/model.joblib")
scaler = load("./models/output/scaler.joblib")


# define swagger custom params
class CustomParams:
    def __init__(
        self,
        HighBP: float = Form(
            ...,
            description="1 high bloodpressure, 0 no",
            ge=0,
            le=1,
        ),
        HighChol: float = Form(
            ...,
            description="1 high cholesterol, 0 no",
            ge=0,
            le=1,
        ),
        BMI: float = Form(
            ...,
            description="underweight < 18.5, normal 18.5-25, overweight 25-30, obese 30-99",
            ge=18.5,
            le=99,
        ),
        Smoker: float = Form(
            ..., description="1 smoker/was a smoker, 0 no", ge=0, le=1
        ),
        Stroke: float = Form(..., description="1 had a stroke, 0 no", ge=0, le=1),
        Diabetes: float = Form(
            ...,
            description="2 diabetes, 1 pre diabetes/borderline diabetes, 0 no",
            ge=0,
            le=2,
        ),
        GenHlth: float = Form(
            ..., description="1 excellent health, 5 very poor health", ge=1, le=5
        ),
        DiffWalk: float = Form(
            ...,
            description="1 difficulty walking and climbing stairs, 0 no",
            ge=0,
            le=1,
        ),
        Sex: float = Form(
            ...,
            description="1 male, 0 female",
            ge=0,
            le=1,
        ),
        Age: float = Form(
            ...,
            description="1 18-24, 2 25-29, 3 30-34, 4 35-39, 5 40-44, 6 45-49, 7 50-59, 8 55-59, 9 60-64, 10 65-69, 11 70-74, 12 75-79, 13 80-80+",
            ge=1,
            le=13,
        ),
    ):
        self.HighBP = HighBP
        self.HighChol = HighChol
        self.BMI = BMI
        self.Smoker = Smoker
        self.Stroke = Stroke
        self.Diabetes = Diabetes
        self.GenHlth = GenHlth
        self.DiffWalk = DiffWalk
        self.Sex = Sex
        self.Age = Age


@app.post("/create_model")
async def create_model():
    try:
        build_model()
        return {"message": "Model creation completed"}
    except Exception as e:
        print(f"Error creating model: {e}")
        raise HTTPException(status_code=400, detail="Error creating model" + str(e))


@app.post("/predict_model")
async def predict_model(data: CustomParams = Depends()):
    try:
        # create df
        data_dict = data.__dict__
        df = pd.DataFrame([data_dict])

        # process data
        processed_df = process_data(df)

        # scale data
        processed_df = pd.DataFrame(
            scaler.transform(processed_df), columns=processed_df.columns
        )

        # make predictions
        predictions = model.predict(processed_df)
        predictions = predictions[0].tolist()
        predictions = [float(p) for p in predictions]

        # return predictions
        return {
            "categorized_predictions": categorize_prediction(predictions, [data_dict]),
        }

    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail="Error predicting data" + str(e))

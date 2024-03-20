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
            description="1 for high blood pressure, 0 for none",
            ge=0,
            le=1,
        ),
        HighChol: float = Form(
            ...,
            description="1 for high cholesterol, 0 for none",
            ge=0,
            le=1,
        ),
        CholCheck: float = Form(
            ...,
            description="1 if cholesterol check done in the last 5 years, 0 otherwise",
            ge=0,
            le=1,
        ),
        BMI: float = Form(
            ...,
            description="Under 18.5: underweight, 18.5-25: normal, 25-30: overweight, 30-35: obesity class I, 35-40: obesity class II, above 40: obesity class III",
            ge=1,
            le=99,
        ),
        Smoker: float = Form(
            ...,
            description="1 for current or past smoker (at least 100 cigarettes), 0 otherwise",
            ge=0,
            le=1,
        ),
        Stroke: float = Form(
            ...,
            description="1 for stroke, 0 for none",
            ge=0,
            le=1,
        ),
        Diabetes: float = Form(
            ...,
            description="2 for type 2 diabetes, 1 for type 1 diabetes, 0 for none",
            ge=0,
            le=2,
        ),
        PhysActivity: float = Form(
            ...,
            description="1 if physical activity in the last 30 days, 0 otherwise",
            ge=0,
            le=1,
        ),
        Fruits: float = Form(
            ...,
            description="1 if at least 1 fruit per day, 0 otherwise",
            ge=0,
            le=1,
        ),
        Veggies: float = Form(
            ...,
            description="1 if at least 1 vegetable per day, 0 otherwise",
            ge=0,
            le=1,
        ),
        HvyAlcoholConsump: float = Form(
            ...,
            description="1 if heavy alcohol consumption (men: at least 14 drinks/week, women: at least 7 drinks/week), 0 otherwise",
            ge=0,
            le=1,
        ),
        AnyHealthcare: float = Form(  # changed from AnyHeatlhcare to AnyHealthcare
            ...,
            description="1 if has health insurance or social coverage, 0 otherwise",
            ge=0,
            le=1,
        ),
        NoDocbcCost: float = Form(
            ...,
            description="1 if hasn't seen a doctor in the last 12 months due to cost, 0 otherwise",
            ge=0,
            le=1,
        ),
        GenHlth: float = Form(
            ...,
            description="1 for excellent health, 5 for very poor health",
            ge=1,
            le=5,
        ),
        MentHlth: float = Form(
            ...,
            description="Number of days with poor mental health out of the last 30 days",
            ge=0,
            le=30,
        ),
        PhysHlth: float = Form(
            ...,
            description="Number of days with poor physical health out of the last 30 days",
            ge=0,
            le=30,
        ),
        DiffWalk: float = Form(
            ...,
            description="1 if difficulty walking or climbing stairs, 0 otherwise",
            ge=0,
            le=1,
        ),
        Sex: float = Form(
            ...,
            description="1 for male, 0 for female",
            ge=0,
            le=1,
        ),
        Age: float = Form(
            ...,
            description="1 for 18-24, 2 for 25-29, ..., 13 for 80+",
            ge=1,
            le=13,
        ),
        # Education: float = Form(
        #     ...,
        #     description="1, 2, 3 for no high school diploma, 4 for high school diploma, 5 for higher education, 6 for graduate degree",
        #     ge=1,
        #     le=6,
        # ),
        # Income: float = Form(
        #     ...,
        #     description="1, 2 for less than $15,000/year, 3, 4 for $15,000-$25,000/year, 5 for $25,000-$35,000/year, 6 for $35,000-$50,000/year, 7, 8 for over $50,000/year",
        #     ge=1,
        #     le=8,
        # ),
    ):
        self.HighBP = HighBP
        self.HighChol = HighChol
        self.CholCheck = CholCheck
        self.BMI = BMI
        self.Smoker = Smoker
        self.Stroke = Stroke
        self.Diabetes = Diabetes
        self.PhysActivity = PhysActivity
        self.Fruits = Fruits
        self.Veggies = Veggies
        self.HvyAlcoholConsump = HvyAlcoholConsump
        self.AnyHealthcare = AnyHealthcare
        self.NoDocbcCost = NoDocbcCost
        self.GenHlth = GenHlth
        self.MentHlth = MentHlth
        self.PhysHlth = PhysHlth
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

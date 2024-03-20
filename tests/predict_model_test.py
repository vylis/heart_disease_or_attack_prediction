from fastapi.testclient import TestClient
import main

client = TestClient(main.app)


def test_predict_model():
    data = {
        "HighBP": 0,
        "HighChol": 0,
        "CholCheck": 1,
        "BMI": 22,
        "Smoker": 1,
        "Stroke": 0,
        "Diabetes": 1,
        "PhysActivity": 1,
        "Fruits": 1,
        "Veggies": 1,
        "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1,
        "NoDocbcCost": 0,
        "GenHlth": 3,
        "MentHlth": 15,
        "PhysHlth": 2,
        "DiffWalk": 0,
        "Sex": 0,
        "Age": 2,
    }

    response = client.post("/predict_model", data=data)
    assert response.status_code == 200
    assert "categorized_predictions" in response.json()

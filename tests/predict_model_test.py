from fastapi.testclient import TestClient
import main

client = TestClient(main.app)


def test_predict_model():
    data = {
        "HighBP": 1,
        "HighChol": 0,
        "BMI": 28,
        "Smoker": 1,
        "Stroke": 0,
        "Diabetes": 1,
        "GenHlth": 2,
        "DiffWalk": 0,
        "Sex": 1,
        "Age": 6,
    }

    response = client.post("/predict_model", data=data)
    assert response.status_code == 200
    assert "categorized_predictions" in response.json()

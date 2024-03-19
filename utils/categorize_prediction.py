# categorize prediction based on prediction value and input data
def categorize_prediction(predictions, data):
    categories = []
    for prediction, data_dict in zip(predictions, data):
        if data_dict["Stroke"] == 1 or (
            (data_dict["Diabetes"] == 2 or data_dict["Diabetes"] == 1)
            and (
                data_dict["Smoker"] == 1
                or data_dict["HighChol"] == 1
                or data_dict["HighBP"] == 1
            )
        ):
            categories.append("Risque très élevé")
        elif (
            data_dict["Diabetes"] == 2
            or data_dict["HighBP"] == 1
            or data_dict["HighChol"] == 1
            or (data_dict["Diabetes"] == 1 and data_dict["Age"] > 1)
        ):
            categories.append("Risque élevé")
        elif prediction < 0.1:
            categories.append("Risque faible")
        elif 0.1 <= prediction < 0.5:
            categories.append("Risque modéré")
        elif 0.5 <= prediction < 0.85:
            categories.append("Risque élevé")
        elif prediction >= 85:
            categories.append("Risque très élevé")
    return categories

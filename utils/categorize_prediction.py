# categorize prediction based on prediction value
def categorize_prediction(predictions):
    categories = []
    for prediction in predictions:
        if prediction < 0.1:
            categories.append("Risque faible")
        elif 0.1 <= prediction < 0.5:
            categories.append("Risque modéré")
        elif 0.5 <= prediction < 0.90:
            categories.append("Risque élevé")
        elif prediction >= 0.90:
            categories.append("Risque très élevé")
    return categories

# Heart Disease Prediction

This API uses machine learning to predict heart disease risk. 

## Getting Started

1. **Create a virtual environment (optional)**: Create a virtual environment.
    ```bash
    python -m venv venv
    ```

2. **Activate the virtual environment (optional)**: Activate the virtual environment.
    ```bash
    source venv/bin/activate
    ```

3. **Install dependencies**: Install the necessary dependencies.
    ```bash
    pip install -r requirements.txt
    ```

4. **Start the API**: Start the application.
    ```bash
    uvicorn main:app --reload
    ```

## Usage

- **Swagger UI**: Use `http://localhost:8000/docs` to access the Swagger where you can test the API routes.

- **Create Model**: Use the `create_model` route and wait for the model to be created.

- **Predict**: Once the model is created, you can use the `predict` route to make predictions.

- **Tests**: Run `pytest` in the terminal to execute the tests.

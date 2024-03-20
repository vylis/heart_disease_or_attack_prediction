# Heart Disease Prediction

- create venv (optional)

```bash
python -m venv venv
```

- activate venv (optional)

```bash
source venv/bin/activate
```

- install requirements

```bash
  pip install -r requirements.txt
```

- start api

```bash
uvicorn main:app --reload
```

- access swagger with your browser

```
http://localhost:8000/docs
```

- first use the create_model route if you don't have the model and wait for the model to be created

- then you can use the predict routes

- you can run unit tests and e2e tests with pytest

```bash
pytest
```

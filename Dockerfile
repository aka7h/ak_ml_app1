FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install fastapi uvicorn joblib scikit-learn pandas xgboost pickle5

# Expose the port the app will run on
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
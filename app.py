from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import joblib
import os


model = joblib.load('final_model.pkl')

# Create FastAPI app
app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open('templates/index.html', 'r') as file:
        return file.read()
    

# Define the input data format
class SalesData(BaseModel):
    Warehouse_block :int
    Mode_of_Shipment :int
    Customer_care_calls :int
    Customer_rating :int
    Cost_of_the_Product :int
    Prior_purchases :int
    Product_importance :int
    Gender :int
    Discount_offered :int
    Weight_in_gms :int

# Prediction endpoint
@app.post("/predict/")
def predict_sales(data: SalesData):
    # Convert input data into a format suitable for prediction
    input_data = [[data.Warehouse_block,
        data.Mode_of_Shipment,
        data.Customer_care_calls,
        data.Customer_rating,
        data.Cost_of_the_Product,
        data.Prior_purchases,
        data.Product_importance,
        data.Gender,
        data.Discount_offered,
        data.Weight_in_gms]]

    # Make prediction
    prediction = model.predict(input_data)

    print(prediction)

    # Return the predicted sales
    return {"predicted_sales": prediction[0]}


if __name__=="__main__":
    uvicorn.run(app,host="0.0.0.0",port=8000)


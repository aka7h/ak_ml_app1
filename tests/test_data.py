import pandas as pd

def test_data_dependencies():
    data = pd.read_csv("../data/train.csv")
    required_columns = ['Warehouse_block',
                    'Mode_of_Shipment',
                    'Customer_care_calls',
                    'Customer_rating',
                    'Cost_of_the_Product',
                    'Prior_purchases',
                    'Product_importance',
                    'Gender',
                    'Discount_offered',
                    'Weight_in_gms']
    for column in required_columns:
        assert column in data.columns, f"Missing required column: {column}"
    assert not data.isnull().any().any(), "Dataset contains null values"
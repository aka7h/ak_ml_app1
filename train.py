# %%
import pandas as pd 
import numpy as np
import sklearn
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle, joblib

# %%

data = pd.read_csv('./data/train.csv')
data.shape

# %%
data.head()

# %%
# print(data['Mode_of_Shipment'].unique())

# print(data['Product_importance'].unique())

# print(data['Gender'].unique())

# print(data['Warehouse_block'].unique())

# %%
wh_dict = {'A':1,'B':2,'C':3,'D':4,'F':5}
g_dict = {'F':1,'M':2}
prd_dict = {'low':1,'medium':2,'high':3}
md_dict = {'Flight':1,'Ship':2,'Road':3}

# %%
data['Warehouse_block'] = data['Warehouse_block'].replace(wh_dict)
data['Mode_of_Shipment'] = data['Mode_of_Shipment'].replace(md_dict)
data['Product_importance'] = data['Product_importance'].replace(prd_dict)
data['Gender'] = data['Gender'].replace(g_dict)

# %%
# train.columns.tolist()

# %%
data.columns = ['ID',
 'Warehouse_block',
 'Mode_of_Shipment',
 'Customer_care_calls',
 'Customer_rating',
 'Cost_of_the_Product',
 'Prior_purchases',
 'Product_importance',
 'Gender',
 'Discount_offered',
 'Weight_in_gms',
 'Reached_on_Time']

# %%
data.describe()

# %%
X = ['Warehouse_block',
 'Mode_of_Shipment',
 'Customer_care_calls',
 'Customer_rating',
 'Cost_of_the_Product',
 'Prior_purchases',
 'Product_importance',
 'Gender',
 'Discount_offered',
 'Weight_in_gms']
Y = 'Reached_on_Time'

# %%
if data.isnull().sum().any():
    print("Data contains missing values! Please handle them before proceeding.")
else:
    print("No missing values in the dataset.")

# %%
X_train, X_test, y_train, y_test = train_test_split(data[X], data[Y], test_size=0.25, random_state=45)

# %%
model = XGBClassifier()

# %%
model.fit(X_train[X], y_train)

# %%
y_pred = model.predict(X_test)
prec = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# %%
print(f"Model Evaluation Metrics:")
print(f"Precision Score: {prec}")
print(f"Recall Score: {recall}")


# %%

model_filename = 'final_model.pkl'
joblib.dump(model, model_filename)
print(f"Model saved as {model_filename}")



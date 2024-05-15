from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

class CustomerInfo(BaseModel):
    Avg_Credit_Limit: float
    total_credit_cards: int
    total_bank_visits: int
    total_visits_online: int
    total_calls: int

app = FastAPI()


@app.post("/classify")
def classify(customerInfo: CustomerInfo):
    with open('kmeans.pickle', 'rb') as f:
        kmeans = pickle.load(f)
        items = list(customerInfo.model_dump().items())
        data = np.array(items)
        print(kmeans.predict(data))
    return customerInfo


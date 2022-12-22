from typing import Union
from fastapi import FastAPI
import pandas as pd
import pickle
import numpy as np

app = FastAPI()

@app.get("/")
async def root(price: Union[float, None] = 0):
  user_data = pd.DataFrame({"price" : [float(price)]})
  pickle_model = pickle.load(open('pipeline.pkl', 'rb'))
  pickle_predict = pickle_model.predict(user_data)
  return {'predict': f"{pickle_predict[0]}"}

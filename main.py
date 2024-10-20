from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the best model
best_model = joblib.load('best_model.joblib')

# Load train data to get feature names for input
train_df = pd.read_csv('train.csv')
feature_names = train_df.columns.drop(['id', 'Rings'])

# Encode Sex feature
le = LabelEncoder()
train_df['Sex'] = le.fit_transform(train_df['Sex'])
sex_encode = {val: key for key, val in enumerate(le.classes_)}

# Define FastAPI app:----
app = FastAPI()


# Define request body model
class InputData(BaseModel):
    Length: float
    Diameter: float
    Height: float
    WholeWeight: float
    ShuckedWeight: float
    VisceraWeight: float
    ShellWeight: float
    Sex: str


# Define prediction endpoint
@app.post("/predict/")
async def predict(input_data: InputData):
    try:
        # Preprocess input data
        sex_mapping = {'male': 'M', 'female': 'F', 'infant': 'I'}
        if input_data.Sex.lower() in sex_mapping:
            input_data.Sex = sex_mapping[input_data.Sex.lower()]

        # Prepare input data
        input_values = [input_data.Length, input_data.Diameter, input_data.Height,
                        input_data.WholeWeight, input_data.ShuckedWeight,
                        input_data.VisceraWeight, input_data.ShellWeight,
                        sex_encode[input_data.Sex]]

        # Predict rings using the best model
        prediction = best_model.predict([input_values])[0]

        return {"Predicted number of rings": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occurred: {e}")


# Define root endpoint
@app.get("/")
def root():
    return {"message": "Welcome to the Abalone Rings Prediction API!"}

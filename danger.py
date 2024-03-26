import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
import os
import streamlit as st

# Read data (replace with your data loading logic)
data = pd.read_csv('soilTexture.csv')

# Check for missing values
missing_values = data.isnull().sum()
st.write(f'Missing values:\n{missing_values}')

# Impute missing values in numerical columns
numerical_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
imputer = SimpleImputer(strategy='mean')
data_numeric = imputer.fit_transform(data[numerical_columns])
data_numeric = pd.DataFrame(data_numeric, columns=numerical_columns)

# Impute missing values in categorical columns
categorical_columns = ['label', 'soil_texture']
imputer = SimpleImputer(strategy='most_frequent')
data_categorical = imputer.fit_transform(data[categorical_columns])
data_categorical = pd.DataFrame(data_categorical, columns=categorical_columns)

# Combine numerical and categorical data
data_imputed = pd.concat([data_numeric, data_categorical], axis=1)

# One-hot encode categorical features with 'drop='first' to avoid multicollinearity
encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
encoded_data = encoder.fit_transform(data_imputed.drop(columns=['label']))
X_encoded = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(data_imputed.drop(columns=['label']).columns))

# Concatenate with the original label column
X = pd.concat([X_encoded, data_imputed['label']], axis=1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X.drop(columns=['label']), X['label'], test_size=0.2, random_state=42)

# Train a Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Function to suggest crop with enhanced handling of unknown categories
def suggest_crop(N, P, K, temperature, humidity, ph, rainfall, soil_texture):
    # Prepare user input for prediction
    user_data = pd.DataFrame({
        'N': [N],
        'P': [P],
        'K': [K],
        'temperature': [temperature],
        'humidity': [humidity],
        'ph': [ph],
        'rainfall': [rainfall],
        'soil_texture': [soil_texture]
    })

    # One-hot encode categorical features
    user_encoded = encoder.transform(user_data)

    # Predict the crop
    prediction = model.predict(user_encoded)
    return prediction[0]
        

# st.set_page_config(page_title='Crop Advice System', page_icon=':seedling:', layout='centered', initial_sidebar_state='expanded')
st.title("Crop Advisor")

st.write("This app suggests the best crop to plant based on the given conditions.")

# User input form
N = st.slider("N", 0.0, 100.0, 30.0)
P = st.slider("P", 0.0, 100.0, 95.0)
K = st.slider("K", 0.0, 100.0, 46.0)
temperature = st.slider("Temperature", 0.0, 50.0, 35.0)
humidity = st.slider("Humidity", 0.0, 100.0, 75.0)
ph = st.slider("pH", 0.0, 14.0, 7.0)
rainfall = st.slider("Rainfall", 0.0, 200.0, 92.0)
soil_texture = st.selectbox("Soil Texture", ['clay', 'loamy', 'sandy', 'silty'])

if st.button("Craft Advice"):
    predicted_crop = suggest_crop(N, P, K, temperature, humidity, ph, rainfall, soil_texture)
    st.success(f"Suggested Crop: Based on the given conditions, the best crop to plant is {predicted_crop}")

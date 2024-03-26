from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from data import check_crops
import numpy as np
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()
# Set your OpenAI API key

# Read data (replace with your data loading logic)
data = pd.read_csv('crops.csv')

# Check for missing values
missing_values = data.isnull().sum()
print(f'Missing values:\n{missing_values}')

# Impute missing values in numerical columns
numerical_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
imputer = SimpleImputer(strategy='mean')
data_numeric = imputer.fit_transform(data[numerical_columns])
data_numeric = pd.DataFrame(data_numeric, columns=numerical_columns)
print(f'Imputed Numerical Data:\n{data_numeric}')

# Impute missing values in categorical columns (replace with appropriate strategies)
categorical_columns = ['label']
imputer = SimpleImputer(strategy='most_frequent')
data_categorical = imputer.fit_transform(data[categorical_columns])
data_categorical = pd.DataFrame(data_categorical, columns=categorical_columns)
print(f'Imputed Categorical Data:\n{data_categorical}')

# Combine numerical and categorical data
data_imputed = pd.concat([data_numeric, data_categorical], axis=1)
print(f'Imputed Data:\n{data_imputed}')

# One-hot encode categorical features with 'drop='first' to avoid multicollinearity
encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
encoded_data = encoder.fit_transform(data_imputed.drop(columns=['label']))
print(f'Encoded Data:\n{encoded_data.toarray()}')

X_encoded = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(data_imputed.drop(columns=['label']).columns))

# Concatenate with the original label column
X = pd.concat([X_encoded, data_imputed['label']], axis=1)
print(f'X:\n{X}')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X.drop(columns=['label']), X['label'], test_size=0.2, random_state=42)

# Train a Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Function to suggest crop with enhanced handling of unknown categories
def recommend(N, P, K, temperature, humidity, ph, rainfall):
    # Prepare user input for prediction
    user_data = pd.DataFrame({
        'N': [N],
        'P': [P],
        'K': [K],
        'temperature': [temperature],
        'humidity': [humidity],
        'ph': [ph],
        'rainfall': [rainfall]
    })

    # Handle unknown categories in user input
    for column in user_data.columns:
        if user_data[column].dtype == 'O' and column in encoder.get_feature_names_out():
            user_data[column] = user_data[column].astype(str)
            categories = set(encoder.get_feature_names_out([column]))
            missing_categories = categories - set(user_data[column].astype('category').cat.categories)
            print(f'Missing categories for {column}: {missing_categories}')

            # Choose a strategy for handling missing categories:
            #   - Add and impute (example):
            user_data[column] = user_data[column].astype
            print(f'Imputed {column}: {user_data[column]}')

    # One-hot encode categorical features
    user_encoded = encoder.transform(user_data)
    print(f'User Encoded: {user_encoded.toarray()}')

    # Predict the crop
    prediction = model.predict(user_encoded)
    print(f'Prediction: {prediction}')
    return prediction[0]

# @app.route('/suggest_crop', methods=['POST'])
# def suggest_crop_route():
#     data = request.json
#     N = data['N']
#     P = data['P']
#     K = data['K']
#     temperature = data['temperature']
#     humidity = data['humidity']
#     ph = data['ph']
#     rainfall = data['rainfall']
#     predicted_crop = suggest_crop(N, P, K, temperature, humidity, ph, rainfall)
#     return jsonify({"predicted_crop": predicted_crop})
def get_advice(predicted_crop, temperature, humidity, ph_value, rainfall):
    advice = ""
    # Example advice based on different conditions
    if temperature > 25 and humidity > 60:
        advice += "Given the high temperature and humidity, ensure proper ventilation and irrigation for optimal growth. "
    if ph_value < 6.5:
        advice += "Adjust soil pH to optimize nutrient uptake. Consider using lime to raise pH if it's too acidic. "
    if rainfall < 500:
        advice += "Supplement irrigation during dry spells to maintain soil moisture levels. "
    # Add more conditions and advice as needed based on the characteristics of the crop
    if predicted_crop == "Rice":
        advice += "Consider adding more nitrogen and phosphorus to enhance the yield of rice. "
    elif predicted_crop == "Wheat":
        advice += "Consider adding more phosphorus to enhance the yield of wheat. "
    elif predicted_crop == "Cotton":
        advice += "Consider adding more phosphorus to enhance the yield of cotton. "
    if predicted_crop in ["Rice", "Wheat", "Cotton"]:
        advice += "Consider adding more phosphorus to enhance the yield of selected crops. "
    if ph_value > 6.5:
        advice += "Consider using lime to raise pH if it's too acidic. "
    if rainfall > 500:
        advice += "Consider using fungicides to control soil erosion. "

    # If no specific advice is provided, give a generic advice
    if not advice:
        advice = "Ensure proper care and management practices for optimal crop growth."

    return advice

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_crop_advice', methods=['POST'])
def get_crop_advice():
    if request.method == 'POST':
        n_value = float(request.form['N'])
        p_value = float(request.form['P'])
        k_value = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph_value = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        predicted_crop = recommend(n_value, p_value, k_value, temperature, humidity, ph_value, rainfall)

        if predicted_crop:
            advice = get_advice(predicted_crop, temperature, humidity, ph_value, rainfall)
            return render_template('result.html', crop=predicted_crop, advice=advice)
        else:
            return render_template('result.html', message="Oops! No crop advice available for the given inputs.")

if __name__ == '__main__':
    app.run(debug=True)

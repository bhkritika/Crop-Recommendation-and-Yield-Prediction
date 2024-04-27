from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder

# Load the pre-trained model and encoder
model_file_path = 'model.pkl'
encoder_file_path = 'your_encoder.pkl'

with open(model_file_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(encoder_file_path, 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)

# Define categorical and numerical features for prediction
categorical_features = ['state_names', 'district_names', 'season_names', 'crop_names']
numerical_features = ['crop_year', 'area']

def predict_production_and_yield(new_data):
    # Convert new_data into DataFrame
    new_data_df = pd.DataFrame(new_data, index=[0])

    # One-hot encode categorical features
    new_encoded_categorical = encoder.transform(new_data_df[categorical_features])
    
    # Convert encoded categorical features to DataFrame
    new_encoded_df = pd.DataFrame(new_encoded_categorical.toarray(),
                                   columns=encoder.get_feature_names_out(categorical_features))

    # Concatenate numerical features with encoded categorical features
    new_input = pd.concat([new_data_df[numerical_features].reset_index(drop=True), new_encoded_df], axis=1)

    # Make predictions using the trained model
    predicted_yield = model.predict(new_input)
    predicted_production = predicted_yield[0]

    # Calculate predicted yield (assuming area is provided in hectares)
    area_ha = new_data['area']
    predicted_yield_value = area_ha / predicted_production
    
    formatted_predicted_yield = f"{predicted_yield_value:.2f}"  # Format predicted yield to two decimal places

    return predicted_production, formatted_predicted_yield




# Load the pre-trained model and encoder for recommending crops
model_file_path_recommendation = 'modelb.pkl'
encoder_file_path_recommendation = 'encoder.pkl'

with open(model_file_path_recommendation, 'rb') as model_file:
    model_recommendation = pickle.load(model_file)

with open(encoder_file_path_recommendation, 'rb') as encoder_file:
    encoder_recommendation = pickle.load(encoder_file)


# Define categorical and numerical features for recommending crops
categorical_features_recommendation = ['soil_type']
numerical_features_recommendation = ['N', 'P', 'K', 'temperature', 'humidity', 'wind_speed', 'precipitation']

def recommend_crop_from_input(input_data):
    # Ensure feature order and names match those used during model training
    feature_order = ['temperature', 'wind_speed', 'precipitation', 'humidity', 'N', 'P', 'K', 'soil_type']

    # Create DataFrame with correct feature order
    new_data = pd.DataFrame({feature: input_data[feature] for feature in feature_order})

    # One-hot encode soil type (assuming 'soil_type' is a categorical feature)
    new_encoded_soil_type = encoder_recommendation.transform(new_data[['soil_type']])

    # Concatenate numerical features with encoded soil type
    new_encoded_data = pd.concat([
        new_data.drop(columns=['soil_type']),  # Exclude soil_type from numerical features
        pd.DataFrame(new_encoded_soil_type.toarray(), columns=encoder_recommendation.get_feature_names_out(['soil_type']))
    ], axis=1)

    # Predict recommended crop using the trained model
    recommended_crop = model_recommendation.predict(new_encoded_data)

    return recommended_crop[0]




app = Flask(__name__)
# Prediction routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict_crop_yield():
    if request.method == 'POST':
        required_fields = ['state_name', 'district_name', 'crop_name', 'season_name', 'crop_year', 'area']

        form_data = {k: request.form.get(k, '') for k in required_fields}

        missing_fields = [k for k in required_fields if not form_data[k]]

        if missing_fields:
            return f"Missing required fields: {', '.join(missing_fields)}. Please go back and fill in all the required fields.", 400

        new_data = {
            'state_names': form_data['state_name'],
            'district_names': form_data['district_name'],
            'crop_names': form_data['crop_name'],
            'season_names': form_data['season_name'],
            'crop_year': int(form_data['crop_year']),
            'area': float(form_data['area'])
        }

        predicted_production, predicted_yield = predict_production_and_yield(new_data)

        return render_template('prediction_result.html',
                                predicted_production=predicted_production,
                                predicted_yield=predicted_yield)
    
# Recommendation routes
@app.route('/recommendation')
def recom():
    return render_template('recom.html')

@app.route('/recommend', methods=['POST'])
def recommend_crop():
    if request.method == 'POST':
        # Extract form data
        temperature = float(request.form['temperature'])
        wind_speed = float(request.form['wind_speed'])
        precipitation = float(request.form['precipitation'])
        humidity = float(request.form['humidity'])
        nitrogen = float(request.form['nitrogen'])
        phosphorus = float(request.form['phosphorus'])
        potassium = float(request.form['potassium'])
        soil_type = request.form['soil_type']

        # Create input data dictionary
        input_data = {
            'temperature': [temperature],
            'wind_speed': [wind_speed],
            'precipitation': [precipitation],
            'humidity': [humidity],
            'N': [nitrogen],
            'P': [phosphorus],
            'K': [potassium],
            'soil_type': [soil_type]
        }

        # Get recommended crop
        recommended_crop = recommend_crop_from_input(input_data)

        return render_template('recom_result.html', crop=recommended_crop)

if __name__ == '__main__':
    app.run(debug=True)

# Crop Recommendation and Yield Prediction
 
Agriculture is a vital sector in India, significantly influencing both the economy and the livelihoods of a vast majority of its population. 
However, the sector faces numerous challenges, magnified by the unpredictable consequences of climate change, such as erratic weather patterns and varying soil conditions. 
Our solution is a comprehensive, machine learning-powered platform designed to revolutionize agricultural practices by offering precision crop yield predictions and adaptive recommendations for sustainable farming. 
This project provides a web application that helps farmers with crop recommendations based on soil and weather conditions, and crop yield predictions based on historical data.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Screenshots](#screenshots)

## Introduction

The Crop Recommendation and Yield Prediction System is designed to assist farmers in making informed decisions about which crops to plant and predict the potential yield based on various parameters.

## Features

- **Crop Recommendation**: Recommends the best crop to plant based on soil composition, weather conditions, and other factors.
- **Crop Yield Prediction**: Predicts the yield of a crop based on state name, district name, crop year, season name, crop name, and area.

## Usage

### Crop Recommendation

1. Navigate to the Crop Recommendation page.
2. Enter the required parameters such as nitrogen level, phosphorus level, potassium level, temperature, humidity, wind speed, precipitation, and soil type.
3. Click the "Get Recommendations" button to receive the recommended crop.

### Crop Yield Prediction

1. Navigate to the Crop Yield Prediction page.
2. Enter the required parameters such as state name, district name, crop year, season name, crop name, and area.
3. Click the "Predict Yield" button to receive the predicted crop yield.

## Technologies Used

- **Frontend**: HTML, CSS
- **Backend**: Flask
- **Machine Learning**: Scikit-learn, Pandas, Joblib

## Dataset

The datasets used for training the models are sourced from [Kaggle]and other reliable agricultural databases.

## Model Training

The machine learning models for crop recommendation and yield prediction were trained using the following steps:

1. Data preprocessing and feature engineering.
2. Model selection and hyperparameter tuning.
3. Training and evaluation using Scikit-learn.
4. Saving the trained models with Joblib.

## Screenshots 

1. Webpage
![1  ](https://github.com/bhkritika/Crop-Recommendation-and-Yield-Prediction/assets/141895513/417df072-8173-489b-a85b-041ad1e030cf)

2. Crop Recommendation Page
![2  ](https://github.com/bhkritika/Crop-Recommendation-and-Yield-Prediction/assets/141895513/1dca0ba8-9c10-485d-9f8f-02b1562e939e)

3. Ouput
![4](https://github.com/bhkritika/Crop-Recommendation-and-Yield-Prediction/assets/141895513/492203c0-47fa-4fb3-a7fd-bed7f7b31ee2)

4. Crop Yield Prediction Page
![3  ](https://github.com/bhkritika/Crop-Recommendation-and-Yield-Prediction/assets/141895513/39afe447-05f2-4608-9477-0928bb876224)

5. Output
![5](https://github.com/bhkritika/Crop-Recommendation-and-Yield-Prediction/assets/141895513/23125926-2420-4253-bc66-b65db055b05b)

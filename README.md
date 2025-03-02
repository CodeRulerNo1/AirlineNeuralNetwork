# Airline Profitability Prediction
This project is a neural network model designed to predict airline profitability based on multiple factors.
In the airline industry, maximizing profitability is a complex challenge influenced by multiple operational and financial factors. Given historical flight performance data, your task is to develop a machine learning model that accurately predicts profit (USD) for each flight based on features.

## Documentation
**1. Objective**

The objective of this project is to create a machine learning model, a Neural Network in this case, to forecast flight profitability (USD) from different operational and financial parameters. The model must provide high accuracy, generalizability, and explainability to facilitate better decision-making for airline operators.

**2. Data Preprocessing**

2.1 Data Cleaning

    One-hot encoded categorical variables (flight number)

    Created new features like delay.

2.2 Data Scaling

    Applied Standard Scaler to normalize numerical features for improved neural network convergence.

**3. Model Architecture**

    The model is a fully connected feedforward neural network (FNN) implemented using TensorFlow/Keras.

3.1 Neural Network Design

![alt text](https://github.com/CodeRulerNo1/AirlineNeuralNetwork/blob/main/NN%20(2).png)
     
![alt text](https://github.com/CodeRulerNo1//AirlineNeuralNetwork/blob/main/model_architecture.png?raw=true)
3.2 Regularization & Optimization

    L2 Regularization (λ = 0.001): Avoids overfitting.

    Adam Optimizer: Learning rate that adapts with each step for quicker convergence.

    Early Stopping: Trains until validation loss no longer improves.
**4. Model Training & Evaluation**

4.1 Training
    
    epochs = 75 

    batch_size = 64

4.2 Evaluation Metrics

    Mean Absolute Error (MAE): Measures average absolute prediction error.

    R² Score: Measures how well predictions fit actual values.

**6. Visualization**

Training Visualisation graphs

![alt text](https://github.com/CodeRulerNo1//AirlineNeuralNetwork/blob/main/Training.png?raw=true)

Important features for the model

![alt text](https://github.com/CodeRulerNo1//AirlineNeuralNetwork/blob/main/Important_features.png?raw=true)
![alt text](https://github.com/CodeRulerNo1//AirlineNeuralNetwork/blob/main/important_features2.png?raw=true)

## Requirements

tensorflow==2.13.0
pandas
numpy
scikit-learn
seaborn
matplotlib

## Steps to train model
1. Open the notebook at [Training_Neural_Network_airline](https://github.com/CodeRulerNo1/AirlineNeuralNetwork/blob/main/Models%20and%20Notebooks/Training_Neural_Network_airline.ipynb)
2. Import your data to dataframe named df
3. Run the code in the Notebook.

## Steps to test model
1. Open the notebook at [Test_Models](https://github.com/CodeRulerNo1/AirlineNeuralNetwork/blob/main/Models%20and%20Notebooks/Test_Models.ipynb)
2. Also download both the models.
3. Import your data to dataframe named df.
4. Run the code in the Notebook.
   
## Training Data Link

[flight_data.csv](https://docs.google.com/spreadsheets/d/1eALZhnY5bEJ4uCi9BCjN2fpx8jRIzwWo/edit?usp=sharing&ouid=109976760607215104976&rtpof=true&sd=true)
## Author

[@Ichhabal Singh](https://www.github.com/CodeRulerNo1)

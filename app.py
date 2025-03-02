from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import pandas as pd

app = Flask(__name__)

COLUMN_NAMES = [
    "Flight Number", "Scheduled Departure Timestamp", "Actual Departure Timestamp", "Delay (Minutes)",
    "Aircraft Utilization (Hours/Day)", "Turnaround Time (Minutes)", "Load Factor (%)",
    "Fleet Availability (%)", "Maintenance Downtime (Hours)", "Fuel Efficiency (ASK)",
    "Revenue (USD)", "Operating Cost (USD)", "Net Profit Margin (%)", "Ancillary Revenue (USD)",
    "Debt-to-Equity Ratio", "Revenue per ASK", "Cost per ASK"
]

# Load trained model
model = tf.keras.models.load_model('airline_profit_model2.h5')

@app.route('/')
def home():
    return render_template('index.html')  # Load frontend

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if 'input' not in data:
            return jsonify({'error': 'Invalid input format. Expecting JSON with "input" key.'})

        #  Handle Comma-Separated Strings
        if isinstance(data['input'], str):
            data['input'] = data['input'].split(',')

        # Convert to DataFrame
        df = pd.DataFrame([data['input']], columns=COLUMN_NAMES)

        # Convert numeric fields
        for col in df.columns:
            if col not in ["Flight Number", "Scheduled Departure Timestamp", "Actual Departure Timestamp"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        #  Convert Flight Number to Numeric
        df["Flight Number"] = df["Flight Number"].astype(str).str.extract(r'(\d+)').fillna(0).astype(int)

        #  Convert Datetime to Unix Timestamp
        df["Scheduled Departure Timestamp"] = pd.to_datetime(df["Scheduled Departure Timestamp"], errors='coerce').astype(int) / 10**9
        df["Actual Departure Timestamp"] = pd.to_datetime(df["Actual Departure Timestamp"], errors='coerce').astype(int) / 10**9

        # Compute Departure Delay
        df["Departure Delay (minutes)"] = (df["Actual Departure Timestamp"] - df["Scheduled Departure Timestamp"]) / 60

        # Drop original datetime columns before passing to model
        df = df.drop(columns=["Scheduled Departure Timestamp", "Actual Departure Timestamp"])

        # Convert DataFrame to NumPy array for model prediction
        input_data = df.to_numpy().reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)[0][0]

        return jsonify({'prediction': float(prediction)/10000})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

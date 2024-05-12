# Import necessary libraries
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# Load the model
csv_file_path = 'Sample_Training_Session_Data.csv'
df = pd.read_csv(csv_file_path)
X = df[['Day of the Week', 'Time of Day', 'Duration', 'Student Registrations', 'Capacity']]
y = df['Attendance Rate']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Day of the Week', 'Time of Day']),
        ('num', 'passthrough', ['Duration', 'Student Registrations', 'Capacity'])
    ])
model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model.fit(X_train, y_train)

# Define a route for the API
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Validate the input
    required_columns = ['Day of the Week', 'Time of Day', 'Student Registrations']
    missing_columns = [col for col in required_columns if col not in data]
    if missing_columns:
        return jsonify({'error': 'Missing data for columns: ' + ', '.join(missing_columns)}), 400

    # Prepare user data for prediction
    user_data = pd.DataFrame({
        'Day of the Week': [data['Day of the Week']],
        'Time of Day': [data['Time of Day']],
        'Duration': [3],  # Fixed value
        'Student Registrations': [data['Student Registrations']],
        'Capacity': [100]  # Fixed value
    })

    # Predict attendance
    predicted_attendance = model.predict(user_data)
    return jsonify({'predicted_attendance_rate': predicted_attendance[0]})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5000)

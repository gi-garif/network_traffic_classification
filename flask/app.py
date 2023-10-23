import pandas as pd
import pickle

from flask import Flask, jsonify, request

# Load the model
pickle_in = open('rfc.pkl', 'rb')
model = pickle.load(pickle_in)

# Init the app
app = Flask("default")


# Setup prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Make predictions for each feature vector
    predictions = model.predict(pd.DataFrame([data]))

    # Create a response with class probabilities for each sample
    result = {"predictions": predictions[0]}

    return jsonify(result)


if __name__ == "__main__":
    # Run the app on local host and port 8000
    app.run(debug=True, host="0.0.0.0", port=8000)

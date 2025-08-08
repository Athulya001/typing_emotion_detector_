from flask import Flask, request, jsonify, render_template
from sklearn.linear_model import LogisticRegression
import numpy as np

app = Flask(__name__)

# Dummy training data (features: avg interval, pause rate, backspace rate)
X_train = np.array([
    [120, 0.02, 0.01],  # fast typing, few pauses/errors = happy
    [400, 0.1, 0.1],    # slow typing, many pauses/errors = sad
    [250, 0.05, 0.03]   # moderate speed and errors = neutral
])
y_train = np.array([1, 2, 0])

# Train model
model = LogisticRegression(multi_class='ovr', max_iter=1000)
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():
    data = request.get_json()
    intervals = data.get('intervals', [])
    pauseCount = data.get('pauseCount', 0)
    backspaceCount = data.get('backspaceCount', 0)
    totalKeys = data.get('totalKeys', 1)

    if len(intervals) == 0:
        return jsonify({'emotion': 'neutral'})

    avg_interval = sum(intervals) / len(intervals)
    pause_rate = pauseCount / totalKeys
    backspace_rate = backspaceCount / totalKeys

    features = np.array([[avg_interval, pause_rate, backspace_rate]])
    pred = model.predict(features)[0]

    emotion_map = {0: 'neutral', 1: 'happy', 2: 'sad'}
    emotion = emotion_map.get(pred, 'neutral')

    return jsonify({'emotion': emotion})

if __name__ == "__main__":
    app.run(debug=True)

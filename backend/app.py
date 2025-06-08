from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

@app.route('/api/hello')
def hello():
    return jsonify(message='Hello from Flask backend')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

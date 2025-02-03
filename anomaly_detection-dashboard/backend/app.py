from flask import Flask, request, jsonify
import yfinance as yf
from anomaly_detection import AnomalyDetector

app = Flask(__name__)

@app.route('/detect', methods=['GET'])
def detect():
    stock = request.args.get('stock', 'AAPL')
    method = request.args.get('method', 'isolation_forest')

    # Fetch stock data
    data = yf.download(stock, start="2023-01-01", end="2023-10-01")

    # Detect anomalies
    detector = AnomalyDetector(method=method)
    result = detector.detect_anomalies(data)

    # Return anomalies
    anomalies = result[result['anomaly'] == 1]
    return jsonify(anomalies.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
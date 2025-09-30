from flask import Flask, Response
from flask_cors import CORS
from evaluate_model import evaluate_model

app = Flask(__name__)
CORS(app)  # permite que frontend en 3000 acceda al backend

@app.route('/video_feed')
def video_feed():
    # evaluate_model devuelve frames procesados como stream
    return Response(
        evaluate_model(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

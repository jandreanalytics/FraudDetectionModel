from flask import Flask, render_template
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    api_url = os.getenv('API_URL', 'http://localhost:5000')
    return render_template('index.html', api_url=api_url)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)

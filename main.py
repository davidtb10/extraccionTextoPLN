import os.path

from flask import Flask, current_app
from flask import render_template
from flask import request
from Config import DevelopmentConfig
from flask import send_from_directory
import Control
import json

app = Flask(__name__)
app.config.from_object(DevelopmentConfig)


@app.route('/')
def index():
    return render_template('index.html', gen_text='')


@app.route('/init-model')
def init_model():
    Control.init()
    response = {'status': 200}
    return json.dumps(response)


@app.route('/execute-model', methods=['POST'])
def index_post():
    question = request.form['promptText']
    text = Control.generate_text(question)
    response = {'status': 200, 'generated_text': text}
    return json.dumps(response)


@app.route("/download")
def download_pdf():
    path = os.path.join(current_app.root_path, app.config["FILES_DIRECTORY"])
    filename = "Texto de referencia.txt"
    return send_from_directory(path, filename, as_attachment=True)


if __name__ == '__main__':
    app.run(port=8090, host='0.0.0.0')

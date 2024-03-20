from flask import Flask, request, render_template, Response, send_file, jsonify, redirect, url_for
import os
import json
from flask_bootstrap import Bootstrap
import pandas as pd
from utils import gen
from pathlib import Path
import cv2



basedir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__)
Bootstrap(app)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/id_cap', methods=["GET", "POST"])
def id_cap():
    user_name = request.args.get('name', 'default')
    return render_template("id_cap.html", user_name=user_name)


@app.route('/video_feed')
def video_feed():
    user_name = request.args.get('name', 'default')
    print("Nombre recibido:", user_name)  # Para depurar
    return Response(gen(user_name), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/capture_data', methods=["GET", "POST"])
def capture_data():
    if request.method == "POST":
        # Aquí puedes procesar y almacenar los datos del formulario si es necesario
        user_name = request.form.get('name', 'default')
        # Redirigir a la página de captura de imagen con el nombre del usuario
        return redirect(url_for('id_cap', name=user_name))
    return render_template("capture_data.html")


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

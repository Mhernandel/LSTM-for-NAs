from flask import Flask, request, render_template, redirect, url_for, send_from_directory, Response
import os
import sys
sys.path.append('/Users/majohernandezdelprado/Desktop/Escuela/Makers/Machile learning repos/lib')
from lib.prepareData import PrepareData
from lib.Model import train_and_save_model
from lib.results import load_and_predict, plot_results, save_predictions_to_csv
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt


app = Flask(__name__, static_folder='static')

UPLOAD_FOLDER = 'Data'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/downloads/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'dataset' not in request.files:
        return redirect(request.url)
    file = request.files['dataset']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        predictions_filename = 'predictions_' + filename
        predictions_path = os.path.join(app.config['UPLOAD_FOLDER'], predictions_filename)

        train_and_save_model(filepath, 'lib/trained_model.h5')
        
        data, predicted_values = load_and_predict(filepath, 'lib/trained_model.h5')
        save_predictions_to_csv(data, predicted_values, predictions_path)
        plot_url = plot_results(data, predicted_values) 
        data_html = data.to_html(classes='table table-striped')
        return render_template("results.html", plot_url=plot_url, download_link=predictions_filename, data_html=data_html, message="File processed successfully!")

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)

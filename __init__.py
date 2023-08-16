from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename


UPLOAD_FOLDER = r"D:/Research Dr. Stanley (MSU)/VRP_Flask/VRP_flask/static/uploads"
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

from VRP_flask import routes


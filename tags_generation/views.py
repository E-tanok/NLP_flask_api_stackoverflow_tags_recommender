from flask import Flask, render_template, url_for, request
import json
from urllib import parse
from sqlalchemy import create_engine
import os

from functions import *
from preprocess import normalize_and_clean
#from context import learning_models_path

import pandas as pd
app = Flask(__name__)
app.config.from_object('config')


from . import models
from config import SQLALCHEMY_DATABASE_URI

@app.route('/', methods =['GET','POST'])
@app.route('/?', methods =['GET','POST'])
@app.route('/index/', methods =['GET','POST'])
def index(message_corpus=None,results=None):
    if request.method == 'POST':
        message_corpus= request.form['message_corpus']
        print(message_corpus)

        if len(message_corpus)>0:
            print()
            results = normalize_and_clean(message_corpus)

    return render_template('index.html',message_corpus=message_corpus,results=results)

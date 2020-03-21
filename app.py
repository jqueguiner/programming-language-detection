import os
import sys
import subprocess
import requests
import ssl
import random
import string
import json

from flask import jsonify
from flask import Flask
from flask import request
import traceback

from app_utils import download
from app_utils import generate_random_filename
from app_utils import clean_me
from app_utils import clean_all
from app_utils import create_directory
from app_utils import get_model_bin
from app_utils import get_multi_model_bin


import data_helper
import keras
import tensorflow as tf
import numpy as np
import defs


try:  # Python 3.5+
    from http import HTTPStatus
except ImportError:
    try:  # Python 3
        from http import client as HTTPStatus
    except ImportError:  # Python 2
        import httplib as HTTPStatus


app = Flask(__name__)


@app.route("/detect", methods=["POST"])
def detect():

    input_path = generate_random_filename(upload_directory, "jpg")

    try:

        url = request.json["url"]

        download(url, input_path)

        results = []

        x = np.array(
            data_helper.turn_file_to_vectors(
                input_path, 
                file_vector_size=defs.file_characters_truncation_limit, 
                breakup=False
                )
            )

        with graph.as_default():
            y = model.predict(x)
            result = model.predict_proba(x)



        for i in range(0, len(defs.langs)):
            if (y[0][i] > 0.5):
                results.append({"language": defs.langs[i], "score": round(100 * y[0][i])})

        return json.dumps(results), 200


    except:
        traceback.print_exc()
        return {'message': 'input error'}, 400


    finally:
        clean_all([
            input_path
        ])

if __name__ == '__main__':
    global model, graph

    upload_directory = '/src/upload/'
    create_directory(upload_directory)

    model_directory = '/src/models/'
    create_directory(model_directory)

    moodel_url_prefix = "http://pretrained-models.auth-18b62333a540498882ff446ab602528b.storage.gra.cloud.ovh.net/text/programming-language/"
    get_model_bin(moodel_url_prefix + "save_tmp.h5", model_directory + "save_tmp.h5")

    model = keras.models.load_model(model_directory + "save_tmp.h5")
    graph = tf.get_default_graph()


    port = 5000
    host = '0.0.0.0'

    app.run(host=host, port=port, threaded=True)



import os

import numpy as np
import Integrated as model

from flask import Flask
from flask import request
from flask import render_template
from flask import jsonify

app = Flask(__name__)
UPLOAD_FOLDER = "./static"



@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", prediction=0, image_loc=None)

@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html", prediction=0, image_loc=None)

@app.route("/predict", methods=["POST"])
def predict():
    add_title = request.form.getlist('skill[]')
    add_title_n = request.form.getlist('number[]')
    # print(add_title)
    if(len(add_title)!=0 and len(add_title_n)!=0):
        for i in range(len(add_title)):
            if(len(add_title[i])!=0 and len(add_title_n[i])!=0):
                model.search(str(add_title[i]),int(add_title_n[i]),debug=False)
        # msg = 'New record created successfully'
        # jsonify(msg)
        return render_template("index.html", prediction=0, image_loc=None)
    else:
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            if(not os.path.exists(image_location)):
                image_file.save(image_location)
            pred=model.prediction(image_location,debug=False)

            x=render_template("test.html", prediction=pred, image_loc=image_file.filename)
            return x
        # add_loop= int(request.form.get("loop"))
        # for i in range(add_loop):
# @app.route("/postskill",methods=["POST","GET"])
# def postskill():
#     if request.method == 'POST':
#         skills = request.form.getlist('skill[]')
#         for value in skills:  
#             print(value)

#         msg = 'New record created successfully'    
#     return jsonify(msg)
        
        
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=12000, debug=True)

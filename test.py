from flask import Flask,render_template
import pandas as pd

app=Flask(__name__)

@app.route('/',methods=['GET'])
def hello_world():
    data = "youtube comment testing"
    return render_template('index.html',data="test")


if __name__=='__main__':
    app.run(port=3000,debug=True)


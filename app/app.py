from flask import Flask, render_template, request


app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/train_model", methods=['GET', 'POST'])
def train():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        pass




if __name__=="__main__":
    app.run(port=5000, debug=True)

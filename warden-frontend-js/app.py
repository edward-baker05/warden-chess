from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def index():
    with open('index.html') as f:
        return f.read()

@app.route('../move', methods=['POST'])
def handle_string():
    string = request.form['str']
    # Do something with the string here
    if string == "Hello World!":
        return "Success"
    return 'Failure'

if __name__ == '__main__':
    app.run()
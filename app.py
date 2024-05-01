from flask import Flask, render_template, request
from fullcode import find_similar_movies

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/find_similar_movies', methods=['POST'])
def find_similar_movies_route():
    input_description = request.form['input_description']
    similar_movies_data = find_similar_movies(input_description)
    return render_template('result.html', similar_movies_data=similar_movies_data)

if __name__ == '__main__':
    app.run(debug=True)

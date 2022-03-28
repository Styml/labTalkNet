from flask import render_template
from app import app

@app.route('/')
@app.route('/start')
def start():
    return render_template('start.html', name=None)
   # return "Hello world!"

@app.route('/about')
def about():
    return render_template('about.html', name=None)

@app.route('/main')
def main():
    return render_template('main.html', name=None)

@app.route("/post", methods=["POST"])
def upvote():
    global votes
    votes = votes + 1
    return str(votes)
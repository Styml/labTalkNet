from flask import render_template
from app import app
from flask import send_file

@app.route('/')
@app.route('/start', methods=['GET', 'POST'])
def start():
    return render_template('start.html', name=None)
   # return "Hello world!"

@app.route('/about')
def about():
    return render_template('about.html', name=None)

@app.route('/main')
def main():
    return render_template('main.html', name=None)

@app.route('/wavfile')
def view_file():
     path_to_file = "output.wav"
     return send_file(
         path_to_file, 
         mimetype="audio/wav", 
         #as_attachment=True, 
         attachment_filename="output.wav")
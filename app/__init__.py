from flask import Flask, render_template, send_file, request
#from app.neuralnet import * 

from scipy.io.wavfile import write

#spec_generator, voc = init()

app = Flask(__name__)
app.config['DEBUG'] = True


@app.route('/')
@app.route('/main')
def main():
    return render_template('main.html', name=None)

@app.route('/start')
def start():
    return render_template('start.html', name=None)
   # return "Hello world!"

@app.route('/about')
def about():
    return render_template('about.html', name=None)

@app.route('/wavfile', methods=['POST'])
def generate_audio():
    text_to_gen = request.form.get('message')
    print(text_to_gen)
    #_, audio = infer(spec_generator, voc, text_to_gen)
    path_to_file = '../sample.wav'
  #  write(path_to_file, 22050, audio)
    #return send_file(
     #   path_to_file,
      #  mimetype="audio/wav",
        #as_attachment=True,
       # attachment_filename="sample.wav")
    return render_template('start.html', path_to_file=path_to_file)

'''@app.route('/wavfile', methods=['POST'])
def generate_audio():
    text_to_gen = request.form.get('message')
    _, audio = infer(spec_generator, voc, text_to_gen)
    path_to_file = 'static/output.wav'
    write(path_to_file, 22050, audio)
    return send_file(
        path_to_file,
        mimetype="audio/wav",
        #as_attachment=True,
        attachment_filename="output.wav")'''
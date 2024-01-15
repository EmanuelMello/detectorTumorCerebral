from flask import Flask, render_template, request, flash, redirect, url_for
import os
from werkzeug.utils import secure_filename
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.models import load_model

app = Flask(__name__, template_folder='template')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['MODEL_PATH'] = 'D:/Dowloads/cnn-novo/cnn/modelTreinado.h5'
app.config['CLASSES'] = {0: 'erro', 1: 'meningioma', 2: 'no_tumor', 3: 'pituitary'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def processar_imagem(imagem_path):
    try:
        nova_imagem = image.load_img(imagem_path, target_size=(64, 64))
        nova_imagem = image.img_to_array(nova_imagem)
        nova_imagem = np.expand_dims(nova_imagem, axis=0)
        nova_imagem /= 255.0

        print(f"Shape da imagem: {nova_imagem.shape}")
        
        resultado = modelo.predict(nova_imagem)
        print(f"Resultado da predição: {resultado}")

        indice_classe_predita = np.argmax(resultado)
        print(f"Índice da classe predita: {indice_classe_predita}")

        classe_predita = app.config['CLASSES'][indice_classe_predita]
        print(f"Classe predita: {classe_predita}")

        return classe_predita
    except Exception as e:
        flash(f"Erro ao processar imagem: {str(e)}", 'error')
        return 'erro'


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
# ...

def upload_file():
    if 'file' not in request.files:
        flash('Nenhum arquivo enviado.', 'error')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('Nenhum arquivo selecionado.', 'error')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        # Crie o diretório 'uploads' se ele não existir
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Processa a imagem
        resultado = processar_imagem(file_path)
        os.remove(file_path)  # Remove o arquivo após o processamento

        return render_template('index.html', resultado=resultado)

        # Processa a imagem
        resultado = processar_imagem(file_path)
        os.remove(file_path)  # Remove o arquivo após o processamento

        flash(f'A imagem foi classificada como: {resultado}', 'success')
        return redirect(request.url)

    flash('Tipo de arquivo não permitido.', 'error')
    return redirect(request.url)

if __name__ == '__main__':
    modelo = load_model(app.config['MODEL_PATH'])
    app.secret_key = 'super_secret_key'
    app.run(debug=True)

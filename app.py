import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import pickle
from flask import request
from tensorflow.keras.preprocessing.sequence import pad_sequences



app = Flask(__name__)







model = load_model('m.h5')
with open("token.pkl", "rb") as f:
    tokenizer = pickle.load(f)
f.close()



def predict_text(text, next_words):
    global model
    global tokenizer
    dim = model.input_shape[1]
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([text])[0]
        token_list = pad_sequences([token_list], maxlen=dim, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        text += " " + output_word

    return(text)    

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods = ['POST'])
def predict():
    if request.method=='GET':
        return render_template('index.html')
    else:
        text = request.form["text"]
        out = predict_text(text, next_words = 10)
        return render_template('predict.html', result=out, text = text)



if __name__ == "__main__":
    app.run()



import streamlit as st
import spacy
import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import pickle
import tensorflow_hub as hub
from spacy.lang.en import English
nlp  = English()
sentencizer = nlp.add_pipe("sentencizer") # create sentence splitting pipeline object

# Setup the app interface
st.set_page_config(
    page_title="SkimLit",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

tf_hub_embedding_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                        trainable = True,
                                        name = "universal_sentence_encoder")

with open('char_vectorizer.pickle', 'rb') as handle:
        char_vectorizer = pickle.load(handle)

with open('char_embed.pickle', 'rb') as handle:
    char_embed = pickle.load(handle)

def build_model():
    # 1. Setup token inputs/model
    token_inputs = layers.Input(shape = [], dtype = tf.string, name = "token_input")
    token_embeddings = layers.Lambda(lambda x: tf_hub_embedding_layer(x), output_shape = (512,))(token_inputs)
    expand_layer = layers.Lambda(lambda embed: tf.expand_dims(embed, axis = 1))(token_embeddings)
    lstm = layers.Bidirectional(layers.LSTM(128), name = 'bidirection_lstm')(expand_layer)
    drop = layers.Dropout(0.25)(lstm)
    token_output = layers.Dense(128, activation = "relu")(drop)
    token_model = tf.keras.Model(inputs = token_inputs,
                                outputs = token_output)

    # 2. Setup char inputs/model
    char_inputs = layers.Input(shape = (1,), dtype = tf.string, name = "char_input")
    char_vectors = char_vectorizer(char_inputs)
    char_embeddings = char_embed(char_vectors)
    char_bi_lstm = layers.Bidirectional(layers.LSTM(25))(char_embeddings)
    char_model = tf.keras.Model(inputs = char_inputs,
                                outputs = char_bi_lstm)

    # 3. Line numbers inputs
    line_number_inputs = layers.Input(shape = (15,), dtype = tf.int32, name = "line_number_input")
    x = layers.Dense(32, activation = "relu")(line_number_inputs)
    line_number_model = tf.keras.Model(inputs = line_number_inputs,
                                    outputs = x)

    # 4. Total lines inputs
    total_lines_inputs = layers.Input(shape = (20,), dtype = tf.int32, name = "total_lines_input")
    y = layers.Dense(32, activation = "relu")(total_lines_inputs)
    total_line_model = tf.keras.Model(inputs = total_lines_inputs,
                                    outputs = y)

    # 5. Combine token and char embeddings into a hybrid embedding
    combined_embeddings = layers.Concatenate(name = "token_char_hybrid_embedding")([token_model.output, 
                                                                                char_model.output])
    z = layers.Dense(256, activation = "relu")(combined_embeddings)
    z = layers.Dropout(0.5)(z)

    # 6. Combine positional embeddings with combined token and char embeddings into a tribrid embedding
    z = layers.Concatenate(name = "token_char_positional_embedding")([line_number_model.output,
                                                                    total_line_model.output,
                                                                    z])

    # 7. Create output layer
    output_layer = layers.Dense(5, activation = "softmax", name = "output_layer")(z)

    # 8. Put together model
    model_5 = tf.keras.Model(inputs = [line_number_model.input,
                                    total_line_model.input,
                                    token_model.input, 
                                    char_model.input],
                            outputs = output_layer)
    return model_5

    
model = build_model()

# Load the model and pipeline
@st.cache_resource
def load_model_and_pipeline():
    # Load the saved model and pipeline (adapt paths as necessary)
    model.load_weights("model_5.weights.h5")

    with open('encoder.pickle', 'rb') as handle:
        encoder = pickle.load(handle)

    return model, encoder

model, encoder = load_model_and_pipeline()

# Preprocessing functions
# def preprocess_text(text, stopwords):
#     """Conditional preprocessing on the text unique to the task."""
#     text = text.lower()
#     pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
#     text = pattern.sub("", text)
#     text = re.sub(r"\([^)]*\)", "", text)
#     text = re.sub(r"([-;;.,!?<=>])", r" \1 ", text)
#     text = re.sub("[^A-Za-z0-9]+", " ", text)
#     text = re.sub(" +", " ", text)
#     text = text.strip()
#     return text

def split_chars(text):
    return " ".join(list(text))

# Main prediction function
def predict_abstract(abstract_lines):
    total_lines_in_sample = len(abstract_lines)
    sample_lines = []

    for i, line in enumerate(abstract_lines):
        sample_dict = {}
        sample_dict["text"] = str(line)
        sample_dict["line_number"] = i
        sample_dict["total_lines"] = total_lines_in_sample - 1
        sample_lines.append(sample_dict)

    # One-hot encoding
    line_numbers = [line["line_number"] for line in sample_lines]
    total_lines = [line["total_lines"] for line in sample_lines]
    abstract_chars = [split_chars(sentence) for sentence in abstract_lines]

    line_numbers_one_hot = tf.one_hot(line_numbers, depth=15)
    total_lines_one_hot = tf.one_hot(total_lines, depth=20)

    # Model prediction
    pred_probs = model.predict(x=(line_numbers_one_hot,
                                  total_lines_one_hot,
                                  tf.constant(abstract_lines),
                                  tf.constant(abstract_chars)))
    preds = tf.argmax(pred_probs, axis=1)
    pred_classes = [encoder.classes_[i] for i in preds]

    return sample_lines, pred_classes


st.title('SkimLit📄🔥')
st.caption('An NLP model to classify abstract sentences into roles (e.g. objective, methods, results, etc.) to enable researchers to skim the literature.')

col1, col2 = st.columns(2)

with col1:
    st.write('#### Enter Abstract Here:')
    abstract = st.text_area(label='', height=200)

    predict = st.button('Extract')

# Logic for prediction
if predict and abstract:
    with st.spinner('Predicting...'):
        abstract_lines = nlp(abstract)
        abstract_lines = [str(sent) for sent in list(abstract_lines.sents)] 
        processed_lines, predictions = predict_abstract(abstract_lines)
    print(predictions)

    with col2:
        categories = ['OBJECTIVE', 'BACKGROUND', 'METHODS', 'RESULTS', 'CONCLUSIONS']
        outputs = {category: '' for category in categories}

        for line, pred_class in zip(abstract_lines, predictions):
            outputs[pred_class] += f"{line}\n"

        for category in categories:
            st.markdown(f"### {category}:")
            st.write(outputs[category])

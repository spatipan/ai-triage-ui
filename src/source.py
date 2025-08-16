
# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# Pipeline
from sklearn.pipeline import Pipeline, make_pipeline

# Data preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, FunctionTransformer, PowerTransformer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base import BaseEstimator, TransformerMixin

# Compose
from sklearn.compose import make_column_selector, make_column_transformer

# Training
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Embedding
import tensorflow_hub as hub
import tensorflow_text

# Class weights calculation for imbalanced data
import tensorflow.keras.backend as K
from sklearn.utils.class_weight import compute_class_weight

# keras tuner
import keras_tuner as kt
from tensorflow import keras

class TriageModel:
    def __init__(self):
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.history = None
        self.evaluation_results = None
        self.predictions = None

        self.metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                        tf.keras.metrics.AUC(name='auc'),
                        tf.keras.metrics.AUC(name='prc', curve='PR'),
                        tf.keras.metrics.Precision(name='precision'),
                        tf.keras.metrics.Recall(name='recall'),]

        self.parameters = {
            'learning_rate': 0.0012,
            'dropout_rate': 0.1,
            'num_hidden_layers_text': 1,
            'num_neurons_text': 128,
            'num_hidden_layers_num': 1,
            'num_neurons_num': 64,
            'num_hidden_layers_concat': 2,
            'num_neurons_concat': 32,
        }

        self.class_weights = None


    def set_parameters(self, parameters):
        self.parameters = parameters

    def import_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.model.summary()

    def create_model(self):
        
        assert self.train_dataset is not None, "need to call method 'import_data()' first"
        assert self.val_dataset is not None, "need to call method 'import_data()' first"
        assert self.test_dataset is not None, "need to call method 'import_data()' first"

        input_text = tf.keras.layers.Input(shape=(self.train_dataset.element_spec[0]['text'].shape[-1],), name='text')
        text = input_text
        for i in range(self.parameters['num_hidden_layers_text']):
            text = tf.keras.layers.Dense(self.parameters['num_neurons_text'], activation='relu')(text)
            text = tf.keras.layers.BatchNormalization()(text)
            text = tf.keras.layers.Dropout(self.parameters['dropout_rate'])(text)

        input_num = tf.keras.layers.Input(shape=(self.train_dataset.element_spec[0]['num'].shape[-1],), name='num')
        norm_num = tf.keras.layers.Normalization()(input_num)
        num = norm_num
        for i in range(self.parameters['num_hidden_layers_num']):
            num = tf.keras.layers.Dense(self.parameters['num_neurons_num'], activation='relu')(num)
            num = tf.keras.layers.BatchNormalization()(num)
            num = tf.keras.layers.Dropout(self.parameters['dropout_rate'])(num)

        concat = tf.keras.layers.Concatenate()([num, text])
        for i in range(self.parameters['num_hidden_layers_concat']):
            concat = tf.keras.layers.Dense(self.parameters['num_neurons_concat'], activation='relu')(concat)
            concat = tf.keras.layers.BatchNormalization()(concat)
            concat = tf.keras.layers.Dropout(self.parameters['dropout_rate'])(concat)

        # if use_wide_num:
        #     concat = tf.keras.layers.Concatenate()([concat, norm_num])

        output = tf.keras.layers.Dense(self.train_dataset.element_spec[1].shape[-1], activation='sigmoid', name='output')(concat)
        model = tf.keras.Model(inputs=[input_num, input_text], outputs=output)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = self.parameters['learning_rate']),
                    loss='binary_crossentropy',
                    metrics=self.metrics)

        self.model = model
        self.model.summary()

    def load_weights(self, weights_path):
        assert self.model is not None, "need to call method 'import_model() / create_model()' first"
        self.model.load_weights(weights_path)


    def import_data(self, train_dataset, val_dataset, test_dataset, classweights = None):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.class_weights = classweights

    def train(self, epochs = 200, batch_size = 32):
        assert self.model is not None, "need to call method 'import_model()' first"
        assert self.train_dataset is not None, "need to call method 'import_data()' first"
        assert self.val_dataset is not None, "need to call method 'import_data()' first"
        assert self.test_dataset is not None, "need to call method 'import_data()' first"


        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='auto',
            verbose=1,
            patience=20,
            baseline=None,
            restore_best_weights=True,
            )

        self.history = self.model.fit(self.train_dataset, validation_data=self.val_dataset, epochs = epochs, batch_size = batch_size,
                                callbacks = [early_stopping, tensorboard_callback])

    def evaluate(self):
        assert self.model is not None, "need to call method 'import_model()' first"
        assert self.test_dataset is not None, "need to call method 'import_data()' first"

        self.evaluation_results = self.model.evaluate(self.test_dataset)
        for i in self.metrics:
            print(i.name, i.result().numpy())

    def save_model(self):
        assert self.model is not None, "need to call method 'import_model()' first"
        assert self.evaluation_results is not None, "need to call method 'evaluate_model()' first"
        assert self.history is not None, "need to call method 'train()' first"

        self.model.save_weights('./weights.weights.h5')
        with open('./evaluation_results.txt', 'w') as f:
            f.write(str(self.evaluation_results))
        with open('./history.txt', 'w') as f:
            f.write(str(self.history.history))
        self.model.save('./model.keras')


    def predict(self):
        assert self.model is not None, "need to call method 'import_model()' first"
        assert self.test_dataset is not None, "need to call method 'import_data()' first"
        self.predictions = self.model.predict(self.test_dataset)

    def hyperparameter_tuning(self):
        pass

    def download(self):
        files.download('./weights.weights.h5')
        files.download('./evaluation_results.txt')
        files.download('./history.txt')
        files.download('./model.keras')
        files.download ('/predictions.csv')


# Custom transformer for text embedding
class TextEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, embedder_url):
        assert embedder_url is not None, "embedder_url must be set"
        self.embedder_url = embedder_url
        self.embedder = None
        self._load_model()

    def _load_model(self):
        if self.embedder is None:
            self.embedder = hub.load(self.embedder_url)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        embeddings = []
        for text in X:
            embeddings.append(self.embedder(str(text)))
        return np.array(embeddings)


class DataPreprocessing:
    def __init__(self):
        self.train = None
        self.val = None
        self.test = None
        self.x_num_cols = None
        self.x_text_cols = None
        self.y_cols = None

        self.num_preprocessor = None
        self.text_embedder_path = '/Users/patipansittiprawiat/.cache/kagglehub/models/google/universal-sentence-encoder/tensorFlow2/multilingual-large/2'
        self.text_embedder = TextEmbedder(self.text_embedder_path)

    def import_data(self, data, x_num_cols, x_text_cols, y_cols):
        train, val, test = data
        self.train = train[x_num_cols + x_text_cols + y_cols]
        self.val = val[x_num_cols + x_text_cols + y_cols]
        self.test = test[x_num_cols + x_text_cols + y_cols]
        self.x_num_cols = x_num_cols
        self.x_text_cols = x_text_cols
        self.y_cols = y_cols

    def fit(self, data, x_num_cols, x_text_cols, y_cols):
        # [1] Proprocess numerical data (impute missing values, scale data)
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ("power", PowerTransformer(method="yeo-johnson", standardize=True)),
            ('scaler', StandardScaler())
        ])

        # [2] Preprocess categorical data (impute missing values, encode data)
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder())
        ])

        # [3] Preprocess text data (impute missing values, tokenize, embed)
        text_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=' ')),
        ])

        # Configure ColumnTransformers with training data
        self.num_preprocessor = ColumnTransformer([
            ('num', num_pipeline, [col for col in x_num_cols if data[col].dtype in ['int64', 'float64']]),
            ('cat', cat_pipeline, [col for col in x_num_cols if data[col].dtype == 'object' and col not in x_text_cols]),
        ])
        self.text_preprocessor = ColumnTransformer([
            ('text', text_pipeline, x_text_cols)
        ])

        # Fit transformers on the training data
        self.num_preprocessor.fit(data[x_num_cols])
        self.text_preprocessor.fit(data[x_text_cols])

        self.x_num_cols = x_num_cols
        self.x_text_cols = x_text_cols
        self.y_cols = y_cols

    def transform(self, data):
        # Transform new data using the already-fitted transformers
        num_features_processed = self.num_preprocessor.transform(data[self.x_num_cols])
        text_features_processed = self.text_preprocessor.transform(data[self.x_text_cols])
        text_features_processed = self.text_embedder.transform(text_features_processed)[:, 0, :]
        print(text_features_processed.shape)
        return num_features_processed, text_features_processed

    def convert_to_dataset(self, data, batch_size=32, prefetch=tf.data.AUTOTUNE):
        num_features_processed, text_features_processed = self.transform(data)
        targets = data[self.y_cols]
        dataset = tf.data.Dataset.from_tensor_slices(({'num': num_features_processed, 'text': text_features_processed}, targets)).batch(batch_size).prefetch(prefetch)
        return dataset

    def _process(self):
        # Fit transformers on the training data
        self.fit(self.train, self.x_num_cols, self.x_text_cols, self.y_cols)

        train_dataset = self.convert_to_dataset(self.train)
        val_dataset = self.convert_to_dataset(self.val)
        test_dataset = self.convert_to_dataset(self.test)

        return train_dataset, val_dataset, test_dataset
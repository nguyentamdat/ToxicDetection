import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from pickle import dump, load


class ToxicDetection:
    def __init__(self):
        self.class_names = ['toxic', 'severe_toxic', 'obscene',
                            'threat', 'insult', 'identity_hate']
        print("Start load word vectorizer")
        self.word_vectorizer = load(open('word.pkl', 'rb'))
        self.word_vectorizer:TfidfVectorizer
        print("Load word vectorizer success")
        self.models = load(open('models.pkl', 'rb'))

    def predict(self, doc: str):
        doc = [doc]
        doc_feature = self.word_vectorizer.transform(doc)
        doc_feature = hstack([doc_feature])
        res = {}
        for class_name in self.class_names:
            res[class_name] = self.models[class_name].predict_proba(doc_feature)[:, 1][0]
        return res
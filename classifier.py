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
        self.class_vn = ['clean', 'offensive', 'hate']
        print("Start load word vectorizer")
        self.word_vectorizer = load(open('word.pkl', 'rb'))
        self.word_vectorizer:TfidfVectorizer
        print("Load word vectorizer success")
        self.models = load(open('models.pkl', 'rb'))

        self.models_vn = load(open('models_vn.pkl', 'rb'))
        self.word_vn = load(open('words_vn.pkl', 'rb'))
        self.char_vn = load(open('chars_vn.pkl', 'rb'))

    def predict(self, doc: str):
        doc = [doc]
        doc_feature = self.word_vectorizer.transform(doc)
        doc_feature = hstack([doc_feature])
        res = False
        for class_name in self.class_names:
            res |= self.models[class_name].predict_proba(doc_feature)[:, 1][0] > 0.9
        return 'toxic' if res else 'clean'

    def predict_vn(self, doc: str):
        doc = [self.preprocess(doc)]
        word_feature = self.word_vn.transform(doc)
        char_feature = self.char_vn.transform(doc)
        doc_features = hstack([char_feature, word_feature])
        res = {}
        for class_name in self.class_vn:
            res[class_name] = self.models_vn[class_name].predict_proba(doc_features)[:, 1][0]
        print(res)
        if res['hate'] > 0.25:
            res = 'hate'
        elif res['offensive'] > 0.25:
            res = 'offensive'
        elif res['clean'] < 0.9:
            res = 'not clean'
        else:
            res = 'clean'
        return res

    def preprocess(self, doc: str):
        doc = doc.lower()
        import re
        doc = re.sub(r"http\S+", "", doc)
        return doc
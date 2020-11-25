import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from pickle import dump, load

# class_names = ['toxic', 'severe_toxic', 'obscene',
#                'threat', 'insult', 'identity_hate']

class_names = ['clean', 'offensive', 'hate']

# train = pd.read_csv('./train.csv').fillna(' ')
# test = pd.read_csv('./test.csv').fillna(' ')

train_text_data_df = pd.read_csv("./vn_hate_speech/02_train_text.csv", quotechar='"', sep=",")
train_label_data_df = pd.read_csv("./vn_hate_speech/03_train_label.csv", quotechar='"', sep=",")
test_text_data_df = pd.read_csv("./vn_hate_speech/04_test_text.csv", quotechar='"', sep=",")
train_set = pd.merge(train_text_data_df, train_label_data_df, on='id', how='left')

def preprocess(doc: str):
    doc = doc.lower()
    import re
    doc = re.sub(r"http\S+", "", doc)
    return doc

train_set['clean'] = (train_set['label_id'] == 0).astype(int)
train_set['offensive'] = (train_set['label_id'] == 1).astype(int)
train_set['hate'] = (train_set['label_id'] == 2).astype(int)
train_text = train_set['free_text'].map(lambda x: preprocess(x))
test_text = test_text_data_df['free_text'].map(lambda x: preprocess(x))

# train_text = train['comment_text']
# test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    ngram_range=(1, 1),
    max_features=20000)
word_vectorizer.fit(all_text)
print("Start load word vectorizer")
# word_vectorizer = load(open('words_vn.pkl', 'rb'))
print("Load word vectorizer success")
print("Start transform word train")
train_word_features = word_vectorizer.transform(train_text)
print("Start transform word test")
test_word_features = word_vectorizer.transform(test_text)
print("Finish transform word")
dump(word_vectorizer, open("words_vn.pkl", "wb"))

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    ngram_range=(2, 6),
    max_features=50000)
char_vectorizer.fit(all_text)
print("Start load char vectorizer")
# char_vectorizer = load(open('chars_vn.pkl', 'rb'))
print("Load char vectorizer success")
print("Start transform char train")
train_char_features = char_vectorizer.transform(train_text)
print("Start transform char test")
test_char_features = char_vectorizer.transform(test_text)
print("Finish transform char")
dump(char_vectorizer, open("chars_vn.pkl", "wb"))


train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])

# train_features = hstack([train_word_features])
# test_features = hstack([test_word_features])

scores = []
# submission = pd.DataFrame.from_dict({'id': test['id']})
models = {}
# models = load(open('models.pkl', 'rb'))
for class_name in class_names:
    train_target = train_set[class_name]
    classifier = LogisticRegression(C=0.1, solver='sag')
    # classifier = models[class_name]
    cv_score = np.mean(cross_val_score(
        classifier, train_features, train_target, cv=3, scoring='roc_auc'))
    scores.append(cv_score)
    print('CV score for class {} is {}'.format(class_name, cv_score))

    classifier.fit(train_features, train_target)
    models[class_name] = classifier
    # submission[class_name] = classifier.predict_proba(test_features)[:, 1]

dump(models, open('models_vn.pkl', 'wb'))

print('Total CV score is {}'.format(np.mean(scores)))

# submission.to_csv('submission.csv', index=False)

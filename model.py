import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from pickle import dump, load

class_names = ['toxic', 'severe_toxic', 'obscene',
               'threat', 'insult', 'identity_hate']

train = pd.read_csv('./train.csv').fillna(' ')
test = pd.read_csv('./test.csv').fillna(' ')

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])

# word_vectorizer = TfidfVectorizer(
#     sublinear_tf=True,
#     strip_accents='unicode',
#     analyzer='word',
#     token_pattern=r'\w{1,}',
#     stop_words='english',
#     ngram_range=(1, 1),
#     max_features=10000)
# word_vectorizer.fit(all_text)
print("Start load word vectorizer")
word_vectorizer = load(open('word.pkl', 'rb'))
print("Load word vectorizer success")
print("Start transform word train")
train_word_features = word_vectorizer.transform(train_text)
print("Start transform word test")
test_word_features = word_vectorizer.transform(test_text)
# dump(word_vectorizer, open("word.pkl", "wb"))
print("Finish transform word")

# char_vectorizer = TfidfVectorizer(
#     sublinear_tf=True,
#     strip_accents='unicode',
#     analyzer='char',
#     ngram_range=(2, 6),
#     max_features=50000)
# char_vectorizer.fit(all_text)
# print("Start load char vectorizer")
# char_vectorizer = load(open('char.pkl', 'rb'))
# print("Load char vectorizer success")
# print("Start transform char train")
# train_char_features = char_vectorizer.transform(train_text)
# print("Start transform char test")
# test_char_features = char_vectorizer.transform(test_text)
# dump(char_vectorizer, open("char.pkl", "wb"))
# print("Finish transform char")


# train_features = hstack([train_char_features, train_word_features])
# test_features = hstack([test_char_features, test_word_features])

train_features = hstack([train_word_features])
test_features = hstack([test_word_features])

scores = []
submission = pd.DataFrame.from_dict({'id': test['id']})
# models = {}
models = load(open('models.pkl', 'rb'))
for class_name in class_names:
    train_target = train[class_name]
    # classifier = LogisticRegression(C=0.1, solver='sag')
    classifier = models[class_name]
    cv_score = np.mean(cross_val_score(
        classifier, train_features, train_target, cv=3, scoring='roc_auc'))
    scores.append(cv_score)
    print('CV score for class {} is {}'.format(class_name, cv_score))

    classifier.fit(train_features, train_target)
    models[class_name] = classifier
    # submission[class_name] = classifier.predict_proba(test_features)[:, 1]

# dump(models, open('models.pkl', 'wb'))

print('Total CV score is {}'.format(np.mean(scores)))

# submission.to_csv('submission.csv', index=False)

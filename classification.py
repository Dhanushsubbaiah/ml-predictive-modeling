import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score,
roc_curve, auc
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
df = pd.read_csv('emails.csv')
# Initialize a Porter stemmer
stemmer = PorterStemmer()
# Define a function to preprocess the text
def preprocess_text(text):
 # Convert to lowercase
 text = text.lower()
 # Remove special characters
 text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
 # Tokenize
 words = word_tokenize(text)
 # Remove stopwords and stem
 words = [stemmer.stem(word) for word in words if word not in
stopwords.words('english')]
 return ' '.join(words)
# Apply the preprocessing to the 'text' column
df['text'] = df['text'].apply(preprocess_text)
# Vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['spam']
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)
# Train a Logistic Regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)
# Train a Naive Bayes model
nb = MultinomialNB()
nb.fit(X_train, y_train)
# Evaluate the models
for model in [lr, nb]:
 y_pred = model.predict(X_test)
 print(f'Confusion Matrix for {model.__class__.__name__}:')
 print(confusion_matrix(y_test, y_pred))
 print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
 print(f'Precision: {precision_score(y_test, y_pred)}')
 # ROC curve
 fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
 roc_auc = auc(fpr, tpr)
 plt.figure()
 plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc})')
 plt.plot([0, 1], [0, 1], 'k--')
 plt.xlim([0.0, 1.0])
 plt.ylim([0.0, 1.05])
 plt.xlabel('False Positive Rate')
 plt.ylabel('True Positive Rate')
 plt.title(f'Receiver Operating Characteristic for {model.__class__.__name__}')
 plt.legend(loc="lower right")
 plt.show()

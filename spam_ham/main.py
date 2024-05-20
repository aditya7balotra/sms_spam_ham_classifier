# %%
import numpy as np
import pandas as pd

# %%
df = pd.read_csv('ham_spam_text_preprocessed.csv')

# %%
df.sample(5)

# %%
df.shape

# %%
# 1. Data cleaning
# 2. EDA
# 3. Text Preprocessing
# 4. Model building
# 5. Evaluation
# 6. Improvement
# 7. Website
# 8. Deploy

# %% [markdown]
# ## 1. Data Cleaning

# %%
df.info()

# %%
# drop last 3 cols
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)

# %%
df.sample(5)

# %%
# renaming the cols
df.rename(columns={'v1':'target','v2':'text'},inplace=True)
df.sample(5)

# %%
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

# %%
df['target'] = encoder.fit_transform(df['target'])

# %%
df.head()

# %%
# missing values
df.isnull().sum()

# %%
# check for duplicate values
df.duplicated().sum()

# %%
# remove duplicates
df = df.drop_duplicates(keep='first')

# %%
df.duplicated().sum()

# %%
df.shape

# %% [markdown]
# ## 2.EDA

# %%
df.head()

# %%
df['target'].value_counts()

# %%
import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(), labels=['ham','spam'],autopct="%0.2f")
plt.show()

# %%
# Data is imbalanced

# %%
import nltk

# %%
!pip install nltk

# %%
nltk.download('punkt')

# %%
df['num_characters'] = df['text'].apply(len)

# %%
df.head()

# %%
# num of words
df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))

# %%
df.head()

# %%
df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))

# %%
df.head()

# %%
df[['num_characters','num_words','num_sentences']].describe()

# %%
# ham
df[df['target'] == 0][['num_characters','num_words','num_sentences']].describe()

# %%
#spam
df[df['target'] == 1][['num_characters','num_words','num_sentences']].describe()

# %%
import seaborn as sns

# %%
plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_characters'])
sns.histplot(df[df['target'] == 1]['num_characters'],color='red')

# %%
plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_words'])
sns.histplot(df[df['target'] == 1]['num_words'],color='red')

# %%
sns.pairplot(df,hue='target')

# %%
sns.heatmap(df.corr(),annot=True)

# %% [markdown]
# ## 3. Data Preprocessing
# - Lower case
# - Tokenization
# - Removing special characters
# - Removing stop words and punctuation
# - Stemming

# %%
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)

# %%
transform_text("I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.")

# %%
df['text'][10]

# %%
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
ps.stem('loving')

# %%
df['transformed_text'] = df['text'].apply(transform_text)

# %%
df.head()

# %%
from wordcloud import WordCloud
wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')

# %%
spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))

# %%
plt.figure(figsize=(15,6))
plt.imshow(spam_wc)

# %%
ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))

# %%
plt.figure(figsize=(15,6))
plt.imshow(ham_wc)

# %%
df.head()

# %%
spam_corpus = []
for msg in df[df['target'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)
        

# %%
len(spam_corpus)

# %%
from collections import Counter
sns.barplot(pd.DataFrame(Counter(spam_corpus).most_common(30))[0],pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()

# %%
ham_corpus = []
for msg in df[df['target'] == 0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)

# %%
len(ham_corpus)

# %%
from collections import Counter
sns.barplot(pd.DataFrame(Counter(ham_corpus).most_common(30))[0],pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()

# %%
# Text Vectorization
# using Bag of Words
df.head()

# %% [markdown]
# ## 4. Model Building

# %%
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)

# %%
X = tfidf.fit_transform(df['transformed_text']).toarray()

# %%
#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
#X = scaler.fit_transform(X)

# %%
# appending the num_character col to X
#X = np.hstack((X,df['num_characters'].values.reshape(-1,1)))

# %%
X.shape

# %%
y = df['target'].values

# %%
from sklearn.model_selection import train_test_split

# %%
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

# %%
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score

# %%
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

# %%
gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))

# %%
mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))

# %%
bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))

# %%
# tfidf --> MNB

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

# %%
svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)
xgb = XGBClassifier(n_estimators=50,random_state=2)

# %%
clfs = {
    'SVC' : svc,
    'KN' : knc, 
    'NB': mnb, 
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, 
    'AdaBoost': abc, 
    'BgC': bc, 
    'ETC': etc,
    'GBDT':gbdt,
    'xgb':xgb
}

# %%
def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision

# %%
train_classifier(svc,X_train,y_train,X_test,y_test)

# %%
accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)

# %%
performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)

# %%
performance_df

# %%
performance_df1 = pd.melt(performance_df, id_vars = "Algorithm")

# %%
performance_df1

# %%
sns.catplot(x = 'Algorithm', y='value', 
               hue = 'variable',data=performance_df1, kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()

# %%
# model improve
# 1. Change the max_features parameter of TfIdf

# %%
temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,'Precision_max_ft_3000':precision_scores}).sort_values('Precision_max_ft_3000',ascending=False)

# %%
temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_scaling':accuracy_scores,'Precision_scaling':precision_scores}).sort_values('Precision_scaling',ascending=False)

# %%
new_df = performance_df.merge(temp_df,on='Algorithm')

# %%
new_df_scaled = new_df.merge(temp_df,on='Algorithm')

# %%
temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_num_chars':accuracy_scores,'Precision_num_chars':precision_scores}).sort_values('Precision_num_chars',ascending=False)

# %%
new_df_scaled.merge(temp_df,on='Algorithm')

# %%
# Voting Classifier
svc = SVC(kernel='sigmoid', gamma=1.0,probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

from sklearn.ensemble import VotingClassifier

# %%
voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)],voting='soft')

# %%
voting.fit(X_train,y_train)

# %%
y_pred = voting.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))

# %%
# Applying stacking
estimators=[('svm', svc), ('nb', mnb), ('et', etc)]
final_estimator=RandomForestClassifier()

# %%
from sklearn.ensemble import StackingClassifier

# %%
clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)

# %%
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# %%




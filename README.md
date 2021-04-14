# Projekt 5 - Natural-Language-Processing
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/yenvyhh/Natural-Language-Processing/main?filepath=Nlp%20-%20Projekt%205.ipynb)

**Um die Librarys in das Notebook zu importieren, müssen zu Beginn folgende Installationen einmalig durchgeführt werden (wenn für die vorherigen Übungen bereits getan, dann ignorieren):**
-> %conda install pandas 
-> %conda install numpy
-> %conda install sqlalchemy 
-> %conda install lxml
-> %conda install openpyxl 
-> %conda install xlrd 
-> %conda install matplotlib 
-> %conda install seaborn 
-> %conda install scikit-learn - sklearn
--> %conda install pydot
--> %conda install graphviz
--> %pip install pydot
--> %pip install graphviz
--> %pip install six
--> %conda install nltk

**Zu Beginn des Notebooks, werden die installierten Librarys wie folgt importiert:**
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

**Die Daten werden nun importiert, als DataFrame abgespeichert und das Head wird angezeigt:**
yelp = pd.read_csv("Yelp.csv")
yelp.head()
**Nach Ausführung sollten von der importierten Datei die ersten 5 Zeilen mit den Spalten 'business_id', 'date', 'review_id', 'stars', 'text', 'type', 'user_id', 'cool', 'useful' und 'funny' angezeigt werden** 

**Informationen und Details des Data Frames bzw. der Daten anzeigen lassen:**     
yelp.info()
yelp.describe()
**Bei Info wird angezeigt, ob die Spalten einen Float, ein Integer oder ein Object sind. Zu dem wird bei RangeIndex angezeigt, dass es 10000 Einträge gibt. Bei Describe wird ein Dataset der Analyse geprintet. Beispiele hierfür sind der Durchschnittswert, der Minimum- oder Maximum-Wert.

yelp["text length"] = yelp["text"].apply(len)
yelp.head()
**Durch Ausführen der Befehle wird eine neue Spalte hinzugefügt, welche die Textlänge der Spalte "text" anzeigt. 

**Darauffolgend erfolgt eine EXPLORATIVE DATENANALYSE, die durch verschiedene Diagrammvisualisierungen dargestellt werden. Ein Beispiel:**
g= sns.FacetGrid(yelp,col="stars")
g.map(plt.hist,'text length')
**Durch Ausführen der ganzen Befehle werden fünf Histogramme (Balkendiagramme) nebeneinander erstellt. Dabei basierend die Diagramme auf den stars-Bewertungen. D.h. das erste Histogramm repräsentiert 1-Stern, das zweite 2-Sterne usw..


**Im nächsten Schritt werden die Daten nach dem NLP klassifiert. Dazu wird zunächst ein neues Data Frame erstellt**
yelp_class= yelp[(yelp["stars"]==1) | (yelp["stars"]==5)]
yelp_class.head()
**Jetzt sollte ein neues Data Frame zusehen sein, dass die selben Spalten wie das Data Frame "Yelp" hat nur eingeschränkt auf 1-Stern und 5 Sterne.**

**Die Daten werden nun in Trainings- und Test gesplittet. Dazu sollte zunächst definiert werden was das X-Array (Daten mit den Features) und was das y-Array (Daten mit der Zielvariable) ist:** 
X= yelp_class["text"]
y= yelp_class["stars"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

**Nach Erstellung der Train- und Testdaten wird das Modell trainiert. Zuvor wird eine Instanz des Schätzers erstellt:**
nb=MultinomialNB()
nb.fit(X_train,y_train)

**Im Anschluss daran werden die Werte mit**
pred = nb.predict(X_test)
**vorhergesagt. Basierend darauf kann ein Klassifizierungsreport und eine Confusion Matrix für das Modell erstellt werden:**
print(confusion_matrix(y_test, pred))
print("\n")
print(classification_report(y_test,pred))
**Je näher die Werte bei precicion, recall und f1-score an 1 sind, desto genauer sind Auswertungen. **

**Am Ende wird das Modell erneut trainiert - diesmal mit des Testdaten. Dazu wird das Text Processing verwendet. Der TF-IDF wird miteingeführt und eine Pipeline wird verwendet.** 
pipeline= Pipeline([
    ("bow",CountVectorizer()),
    ("tfidf",TfidfTransformer()),
    ("classifier",MultinomialNB())
])
**Nach Ausführen der obigen Befehle wird eine Pipeline erstellt und kann jetzt verwendet werden.**
X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)
pipeline.fit(X_train,y_train)
pred_p= pipeline.predict(X_test)
print(confusion_matrix(y_test, pred_p))
print("\n")
print(classification_report(y_test,pred_p,labels=np.unique(pred_p)))
**Es sollte nun ein Classification und eine Confusion Matrix basierend auf den Testdaten zu sehen sein.**



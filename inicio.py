import email
import re
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk import PorterStemmer
from nltk.corpus import stopwords
from html.parser import HTMLParser
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ValidationCurveDisplay
import numpy as np

from sklearn.metrics import confusion_matrix
import seaborn as sns


import matplotlib.pyplot as plt



#Quitar las Tags HTML del Texto

class HTMLStripper(HTMLParser):
    
    def __init__(self):
        # Inicializamos la clase padre
        super().__init__()
        # Atributo que almacena los datos
        self.data = []
        
    def handle_data(self, data):
        self.data.append(data)


################################################        


class EmailParser:
    
    def parse(self, correo):
        # Obtenemos el cuerpo del correo electronico
        pcorreo = " ".join(self.get_body(correo))
        # Eliminamos los tags HTML
        pcorreo = self.strip_tags(pcorreo)
        # Eliminamos las urls
        pcorreo = self.remove_urls(pcorreo)
        # Transformamos el texto en tokens
        pcorreo = word_tokenize(pcorreo)
        # Eliminamos stopwords
        # Eliminamos puntuation
        # Hacemos stemming
        pcorreo = self.clean_text(pcorreo)
        return " ".join(pcorreo)      
        
    def get_body(self, correo):
        # Definicion una funcion interna que procese el cuerpo
        pcorreo = email.message_from_string(correo)
        return self._parse_body(pcorreo.get_payload())
    
    def _parse_body(self, payload):
        body = []
        if type(payload) is str:
            return [payload]
        elif type(payload) is list:
            for p in payload:
                body += self._parse_body(p.get_payload())
        return body      

    def strip_tags(self, correo):
        html_stripper = HTMLStripper()
        html_stripper.feed(correo)
        return ''.join(html_stripper.data)
    
    def remove_urls(self, correo):
        return re.sub(r"http\S+", "", correo)
    
    def clean_text(self, correo):
        pcorreo = []
        st = PorterStemmer()
        punct = list(punctuation) + ["\n", "\t"]
        for word in correo:
            if word not in stopwords.words('english') and word not in punct:
                # Aplicamos stemming
                pcorreo.append(st.stem(word))
        return pcorreo
    

    ##### FIN de Clases ###################

# Funcion para leer los correos y meterlos en un arreglo
# Parametros de la funcion indice = Ruta del archivo Indice , num= cantidad de correos a Leer .
# El archivos Indice Tiene el siguiente Parametro
#  spam    ../data/inmail.1
# spam Marcado para correo Spam
# ham Marcador par correo Valido 
def leer_correos(indice, num):
    with open(indice, 'r') as f:
        labels = f.read().splitlines()
    # Leemos los correos de disco
    X = []  # Correos electronicos como tal
    y = []   # Meter las Etiquetas
    for l in labels[:num]:
        label, email_path = l.split(' ../')
        y.append(label)
        with open(email_path, errors='ignore') as f: # Ignoramos los errores
            X.append(f.read())  # Meter el email a la lista X
    return X, y



#X,y = leer_correos('full/index',200)
#print(X[0])
#parser = EmailParser()
#parser.parse(X[1])



#Funcion para Crear el dataset

def crear_dataset(indice, num):
    email_parser = EmailParser()
    X, y = leer_correos(indice, num)
    X_proc = []
    for i, email in zip(range(len(X)), X):
        print("\rParsing email: {0}".format(i+1), end='')
        X_proc.append(email_parser.parse(email))
    return X_proc, y

##########################################################################################

###ETAPA DE ENTRENAMIENTO

# Leemos únicamente un subconjunto de 100 correos electrónicos
X, y = crear_dataset('full/index', 100)
vectorizer = CountVectorizer()
vectorizer.fit(X)
X_vect = vectorizer.transform(X)
print(X_vect.toarray())

clf = LogisticRegression()

clf.fit(X_vect, y)






###################################################################################################
#Predicción
# Leemos 150 correos de nuestro conjunto de datos y nos quedamos únicamente con los 50 últimos 
# Estos 50 correos electrónicos no se han utilizado para entrenar el algoritmo
X, y = crear_dataset('full/index', 150)

X_test = X[100:]
y_test = y[100:]


# Aplicamos CountVectorizer
X_test = vectorizer.transform(X_test)

# Prediccion para un correo
y_pred = clf.predict(X_test)


print("\nPrediccion:\n", y_pred)
print("\nEtiquetas reales:\n", y_test)

print("Accuracy: {:.3f}".format(accuracy_score(y_test, y_pred)))

#Graficamos con la matriz confusion

# Naive Bayes
y_pred_nb = clf.predict(X_test)
y_true_nb = y_test
cm = confusion_matrix(y_true_nb, y_pred_nb)
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred_nb")
plt.ylabel("y_true_nb")
plt.show()



####################################################################################

#Entrenamiento con un conjunto de Datos Mayor
# Leemos 5000 correos electrónicos
X, y = crear_dataset('full/index', 5000)

# Utilizamos 4000 correos electrónicos para entrenar el algoritmo y 1000 para realizar pruebas
X_train, y_train = X[:4000], y[:4000]
X_test, y_test = X[4000:], y[4000:]

vectorizer = CountVectorizer()
vectorizer.fit(X_train)

X_train = vectorizer.transform(X_train)
X_train.toarray()
clf = LogisticRegression()
clf.fit(X_train, y_train)

X_test = vectorizer.transform(X_test)
y_pred = clf.predict(X_test)
print(y_pred)
print('Accuracy: {:.3f}'.format(accuracy_score(y_test, y_pred)))

#Graficamos con la matriz confusion

# Naive Bayes
y_pred_nb = clf.predict(X_test)
y_true_nb = y_test
cm = confusion_matrix(y_true_nb, y_pred_nb)
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred_nb")
plt.ylabel("y_true_nb")
plt.show()










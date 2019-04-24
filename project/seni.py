from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap

from PyQt5 import QtCore,QtGui,QtWidgets,uic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
import matplotlib.pyplot as plt


class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyWindow,self).__init__()
        uic.loadUi('review.ui',self)
        self.analyse.clicked.connect(self.open)


    def open(self):
        def analyse_text(classifier, vectorizer, text):
            result=text,classifier.predict(vectorizer.transform([text]))
            text, analysis_result = result
            if analysis_result[0] == '1':
                self.textBrowser.setText("Positive")
            else:
                self.textBrowser.setText("Negative")

        analyse_text(classifier, vectorizer ,self.textEdit.toPlainText())




def get_all_data():
    root = "pr\data/"

    with open(r"\Users\zeeshan\Videos\prokject\data\product.txt", "r") as text_file:
        data = text_file.read().split('\n')
             
    with open(r"C:\Users\zeeshan\Videos\prokject\data\amazon_cells_labelled.txt", "r") as text_file:
        data += text_file.read().split('\n')

    with open(r"C:\Users\zeeshan\Videos\prokject\data\amazon_laptop.txt", "r") as text_file:
        data += text_file.read().split('\n')

    return data
    get_all_data()

def preprocessing_data(data):
    processing_data = []
    for single_data in data:
        if len(single_data.split("\t")) == 2 and single_data.split("\t")[1] != "":
            processing_data.append(single_data.split("\t"))

    return processing_data
    all_data=get_all_data()
    preprocessing_data(all_data)


def split_data(data):
    total = len(data)
    training_ratio = 0.75
    training_data = []
    evaluation_data = []
    for indice in range(0, total):
        if indice < total * training_ratio:
            training_data.append(data[indice])
        else:
            evaluation_data.append(data[indice])

    return training_data, evaluation_data

def preprocessing_step():
    data = get_all_data()
    processing_data = preprocessing_data(data)

    return split_data(processing_data)

def training_step(data, vectorizer):
    training_text = [data[0] for data in data]
    training_result = [data[1] for data in data]

    training_text = vectorizer.fit_transform(training_text)

    return BernoulliNB().fit(training_text, training_result)

training_data, evaluation_data = preprocessing_step()
vectorizer = CountVectorizer(binary = 'true')
classifier = training_step(training_data, vectorizer)
result = classifier.predict(vectorizer.transform(["love this movie!"]))

result[0]
        

        

if __name__=='__main__':
    import sys
    app=QtWidgets.QApplication(sys.argv)
    window=MyWindow()
    window.show()
    #print(name)
    sys.exit(app.exec())

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

def assign_labels(data):
    def assign_new_labels(value):
        if value <= 55.4:
            return 'bezbedno'
        elif 55.5 <= value <= 150.4:
            return 'nebezbedno'
        elif value >= 150.5:
            return 'opasno'
        else:
            return -1

    data['Labels'] = data['PM_US Post'].apply(assign_new_labels)

def evaluation_classifier(conf_mat):
    TP = conf_mat[1, 1]
    TN = conf_mat[0, 0]
    FP = conf_mat[0, 1]
    FN = conf_mat[1, 0]
    precision = TP / (FP + TP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    F_score = 2 * precision * sensitivity / (precision * sensitivity)
    
    print('precision: ', precision)
    print('accuracy: ', accuracy)
    print('sensitivity/recall: ', sensitivity)
    print('specificity: ', specificity)
    print('F score: ', F_score)

def find_best_knn_params(x_train, y_train):
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    parameters = {'metric': ['minkowski', 'chebyshev', 'euclidean', 'manhattan', 'hamming'],
                  'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]}

    knn = KNeighborsClassifier()
    clf = GridSearchCV(estimator=knn, param_grid=parameters, scoring='accuracy', cv=kfold, refit=True, verbose=3)
    clf.fit(x_train, y_train)

    print(clf.best_score_)
    print(clf.best_params_)

def knn_classification(x_train, y_train, x_test, y_test, best_params=None):
    if best_params is not None:
        knn = KNeighborsClassifier(**best_params)
    else:
        knn = KNeighborsClassifier(n_neighbors=14, metric='manhattan')

    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test) 
    conf_mat = confusion_matrix(y_test, y_pred)
    print(conf_mat)

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=np.unique(y_test))
    disp.plot(cmap="Blues")
    plt.show()

    print(evaluation_classifier(conf_mat))
    print(classification_report(y_test, y_pred))
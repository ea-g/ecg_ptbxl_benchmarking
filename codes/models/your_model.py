from models.base_model import ClassificationModel
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import pickle
import numpy as np


def matrix_2_df(matrix, column_prefix='lead_'):
    """
    Converts 3-D numpy matrix to dataframe for input to sktime models.
    """
    from collections import defaultdict
    from pandas import Series, DataFrame
    output = defaultdict(list)
    for i in matrix[:, :, :]:
        for c in range(matrix.shape[-1]):
            output[column_prefix + str(c + 1)].append(Series(i[:, c:c + 1].flatten()))
    return DataFrame(data=output)


class MiniRocketModel(ClassificationModel):
    def __init__(self, name, n_classes,  sampling_frequency, outputfolder, input_shape, regularizer_C=.001, classifier='LR', toler=1e-3):
        self.name = name
        self.n_classes = n_classes
        self.sampling_frequency = sampling_frequency
        self.outputfolder = outputfolder
        self.input_shape = input_shape
        self.regularizer_C = regularizer_C
        self.classifier = classifier
        self.toler = toler

    def fit(self, X_train, y_train, X_val, y_val):
        mini_mv = MiniRocketMultivariate().fit(matrix_2_df(X_train))
        pickle.dump(mini_mv, open(self.outputfolder + 'miniR_clf.pkl', 'wb'))
        XF_train = mini_mv.transform(matrix_2_df(X_train))
        XF_val = mini_mv.transform(matrix_2_df(X_val))

        if self.classifier == 'LR':
            if self.n_classes > 1:
                clf = OneVsRestClassifier(
                    LogisticRegression(C=self.regularizer_C, solver='saga', max_iter=3000, n_jobs=-2, tol=self.toler))
            else:
                clf = LogisticRegression(C=self.regularizer_C, solver='saga', max_iter=3000, n_jobs=-2, tol=self.toler)
            clf.fit(XF_train, y_train)
            pickle.dump(clf, open(self.outputfolder + 'clf.pkl', 'wb'))

    def predict(self, X):
        mini_mv = pickle.load(open(self.outputfolder + 'miniR_clf.pkl', 'rb'))
        XF = mini_mv.transform(matrix_2_df(X))
        if self.classifier == 'LR':
            clf = pickle.load(open(self.outputfolder+'clf.pkl', 'rb'))
            if self.n_classes > 1:
                return clf.predict_proba(XF)
            else:
                return clf.predict_proba(XF)[:,1][:,np.newaxis]
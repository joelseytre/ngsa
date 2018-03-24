from predictors.NGSApredictor import NGSApredictor
from sklearn import svm
from sklearn.metrics import f1_score


class BaselineSVM(NGSApredictor):
    def run(self):
        model = svm.LinearSVC()

        model.fit(self.features['train_data'], self.train_labels)
        self.fscore_t = f1_score(self.train_labels, model.predict(self.features['train_data']))
        self.fscore_v = f1_score(self.valid_labels, model.predict(self.features['valid_data']))
        print("SVM model: F1 score - Training %.3f - Validation %.3f" % (self.fscore_t, self.fscore_v))

        if not self.settings['development_mode']:
            predictions = list(model.predict(self.features['testing']))
            self.prediction = zip(range(len(self.testing_set)), predictions)

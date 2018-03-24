from predictors.NGSApredictor import NGSApredictor
from sklearn import svm
from sklearn.metrics import f1_score
import lightgbm as lgb


class Gradboost(NGSApredictor):
    def run(self):
        model = lgb.LGBMClassifier(objective='binary', reg_lambda=10, n_estimators=10000)
        # model = lgb.LGBMClassifier(objective='binary', n_estimators=10000)

        # model.fit(self.features['train_data'], self.train_labels)
        model.fit(self.features['train_data'], self.train_labels,
                   eval_set=[(self.features['valid_data'], self.valid_labels)], early_stopping_rounds=50)

        self.fscore_t = f1_score(self.train_labels, model.predict(self.features['train_data']))
        self.fscore_v = f1_score(self.valid_labels, model.predict(self.features['valid_data']))
        print("Gradboost model: F1 score - Training %.3f - Validation %.3f" % (self.fscore_t, self.fscore_v))

        if not self.settings['development_mode']:
            predictions = list(model.predict(self.features['testing']))
            self.prediction = zip(range(len(self.testing_set)), predictions)

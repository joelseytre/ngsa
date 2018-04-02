from predictors.NGSApredictor import NGSApredictor
from sklearn import svm
from sklearn.metrics import f1_score
import lightgbm as lgb
import numpy as np

class Gradboost(NGSApredictor):
    def run(self):
        # model = lgb.LGBMClassifier(objective='binary', reg_lambda=lmbda, n_estimators=10000)
        model = lgb.LGBMClassifier(objective='binary', reg_lambda=10, n_estimators=10000)
        # model = lgb.LGBMClassifier(objective='binary', n_estimators=10000)

        # model.fit(self.features['train_data'], self.train_labels)
        model.fit(self.features['train_data'], self.train_labels,
                   eval_set=[(self.features['valid_data'], self.valid_labels)],
                    early_stopping_rounds=50, verbose=False)

        self.fscore_t = f1_score(self.train_labels, model.predict(self.features['train_data']))
        self.fscore_v = f1_score(self.valid_labels, model.predict(self.features['valid_data']))
        print("Gradboost model: F1 score - Training %.3f - Validation %.3f" % (self.fscore_t, self.fscore_v))

        if not self.settings['development_mode']:
            predictions = list(model.predict(self.features['testing']))
            self.prediction = zip(range(len(self.testing_set)), predictions)

    def run_(self, feat):
        if feat is None:
            self.run()
        else:
            features_list = ['overlap_title', 'temp_diff', 'comm_auth',
                             'tfidf_distance_corpus', 'tfidf_distance_titles', 'tfidf_distance_authors',
                             'num_inc_edges', 'shortest_path_dijkstra', 'shortest_path_dijkstra_und', 'jaccard_und']
            feat_to_idx = dict(zip(features_list, range(len(features_list))))

            feat_train = np.delete(self.features['train_data'], [feat_to_idx[f] for f in feat], 1)
            feat_valid = np.delete(self.features['valid_data'], [feat_to_idx[f] for f in feat], 1)

            model = lgb.LGBMClassifier(objective='binary', reg_lambda=10, n_estimators=10000)

            model.fit(feat_train, self.train_labels,
                   eval_set=[(feat_valid, self.valid_labels)],
                    early_stopping_rounds=50, verbose=False)
            fscore_t = f1_score(self.train_labels, model.predict(feat_train))
            fscore_v = f1_score(self.valid_labels, model.predict(feat_valid))
            print("Gradboost Model with removed features: ", feat)
            print("Gradboost model: F1 score - Training %.3f - Validation %.3f" % (fscore_t, fscore_v))




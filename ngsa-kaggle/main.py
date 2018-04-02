from predictor_settings import predictor_settings
from predictors.NGSApredictor import NGSApredictor
from predictors.svm_baseline import BaselineSVM
from predictors.gradboost import Gradboost

## choose model here
# predictor = NGSApredictor(predictor_settings)
# predictor = BaselineSVM(predictor_settings)
predictor = Gradboost(predictor_settings)

if not predictor.settings['load_features']:
    predictor.process_data()
else:
    predictor.load_features("stored_training_v3.txt", "stored_testing_v3.txt")

if predictor.settings['store_features']:
    predictor.store_features("stored_training_v3.txt", "stored_testing_v3.txt")

predictor.describe_data()
predictor.run()

if not predictor.settings['development_mode']:
    predictor.save_predictions()

import datetime

predictor_settings = dict()
predictor_settings['print'] = True
# the name of the file is adjusted so that it contains date and time
predictor_settings['output_name'] = "gradboost_tfidf_dijkstra_count"
predictor_settings['output_name'] = predictor_settings['output_name'] \
              + "_" + str(datetime.date.today().day) \
              + "-" + str(datetime.date.today().month) \
              + "-" + str(datetime.date.today().year) \
              + "-" + str(datetime.datetime.now().hour) \
              + "h" + str("%02.f" % datetime.datetime.now().minute)

# decides what ratio of the 615k training nodes we are using to compute the features. Default: 0.1
predictor_settings['training_ratio'] = 0.1
# decides what part of the training data we use. Default: 0.9
predictor_settings['train_valid_ratio'] = 0.9
# if True, don't compute test dataset
predictor_settings['development_mode'] = False
# should we store the features computed
predictor_settings['store_features'] = False
# should we use the features that were stored
predictor_settings['load_features'] = True

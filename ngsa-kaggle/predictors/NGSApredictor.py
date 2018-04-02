import csv
import numpy as np
# import nltk
import os
import random
import tqdm
import igraph
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn import preprocessing


class NGSApredictor:
    def __init__(self, predictor_settings):
        self.settings = predictor_settings.copy()
        # do some nltk stuff (for stopwords, tokenization,...)
        # nltk.download('punkt')  # for tokenization
        # nltk.download('stopwords')
        # self.stpwds = set(nltk.corpus.stopwords.words("english"))
        # self.stemmer = nltk.stem.PorterStemmer()

        # read training and test set
        print(os.path.dirname(__file__))
        with open(os.path.join(os.path.dirname(__file__), '..', 'data', 'testing_set.txt'), "r") as f:
            reader = csv.reader(f)
            testing_set = list(reader)
        with open(os.path.join(os.path.dirname(__file__), '..', 'data', 'training_set.txt'), "r") as f:
            reader = csv.reader(f)
            training_set = list(reader)
        self.training_set = [element[0].split(" ") for element in training_set]
        self.testing_set = [element[0].split(" ") for element in testing_set]
        to_keep = random.sample(range(len(self.training_set)), k=int(round(len(training_set)*self.settings['training_ratio'])))
        self.training_set_reduced = [self.training_set[i] for i in to_keep]

        with open(os.path.join(os.path.dirname(__file__), '..', 'data', 'node_information.csv'), "r") as f:
            reader = csv.reader(f)
            self.node_info = list(reader)
        # EXTRACT INFORMATION FROM "node_information.csv"
        # the columns of the data frame below are:
        # (1) paper unique ID (integer)
        # (2) publication year (integer)
        # (3) paper title (string)
        # (4) authors (strings separated by ,)
        # (5) name of journal (optional) (string)
        # (6) abstract (string) - lowercased, free of punctuation except intra-word dashes
        self.IDs = [element[0] for element in self.node_info]
        self.publication_years = [element[1] for element in self.node_info]
        self.titles = [element[2] for element in self.node_info]
        self.authors = [element[3] for element in self.node_info]
        self.journals = [element[4] for element in self.node_info]
        self.corpus = [element[5] for element in self.node_info]
        
        self.vectorizer_corpus = TfidfVectorizer(stop_words="english")
        self.features_TFIDF_corpus = self.vectorizer_corpus.fit_transform(self.corpus)
        
        self.vectorizer_titles = TfidfVectorizer()
        self.features_TFIDF_titles = self.vectorizer_titles.fit_transform(self.titles)
        
        self.vectorizer_authors = TfidfVectorizer()
        self.features_TFIDF_authors = self.vectorizer_authors.fit_transform(self.authors)
        
        if self.settings['print']:
            print("\n\nPredicting stuff... \nTraining: %i \nReduced training: %i \nTesting: %i\n"
                  % (len(self.training_set), int(round(len(self.training_set_reduced))), len(self.testing_set)))

        self.features = dict()
        self.training_labels = None
        self.train_labels = None
        self.valid_labels = None
        self.prediction = None
        self.fscore_v = None
        self.fscore_t = None
        self.G = igraph.Graph(directed=True)
        self.G_und = None

    def process_data(self):
        # build graph
        edges = [(element[0], element[1]) for element in self.training_set if element[2] == "1"]
        nodes = self.IDs
        self.G.add_vertices(nodes)
        self.G.add_edges(edges)
        self.G_und = self.G.as_undirected()

        datasets = [self.training_set_reduced, self.testing_set]
        datasets_name = ["training", "testing"]
        num_datasets = 2 - self.settings['development_mode']
        for k in range(num_datasets):
            dataset = datasets[k]
            overlap_title = []
            temp_diff = []
            comm_auth = []
            tfidf_distance_corpus = []
            tfidf_distance_titles = []
            tfidf_distance_authors = []
            shortest_path_dijkstra = []
            shortest_path_dijkstra_und = []
            jaccard_und = []
            num_inc_edges = []
            counter = 0
            for i in tqdm.tqdm(range(len(dataset))):
                source = dataset[i][0]
                target = dataset[i][1]

                source_info = [element for element in self.node_info if element[0] == source][0]
                target_info = [element for element in self.node_info if element[0] == target][0]

                index_source = self.IDs.index(source)
                index_target = self.IDs.index(target)

                # convert to lowercase and tokenize
                source_title = source_info[2].lower().split(" ")
                # remove stopwords
                source_title = [token for token in source_title if token not in self.stpwds]
                source_title = [self.stemmer.stem(token) for token in source_title]

                target_title = target_info[2].lower().split(" ")
                target_title = [token for token in target_title if token not in self.stpwds]
                target_title = [self.stemmer.stem(token) for token in target_title]

                source_auth = source_info[3].split(",")
                target_auth = target_info[3].split(",")

                overlap_title.append(len(set(source_title).intersection(set(target_title))))
                temp_diff.append(int(source_info[1]) - int(target_info[1]))
                comm_auth.append(len(set(source_auth).intersection(set(target_auth))))
                tfidf_distance_corpus.append(
                    linear_kernel(self.features_TFIDF_corpus[index_source:index_source+1],
                                  self.features_TFIDF_corpus[index_target:index_target+1]).flatten())
                tfidf_distance_titles.append(
                    linear_kernel(self.features_TFIDF_titles[index_source:index_source+1],
                                  self.features_TFIDF_titles[index_target:index_target+1]).flatten())
                tfidf_distance_authors.append(
                    linear_kernel(self.features_TFIDF_authors[index_source:index_source+1],
                                  self.features_TFIDF_authors[index_target:index_target+1]).flatten())

                num_inc_edges.append(len([element[1] for element in self.training_set if element[1] == target])
                                     + len([element[1] for element in self.testing_set if element[1] == target]))
                # print("%i - %i" % (len([element[1] for element in self.training_set if element[1] == target]), len([element[1] for element in self.testing_set if element[1] == target])))
                # print("avg edge: %.1f" % np.mean(num_inc_edges))

                # delete the edge, compute shortest path, then  put it back (ONLY FOR TRAINING DATA, WHEN EDGE EXISTED)
                if k == 0 and dataset[i][2] == "1":
                    self.G.delete_edges((index_source, index_target))
                    self.G_und.delete_edges((index_source, index_target))
                shortest_path_dijkstra.append(
                    min(100000, self.G.shortest_paths_dijkstra(source=source, target=target)[0][0])
                )
                short_path_und = min(100000, self.G_und.shortest_paths_dijkstra(source=source, target=target)[0][0])
                shortest_path_dijkstra_und.append(
                    short_path_und
                )
                if short_path_und > 2:
                    jacc = 0
                else:
                    jacc = self.G_und.similarity_jaccard(pairs=[(index_source, index_target)])[0]
                jaccard_und.append(jacc)

                # if min(100000, self.G.shortest_paths_dijkstra(source=source, target=target)[0][0]) != min(100000, self.G_und.shortest_paths_dijkstra(source=source, target=target)[0][0]):
                #     print("%i is different than %i" % (int(min(100000, self.G.shortest_paths_dijkstra(source=source, target=target)[0][0])), int(min(100000, self.G_und.shortest_paths_dijkstra(source=source, target=target)[0][0]))))
                if k == 0 and dataset[i][2] == "1":
                    self.G.add_edge(index_source, index_target)
                    self.G_und.add_edge(index_source, index_target)

                counter += 1
                if counter % 1000 == 0 and self.settings['print']:
                    print("\n %i %s examples processed" % (counter, datasets_name[k]))
            print("Done processing %s: %i elements\n\n" % (datasets_name[k], counter))

            feat = np.array([overlap_title, temp_diff, comm_auth,
                             tfidf_distance_corpus, tfidf_distance_titles, tfidf_distance_authors,
                             num_inc_edges, shortest_path_dijkstra, shortest_path_dijkstra_und, jaccard_und]).T
            self.features[datasets_name[k]] = feat
            if k == 0:
                self.training_labels = np.array([int(element[2]) for element in dataset])
                m = self.features['training'].mean(axis=0)
                std = self.features['training'].std(axis=0)

            self.features[datasets_name[k]] = (self.features[datasets_name[k]] - m) / std

        idx_mid = int(len(self.features['training'])*self.settings['train_valid_ratio'])
        self.features['train_data'] = self.features['training'][:idx_mid]
        self.features['valid_data'] = self.features['training'][idx_mid:]
        self.train_labels = self.training_labels[:idx_mid]
        self.valid_labels = self.training_labels[idx_mid:]

    def save_predictions(self):
        self.settings['output_name'] += ("-FS%.3f.csv" % self.fscore_v)
        path_out = os.path.join(os.getcwd(), "out")
        os.chdir(path_out)
        if self.prediction is None:
            print ("No Data => random predictions")
            self.settings['output_name'] = "rand_predictions.csv"
            random_predictions = np.random.choice([0, 1], size=len(self.testing_set))
            self.prediction = zip(range(len(self.testing_set)), random_predictions)
        with open(self.settings['output_name'], "wb") as predfile:
            csv_out = csv.writer(predfile)
            csv_out.writerow(['id', 'category'])
            for row in self.prediction:
                csv_out.writerow(row)

        print("Stored results under the name '%s' at location '%s'" % (self.settings['output_name'], os.getcwd()))

    def run(self):
        pass

    def store_features(self, name_training, name_testing):
        np.savetxt(name_training, self.features['training'])
        np.savetxt(name_training[:-4]+"_labels.txt", self.training_labels)
        try:
            np.savetxt(name_testing, self.features['testing'])
        except:
            np.savetxt(name_testing, np.array([]))
        print("Stored features!")

    def load_features(self, name_training, name_testing):
        self.features['training'] = np.loadtxt(name_training)
        self.features['testing'] = np.loadtxt(name_testing)
        self.training_labels = np.loadtxt(name_training[:-4]+"_labels.txt", dtype=int)

        idx_mid = int(len(self.features['training'])*self.settings['train_valid_ratio'])
        self.features['train_data'] = self.features['training'][:idx_mid]
        self.features['valid_data'] = self.features['training'][idx_mid:]
        self.train_labels = self.training_labels[:idx_mid]
        self.valid_labels = self.training_labels[idx_mid:]
        print("Read features from stored data!")

    def describe_data(self):
        print("\nmeans")
        print(self.features['train_data'].mean(axis=0))
        print(self.features['valid_data'].mean(axis=0))
        print(self.features['training'].mean(axis=0))
        if not self.settings['development_mode']:
            print(self.features['testing'].mean(axis=0))

        print("\nstds")
        print(self.features['train_data'].std(axis=0))
        print(self.features['valid_data'].std(axis=0))
        print(self.features['training'].std(axis=0))
        if not self.settings['development_mode']:
            print(self.features['testing'].std(axis=0))

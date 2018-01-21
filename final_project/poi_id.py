#!/usr/bin/python

import sys
import pickle
import pprint
import matplotlib.pyplot
sys.path.append("../tools/")
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from feature_format import featureFormat, targetFeatureSplit
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier
from tester import test_classifier, dump_classifier_and_data


with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Remove outlier
data_dict.pop('TOTAL')

my_dataset = data_dict

# Define features
features_list = ["poi", "deferred_income", "exercised_stock_options", "long_term_incentive", "salary",
                 "total_stock_value"]
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

# Naive Bayes:
# Define classifier, selector, pipeline, parameters to tune, and cross validation method
clf = GaussianNB()

select = SelectKBest()

steps = [("feature_selection", select),
         ("classifier", clf)]

nb_pipeline = Pipeline(steps)

nb_params2 = dict(feature_selection__k = ["all", 1, 2, 3, 4, 5])

sss = StratifiedShuffleSplit(labels, 10, test_size = 0.3, random_state = 12)

# Function to optimize and evaluate classifier
def getClassifierAndData(pipeline, params, cross_val, features, labels, features_list):

    # Find parameters to optimize F1 score using Grid Search
    cv = GridSearchCV(pipeline,
                      param_grid = params,
                      scoring = "f1",
                      cv = sss)

    cv.fit(features, labels)

    clf = cv.best_estimator_

    X_new = clf.named_steps["feature_selection"]
    
    clf_type = clf.named_steps["classifier"]
    
    # Print feature importances if classifier is a decision tree
    if isinstance(clf_type, DecisionTreeClassifier):
        print "-- Feature Importances --\n", clf_type.feature_importances_, "\n"

    # Get SKB scores rounded to 1 decimal place
    feature_scores = ['%.1f' % elem for elem in X_new.scores_]

    # Collect and print SKB features used and scores
    feature_scores_dict = {features_list[i+1] : feature_scores[i] for i in X_new.get_support(indices = True)}
    print "-- SKB Features & Scores --"
    pprint.pprint(feature_scores_dict)
    print ""

    # Evaluate classifier with provided function to find precision and recall
    test_classifier(clf, my_dataset, features_list)
    return clf


# Get data for NB classifier with new feature list
nb_clf2b = getClassifierAndData(nb_pipeline, nb_params2, sss, features, labels, features_list)

# Since NB gives better F1 score than DT, assign clf to NB clf
clf = nb_clf2b 

# Dump classifier, dataset and features list into pickle files
dump_classifier_and_data(clf, my_dataset, features_list)
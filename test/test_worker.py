import pytest
import pandas as pd
from worker import get_data, get_classifier, CatBoostClassifier, RandomForestClassifier

def test_get_data():
    '''Test if get_data returns a pandas DataFrame.'''
    data = get_data()
    assert isinstance(data, pd.DataFrame), "get_data should return a pandas DataFrame"

def test_get_classifier_catboost():
    '''Test if get_classifier correctly returns a CatBoostClassifier.'''
    classifier = get_classifier("CatBoostClassifier")
    assert isinstance(classifier, CatBoostClassifier), "get_classifier should return an instance of CatBoostClassifier"

def test_get_classifier_randomforest():
    '''Test if get_classifier correctly returns a RandomForestClassifier.'''
    classifier = get_classifier("RandomForestClassifier")
    assert isinstance(classifier, RandomForestClassifier), "get_classifier should return an instance of RandomForestClassifier"

def test_get_classifier_invalid():
    '''Test if get_classifier returns None for an invalid classifier name.'''
    classifier = get_classifier("InvalidClassifier")
    assert classifier is None, "get_classifier should return None for an invalid classifier name"

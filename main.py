import pickle
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from yellowbrick.classifier import ConfusionMatrix


def split_data_with_processing(dataframe):
    le_play = LabelEncoder()

    dataframe["play"] = le_play.fit_transform(dataframe["play"])

    # Label encode 'temperature' - temperatura
    temp_order = {'cool': 0, 'mild': 1, 'hot': 2}
    dataframe['temperature'] = dataframe['temperature'].map(temp_order)

    # Label encode 'humidity' - humidade
    humidity_order = {'normal': 0, 'high': 1}
    dataframe['humidity'] = dataframe['humidity'].map(humidity_order)

    dataframe = pd.get_dummies(dataframe, columns=["outlook"])

    # Split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        dataframe.drop(columns=["play"]),
        dataframe["play"],
        test_size=0.2,
        random_state=0
    )

    print(dataframe.columns)
    print(dataframe.values)

    # Salva o conjunto de treino e teste
    with open("weather_train_test.pkl", "wb") as f:
        pickle.dump((x_train.columns, x_train.values, x_test.values, y_train.values, y_test.values), f)


def naive_bayes():
    # Carregando os conjuntos de dados treino e teste do arquivo usando pickle
    with open("weather_train_test.pkl", 'rb') as f:
        feature_names, x_train, x_test, y_train, y_test = pickle.load(f)

    gnb = GaussianNB()
    gnb.fit(x_train, y_train)

    print(feature_names)
    # windy: Sim, outlook_overcast: Nao, outlook_rainy: Sim, outlook_sunny: False, humidity_high: Nao,
    # humidity_normal: Sim, temperature_cool: Sim, temperature_hot: Nao, temperature_mild: Nao
    y_pred = gnb.predict([[0, 0, True, False, True, False]])
    print(f"Classe prevista {y_pred}")

    cm = ConfusionMatrix(gnb, classes=["Nao", "Sim"])
    cm.fit(x_train, y_train)
    cm.score(x_test, y_test)
    cm.show(outpath=f"{gnb.__class__.__name__}_confusion_matrix.png")


def decision_tree():
    # Load training and test sets from pickle file
    with open("weather_train_test.pkl", 'rb') as f:
        feature_names, x_train, x_test, y_train, y_test = pickle.load(f)

    dt_param_grid = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    start_time = time.time()
    dt_grid = GridSearchCV(DecisionTreeClassifier(), dt_param_grid, cv=5)
    dt_grid.fit(x_train, y_train)
    end_time = time.time()
    print("Best DecisionTree parameters (GridSearch):", dt_grid.best_params_)
    print("Best DecisionTree score (GridSearch):", dt_grid.best_score_)
    print(f"Tempo de execução (GridSearch): {end_time - start_time:.2f} segundos")

    cm = ConfusionMatrix(dt_grid, classes=["Nao", "Sim"])
    cm.fit(x_train, y_train)
    cm.score(x_test, y_test)
    cm.show(outpath=f"{dt_grid.__class__.__name__}_confusion_matrix_grid.png")


    # DecisionTree com RandomizedSearchCV
    start_time = time.time()
    dt_random = RandomizedSearchCV(DecisionTreeClassifier(), dt_param_grid, n_iter=100, cv=5, random_state=0)
    dt_random.fit(x_train, y_train)
    end_time = time.time()
    print("\nBest DecisionTree parameters (RandomSearch):", dt_random.best_params_)
    print("Best DecisionTree score (RandomSearch):", dt_random.best_score_)
    print(f"Tempo de execução (RandomSearch): {end_time - start_time:.2f} segundos")

    cm = ConfusionMatrix(dt_random, classes=["Nao", "Sim"])
    cm.fit(x_train, y_train)
    cm.score(x_test, y_test)
    cm.show(outpath=f"{dt_grid.__class__.__name__}_confusion_matrix_random.png")


def random_forest():
    print("\n")

    # Load training and test sets from pickle file
    with open("weather_train_test.pkl", 'rb') as f:
        feature_names, x_train, x_test, y_train, y_test = pickle.load(f)

    rf_param_dist = {
        'n_estimators': [1, 2, 4, 5, 6, 7, 8, 9, 10],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None] + list(range(1, 10)),
        'min_samples_split': range(2, 11),
        'min_samples_leaf': range(1, 5),
        'bootstrap': [True, False]
    }

    # RandomForest com GridSearch
    start_time = time.time()
    rf_grid = GridSearchCV(RandomForestClassifier(), rf_param_dist, cv=5)
    rf_grid.fit(x_train, y_train)
    end_time = time.time()
    print("Best RandomForest parameters (GridSearch):", rf_grid.best_params_)
    print("Best RandomForest score (GridSearch):", rf_grid.best_score_)
    print(f"Tempo de execução (GridSearch): {end_time - start_time:.2f} segundos")

    cm = ConfusionMatrix(rf_grid, classes=["Nao", "Sim"])
    cm.fit(x_train, y_train)
    cm.score(x_test, y_test)
    cm.show(outpath=f"{rf_grid.__class__.__name__}_confusion_matrix_grid.png")

    # RandomForest com RandomizedSearchCV
    start_time = time.time()
    rf_random = RandomizedSearchCV(RandomForestClassifier(), rf_param_dist, n_iter=100, cv=5, random_state=0)
    rf_random.fit(x_train, y_train)
    end_time = time.time()
    print("\nBest RandomForest parameters (RandomSearch):", rf_random.best_params_)
    print("Best RandomForest score (RandomSearch):", rf_random.best_score_)
    print(f"Tempo de execução (RandomSearch): {end_time - start_time:.2f} segundos")

    cm = ConfusionMatrix(rf_random, classes=["Nao", "Sim"])
    cm.fit(x_train, y_train)
    cm.score(x_test, y_test)
    cm.show(outpath=f"{rf_random.__class__.__name__}_confusion_matrix_random.png")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = pd.read_csv("weather.nominal.csv", delimiter=",", )
    split_data_with_processing(df)
    naive_bayes()
    decision_tree()
    random_forest()

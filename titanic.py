import pandas as pd
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


def process_data(df):
    # Preenchendo dados faltantes
    # Removendo instâncias com valores nulos em 'Sex, 'Age', 'Embarked', 'Fare'
    df = df.dropna(subset=['Sex'])
    df = df.dropna(subset=['Age'])
    df = df.dropna(subset=['Embarked'])
    df = df.dropna(subset=['Fare'])

    # Transformação de características categóricas em numéricas
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])

    # Removendo colunas que não serão usadas no treinamento
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    return df


def train_bagging(x_train, y_train):
    bagging = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42)
    bagging.fit(x_train, y_train)
    return bagging


def train_boosting(x_train, y_train):
    boosting = XGBClassifier(n_estimators=100, random_state=42)
    boosting.fit(x_train, y_train)
    return boosting


def train_random_forest(x_train, y_train):
    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
    random_forest.fit(x_train, y_train)
    return random_forest


def essemble_models():
    # Carregando os dados
    df_train = pd.read_csv("titanic/train.csv")
    df_test = pd.read_csv("titanic/test.csv")

    # Processamento de dados
    df_train = process_data(df_train)
    df_test = process_data(df_test)

    # Separando características e rótulos
    x_train = df_train.drop('Survived', axis=1)
    y_train = df_train['Survived']

    # Treinando os modelos e fazendo previsões
    model_bagging = train_bagging(x_train, y_train)
    predictions_bagging = model_bagging.predict(df_test)

    model_boosting = train_boosting(x_train, y_train)
    predictions_boosting = model_boosting.predict(df_test)

    model_rf = train_random_forest(x_train, y_train)
    predictions_rf = model_rf.predict(df_test)

    # Imprimindo as previsões
    print("Previsões com Bagging:")
    print(predictions_bagging)
    print("\nPrevisões com Boosting:")
    print(predictions_boosting)
    print("\nPrevisões com Random Forest:")
    print(predictions_rf)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    essemble_models()


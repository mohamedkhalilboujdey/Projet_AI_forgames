import pandas as pd
from sklearn.tree import DecisionTreeClassifier

class BossDecisionTree:
    def __init__(self):
        data = pd.read_csv("boss_data.csv")

        X = data[["jump_rate", "duck_rate", "score"]]
        y = data["attack"]

        self.model = DecisionTreeClassifier(max_depth=3)
        self.model.fit(X, y)

    def predict(self, jump_rate, duck_rate, score):
        return self.model.predict([[jump_rate, duck_rate, score]])[0]

from __future__ import annotations

from collections import defaultdict

import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):

        model = self.base_model_class(**self.base_model_params)
        s = -1 * self.loss_derivative(y, predictions)
        indx = np.random.choice(x.shape[0], size=int(x.shape[0] * self.subsample))
        x_boot = x[indx]
        s_boot = s[indx]
        model.fit(x_boot,s_boot)
        new_prediction = model.predict(x)

        gamma = self.find_optimal_gamma(y, predictions, new_prediction)



        self.gammas.append(gamma)
        self.models.append(model)

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])
        best_score = 0
        n_bad_rounds = 0
        for _ in range(self.n_estimators):
            self.fit_new_base_model(x_train, y_train, train_predictions)
            current_model = self.models[-1]
            current_gamma = self.gammas[-1]
            train_predictions += self.learning_rate * current_gamma * current_model.predict(x_train)
            valid_predictions +=  self.learning_rate * current_gamma * current_model.predict(x_valid)
            train_loss = self.loss_fn(y_train, train_predictions)
            val_loss = self.loss_fn(y_valid, valid_predictions)

            self.history['train'].append(train_loss)
            self.history['val'].append(val_loss)

            if self.early_stopping_rounds is not None:
                current_score = roc_auc_score(y_valid, self.sigmoid(valid_predictions))

                if current_score > best_score:
                    best_score = current_score
                else:
                    n_bad_rounds += 1

                    if n_bad_rounds == self.early_stopping_rounds:
                        break


        if self.plot:
            sns.lineplot(data=self.history)





    def predict_proba(self, x):
        predictions = np.zeros(x.shape[0])
        for gamma, model in zip(self.gammas, self.models):
            predictions += self.learning_rate * gamma * model.predict(x)
        
        probabilities = np.zeros((x.shape[0], 2))
        probabilities[:, 1] = self.sigmoid(predictions)
        probabilities[:, 0] = 1 - probabilities[:, 1]
        return probabilities


    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        total_importances_accumulator = np.zeros(self.models[0].feature_importances_.shape)
        
        for individual_model in self.models:
            total_importances_accumulator += individual_model.feature_importances_
        
        average_importances = total_importances_accumulator / len(self.models)
        normalized_importances = average_importances / average_importances.sum()
        
        return normalized_importances

        



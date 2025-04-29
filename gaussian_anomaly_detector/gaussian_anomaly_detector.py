import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import pickle
import matplotlib.pyplot as plt


class GaussianAnomalyDetector(BaseEstimator, ClassifierMixin):
    def __init__(self, features=None, epsilons=None):
        self.features = features
        self.epsilons = epsilons if epsilons is not None else [1e-15, 1e-10, 1e-5, 1e-3, 1e-2, 1e-1]
        self.means_ = None
        self.stds_ = None
        self.best_log_epsilon_ = None
        self.performance_ = None

    def fit(self, X, y):
        # Auto-select features if not provided
        if self.features is None:
            self.features = [col for col in X.columns]

        df = X.copy()
        df['Class'] = y
        df_normal = df[df['Class'] == 0]

        self.means_ = df_normal[self.features].mean()
        self.stds_ = df_normal[self.features].std()
        return self

    def _log_prob(self, X):
        z_scores = (X[self.features] - self.means_) / self.stds_
        probs = (1.0 / (np.sqrt(2 * np.pi) * self.stds_)) * np.exp(-0.5 * (z_scores ** 2))
        return np.log(probs + 1e-12).sum(axis=1)

    def predict_proba(self, X):
        log_probs = self._log_prob(X)
        return log_probs.to_frame(name='LogProbability')

    def predict(self, X):
        if self.best_log_epsilon_ is None:
            raise RuntimeError("Call score(X, y) first to select the best threshold.")
        log_probs = self._log_prob(X)
        return (log_probs < self.best_log_epsilon_).astype(int)

    def score(self, X, y):
        log_probs = self._log_prob(X)
        performances = []

        for e in np.log(self.epsilons):
            preds = (log_probs < e).astype(int)
            f1 = f1_score(y, preds, zero_division=0)
            performances.append((e, f1))

        # Choose the epsilon with highest F1
        best_e, best_f1 = max(performances, key=lambda x: x[1])
        self.best_log_epsilon_ = best_e
        self.performance_ = performances
        return best_f1
    def __repr__(self):
        return (f"GaussianAnomalyDetector(features={self.features}, "
                f"epsilons={self.epsilons}, "
                f"trained={self.trained_status()})")

    def trained_status(self):
        return self.means_ is not None
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
        
    def plot_precision_recall(self, X, y):
        if self.performance_ is None:
            raise RuntimeError("Must call score(X, y) before plotting precision-recall.")
        
        log_probs = self._log_prob(X)
        precisions = []
        recalls = []

        for e in np.log(self.epsilons):
            preds = (log_probs < e).astype(int)
            precision = precision_score(y, preds, zero_division=0)
            recall = recall_score(y, preds, zero_division=0)
            precisions.append(precision)
            recalls.append(recall)

        plt.figure(figsize=(7,5))
        plt.plot(recalls, precisions, marker='o')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision vs Recall Curve')
        plt.grid()
        plt.show()





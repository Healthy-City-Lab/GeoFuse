import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
from skrebate import ReliefF
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mutual_info_score
import logging


class FusionOptimizer:
    def __init__(self, df, outcome_col, features):
        """
        df: Input DataFrame.
        outcome_col: Target variable name.
        features: List of column names to fuse.
        """
        self.df = df.dropna(subset=[outcome_col] + features).copy()
        self.target = outcome_col
        self.features = features
        self.best_params = {}

        # Normalize features 0-1 (Crucial for comparable weights)
        scaler = MinMaxScaler()
        self.df[self.features] = scaler.fit_transform(self.df[self.features])

    def feature_importance_relieff(self, n_neighbors=100):
        """
        Runs ReliefF to determine feature quality before optimization.
        Matches logic from CGI.ipynb.
        """
        X = self.df[self.features].values
        y = self.df[self.target].values

        fs = ReliefF(n_features_to_select=len(self.features), n_neighbors=n_neighbors)
        fs.fit(X, y)

        importance = dict(zip(self.features, fs.feature_importances_))
        return importance

    def run_optimization(
        self, total_trials=100, random_trials=20, maximize_method="pearson"
    ):
        """
        Runs the Optuna optimization.
        total_trials: Total iterations.
        random_trials: Initial random exploration (n_startup_trials).
        maximize_method: 'pearson', 'spearman', or 'mutual_info'.
        """
        print(
            f"Starting Optimization: {random_trials} Random + {total_trials - random_trials} TPE Trials"
        )

        # TPESampler handles the switch from Random to Optimized sampling
        sampler = TPESampler(n_startup_trials=random_trials)

        study = optuna.create_study(direction="maximize", sampler=sampler)

        # Pass the method to the objective function via lambda
        study.optimize(
            lambda t: self._objective(t, maximize_method), n_trials=total_trials
        )

        self.best_params = study.best_params
        print(f"Optimization Complete. Best Score: {study.best_value}")
        return self.best_params

    def _objective(self, trial, method):
        # 1. Suggest Weights (0.0 to 1.0)
        weights = {f: trial.suggest_float(f, 0.0, 1.0) for f in self.features}

        # 2. Calculate Composite Index (Vectorized)
        composite = np.zeros(len(self.df))
        for feature, weight in weights.items():
            composite += self.df[feature].values * weight

        # 3. Calculate Fit Score
        if method == "pearson":
            score = self.df[self.target].corr(pd.Series(composite), method="pearson")
        elif method == "spearman":
            score = self.df[self.target].corr(pd.Series(composite), method="spearman")
        elif method == "mutual_info":
            # Binning required for Mutual Info
            score = mutual_info_score(
                pd.qcut(self.df[self.target], 10, labels=False, duplicates="drop"),
                pd.qcut(composite, 10, labels=False, duplicates="drop"),
            )

        return score

    def apply_best_weights(self):
        """Returns the DataFrame with the new 'CGI' column added."""
        if not self.best_params:
            raise ValueError("Run optimization first.")

        self.df["CGI"] = np.zeros(len(self.df))
        for feature, weight in self.best_params.items():
            self.df["CGI"] += self.df[feature].values * weight

        return self.df

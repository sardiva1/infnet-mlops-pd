"""
transformers/stateful.py — Transformadores Stateful (aprendem parâmetros do treino).

⚠  AVISO MLOps — Data Leakage
   Estes transformadores aprendem estatísticas dos dados de treino no fit().
   NUNCA aplique fit() em todo o dataset antes do split treino/holdout.

   Fluxo correto:
       imputer.fit(X_treino).transform(X_treino)   # aprende no treino
       imputer.transform(X_holdout)                  # aplica no holdout

   Integração no pipeline de modelagem (modelagem.py):
       pipe = Pipeline([
           ('scaler',  StandardScalerTransformer(...)),
           ('modelo',  Ridge()),
       ])
       pipe.fit(X_treino, y_treino)

   NÃO use estes transformadores no preprocessamento.py — esse script roda
   antes do split e aplicaria fit() em dados de teste, causando data leakage.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.preprocessing.base import BaseFeatureTransformer


class StandardScalerTransformer(BaseFeatureTransformer):
    """
    Aplica Z-score normalization: z = (x − μ) / σ.

    Por que StandardScaler?
    - Regressão linear e SVM são sensíveis à escala das features.
    - Gradient boosting e Random Forest NÃO precisam de escalonamento.

    Parâmetros aprendidos no fit (APENAS no conjunto de treino):
        mean_  (dict): {coluna: média}
        std_   (dict): {coluna: desvio padrão}

    Colunas com std=0 são ignoradas (constantes — sem informação).

    Raises:
        RuntimeError: Se transform() for chamado antes de fit().
    """

    def __init__(self, columns: list[str], logger: Any = None) -> None:
        self.columns = columns
        self.logger = logger

    def fit(self, X: pd.DataFrame, y=None) -> "StandardScalerTransformer":
        """Aprende média e desvio padrão das colunas especificadas."""
        self.mean_: dict[str, float] = {}
        self.std_: dict[str, float] = {}
        ausentes: list[str] = []

        for col in self.columns:
            if col not in X.columns:
                ausentes.append(col)
                continue

            mu = float(X[col].mean())
            sigma = float(X[col].std())

            if sigma == 0:
                self._warn(
                    "StandardScalerTransformer.fit: '%s' tem std=0 (constante) — ignorada.", col
                )
                continue

            self.mean_[col] = mu
            self.std_[col] = sigma

        if ausentes:
            self._warn(
                "StandardScalerTransformer.fit: colunas ausentes ignoradas: %s", ausentes
            )

        self._log(
            "StandardScalerTransformer.fit: parâmetros aprendidos para %d colunas.",
            len(self.mean_),
        )
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Aplica Z-score nas colunas ajustadas no fit."""
        if not hasattr(self, "mean_"):
            raise RuntimeError(
                "StandardScalerTransformer não foi ajustado. Chame fit() antes de transform()."
            )

        X = X.copy()
        escalonadas: list[str] = []

        for col, mu in self.mean_.items():
            if col not in X.columns:
                continue
            X[col] = (X[col] - mu) / self.std_[col]
            escalonadas.append(col)

        self._log(
            "StandardScalerTransformer.transform: %d colunas escalonadas (z-score).",
            len(escalonadas),
        )
        return X

    @property
    def scale_params(self) -> pd.DataFrame:
        """Retorna DataFrame com média e desvio padrão aprendidos (útil para auditoria)."""
        return pd.DataFrame({"mean": self.mean_, "std": self.std_}).rename_axis("feature")

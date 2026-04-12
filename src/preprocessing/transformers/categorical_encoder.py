"""
transformers/ratio_features.py — Transformador de Features de Razão.

Cria features normalizadas pelo tamanho do bloco censitário (nº de domicílios).
Stateless: fit() é no-op; transform() não aprende parâmetros dos dados.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.preprocessing.base import BaseFeatureTransformer


class CategoricalTransformer(BaseFeatureTransformer):
    """
    One-hot dummies para variáveis categóricas.
    
    Necessárias para regressão linear (sem assumir ordem).
    drop_first controla se a primeira categoria é dropada (evita multicolinearidade).

    Config (preprocessing.yaml → categorical_encoding):
        - column: "Transmission"
          one_hot_prefix: "trans"
          drop_first: true

    Exemplo:
        encoder = CategoricalTransformer(categoricals=config['categorical_encoding'], logger=logger)
        df = encoder.fit_transform(df)
    """

    def __init__(self, categoricals: list[dict], logger: Any = None) -> None:
        self.categoricals = categoricals
        self.logger = logger

    def fit(self, X: pd.DataFrame, y=None) -> "CategoricalTransformer":
        """
        Fit é no-op para este transformador (stateless).
        Apenas valida que as colunas categóricas existem.
        """
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Converte variáveis categóricas em dummies e adiciona ao DataFrame.
        """
        X = X.copy()
        criadas: list[str] = []

        for spec in self.categoricals:
            col_name = spec["column"]
            prefix = spec["one_hot_prefix"]
            drop_first = spec["drop_first"]

            if col_name not in X.columns:
                self._warn(
                    "CategoricalTransformer: coluna '%s' ausente — pulando criação de dummies",
                    col_name,
                )
                continue

            # ── One-hot dummies ───────────────────────────────────────────────────
            dummies = pd.get_dummies(
                X[col_name], 
                prefix=prefix, 
                drop_first=drop_first,
                dtype=int
            )
            
            # Concatenar dummies ao DataFrame
            X = pd.concat([X, dummies], axis=1)
            
            if drop_first:
                # Remover a coluna categórica original
                X = X.drop(columns=[col_name])
            
            n_dummies = len(dummies.columns)
            self._log(
                "CategoricalTransformer: '%s' convertida em %d dummies (prefix='%s', drop_first=%s)",
                col_name, n_dummies, prefix, drop_first,
            )
            criadas.append(col_name)

        self._log("CategoricalTransformer: %d coluna(s) categóricas processadas: %s", len(criadas), criadas)
        return X

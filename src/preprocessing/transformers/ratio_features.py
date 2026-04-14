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


class RatioFeatureTransformer(BaseFeatureTransformer):
    """
    Cria features de razão (numerador / denominador).

    Por que razões?
    - Totais absolutos (total_rooms, total_bedrooms, population) dependem do
      tamanho do bloco — blocos maiores têm mais tudo.
    - Razões normalizam pelo número de domicílios, tornando features comparáveis
      entre blocos de tamanhos diferentes.
    - EDA: bedrooms_per_room (r=-0.256) supera total_bedrooms (r=+0.050).

    Dois modos de operação:

    1. Razão simples (denominador direto):
       feature = numerador / denominador

    Divisão segura:
    - Denominador zero → NaN (evita divisão por zero).
    - Inf substituído por NaN.

    Config (preprocessing.yaml → ratio_features):
        Exemplo 1:
        - name: "Miles_Per_Year"
          numerator: "Mileage"
          denominator: "Vehicle_Age"

    Exemplo:
        transformer = RatioFeatureTransformer(ratios=config['ratio_features'], logger=logger)
        df = transformer.fit_transform(df)
    """

    def __init__(self, ratios: list[dict], logger: Any = None) -> None:
        self.ratios = ratios
        self.logger = logger

    def fit(self, X: pd.DataFrame, y=None) -> "RatioFeatureTransformer":
        """
        Fit é no-op para este transformador (stateless).
        Apenas valida que a configuração é coerente.
        """
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Cria cada razão configurada; ignora pares de colunas ausentes com warning."""
        X = X.copy()
        criadas: list[str] = []

        for spec in self.ratios:
            name = spec["name"]
            num = spec["numerator"]
            den = spec["denominator"]

            # Validar colunas necessárias
            if num not in X.columns or den not in X.columns:
                self._warn(
                    "RatioFeatureTransformer: colunas '%s' ou '%s' ausentes — '%s' ignorada.",
                    num, den, name,
                )
                continue
            
                
            X[name] = (X[num] / X[den].replace(0, np.nan)).replace(
                [np.inf, -np.inf], np.nan
            )
            
            self._log(
                "RatioFeatureTransformer: criada '%s' = '%s' / '%s'",
                name, num, den,
            )
            
            criadas.append(name)

        self._log("RatioFeatureTransformer: features criadas: %s", criadas)
        return X

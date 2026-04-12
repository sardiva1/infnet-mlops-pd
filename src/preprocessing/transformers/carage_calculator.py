from src.preprocessing.base import BaseFeatureTransformer

import logging
from typing import Any

class CarAgeTransformer(BaseFeatureTransformer):
    """
    Transformador que calcula a idade do veículo baseado no ano do modelo.
    
    Args:
        reference_year: Ano de referência para cálculo da idade (padrão: 2026)
        year_column: Nome da coluna com o ano do modelo (padrão: 'Model_Year')
    """
    
    def __init__(self, reference_year: int = 2026, year_column: str = "Model_Year", logger: Any = None):
        self.reference_year = reference_year
        self.year_column = year_column
        
    def fit(self, X, y=None):
        """Não há aprendizado neste transformer, apenas retorna a si mesmo."""
        return self
    
    def transform(self, X):
        """Calcula Vehicle_Age e remove a coluna original de ano."""
        X_copy = X.copy()
        X_copy['Vehicle_Age'] = self.reference_year - X_copy[self.year_column]
        
        # Remover a coluna original de ano para evitar multicolinearidade
        X_copy = X_copy.drop(columns=[self.year_column])
        
        return X_copy


# Exemplo de uso:
# transformer = CarAgeTransformer(reference_year=2026, year_column='Model_Year')
# df_transformed = transformer.fit_transform(df)
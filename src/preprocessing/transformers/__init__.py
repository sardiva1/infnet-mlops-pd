"""
transformers/ — Transformadores de Feature Engineering para o pipeline de pré-processamento.

Classes disponíveis:
  • CarAgeTransformer: Calcula idade do veículo (reference_year - Model_Year)
  • TransmissionEncoder: One-hot encoding da coluna Transmission
  • FuelTypeEncoder: One-hot encoding da coluna Fuel_Type
  • RatioFeatureTransformer: Cria features de razão normalizadas
  • FeatureSelector: Seleciona subconjunto final de features
  • StandardScalerTransformer: Escalador stateful (use apenas após split treino/teste)
"""
from src.preprocessing.transformers.carage_calculator import CarAgeTransformer
from src.preprocessing.transformers.categorical_encoder import CategoricalTransformer
from src.preprocessing.transformers.ratio_features import RatioFeatureTransformer
from src.preprocessing.transformers.feature_selector import FeatureSelector
from src.preprocessing.transformers.stateful import StandardScalerTransformer

__all__ = [
    "CarAgeTransformer",
    "CategoricalTransformer",
    "RatioFeatureTransformer",
    "FeatureSelector",
    "StandardScalerTransformer",
]

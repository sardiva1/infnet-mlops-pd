"""
production_app/utils/pipeline_utils.py — Cadeia completa de pré-processamento para inferência.

Replica exatamente as transformações de preprocessamento.py usando as mesmas
classes de transformadores de src/ e as mesmas configurações do YAML.

Entrada bruta (features originais fornecidas pelo usuário):
    Brand, Model_Year, Engine_Size, Fuel_Type, Transmission, Mileage, Doors, 
    Owner_Count, Horsepower

Saída: DataFrame de uma única linha pronto para predição — contém apenas as colunas
features_to_keep definidas em config/preprocessing.yaml, excluindo o target (Price).

Ordem da cadeia (espelha preprocessamento.py):
    1. CarAgeTransformer         — Model_Year → Vehicle_Age
    2. CategoricalTransformer    — Transmission, Fuel_Type → one-hot encoding
    3. RatioFeatureTransformer   — razões: Miles_Per_Year, HP_per_Liter
    4. FeatureSelector           — mantém apenas features_to_keep (sem target)

NOTA: StandardScalerTransformer e FeatureReducer NÃO são aplicados aqui.
Eles residem dentro do sklearn Pipeline encapsulado pelo MLflow
(imputer → scaler → reducer → estimador) e são aplicados automaticamente
pelo modelo ao chamar model.predict().
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from datetime import date

# ── Bootstrap de paths: torna src/ importável independentemente do CWD ────────
_HERE         = Path(__file__).resolve().parent   # production_app/utils/
_APP_DIR      = _HERE.parent                      # production_app/
_PROJECT_ROOT = _APP_DIR.parent                   # infnet-mlops-pd/
_CONFIG_DIR   = _PROJECT_ROOT / "config"
_DATA_DIR     = _PROJECT_ROOT / "data"

for _p in [str(_PROJECT_ROOT), str(_CONFIG_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.preprocessing import (
    CarAgeTransformer,
    CategoricalTransformer,
    RatioFeatureTransformer,
    FeatureSelector,
)


# ─────────────────────────────────────────────────────────────────────────────
# Carregamento de config (cache em nível de módulo — lido apenas uma vez)
# ─────────────────────────────────────────────────────────────────────────────

def _carregar_config_preprocessing() -> dict[str, Any]:
    """Carrega preprocessing.yaml sem depender do PipelineContext completo."""
    caminho = _CONFIG_DIR / "preprocessing.yaml"
    with open(caminho, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


_CFG_PREP: dict[str, Any] = _carregar_config_preprocessing()

# Coluna target e lista de features a manter (excluindo o target)
_TARGET_COL: str = _CFG_PREP.get("feature_selection", {}).get("target", "Price")
_FEATURES_TO_KEEP: list[str] = [
    c for c in _CFG_PREP.get("feature_selection", {}).get("features_to_keep", [])
    if c != _TARGET_COL
]

# ── Caminhos dos parquets ─────────────────────────────────────────────────────
_PARQUET_PROCESSADO = _DATA_DIR / "processed" / "car_price.parquet"
_PARQUET_FEATURES   = _DATA_DIR / "features"  / "car_price_features.parquet"

# ─────────────────────────────────────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────────────────────────────────────

def preprocessar_entradas(raw: dict[str, Any]) -> pd.DataFrame:
    """
    Converte um dicionário de entradas brutas do usuário em features prontas para o modelo.

    Aplica a cadeia completa de feature engineering na mesma ordem do pipeline
    de treinamento, garantindo consistência entre treino e inferência.

    Parâmetros
    ----------
    raw : dict
        Chaves obrigatórias:
            Brand, Model_Year, Engine_Size, Fuel_Type, Transmission, Mileage,
            Doors, Owner_Count, Horsepower

    Retorna
    -------
    pd.DataFrame
        DataFrame de uma linha com as colunas features_to_keep (sem target).

    Lança
    -----
    KeyError
        Se alguma chave obrigatória estiver ausente em `raw`.
    """
    df = pd.DataFrame([raw])

    # 1. Transformar Model_Year em Vehicle_Age (stateless)
    car_age_cfg = _CFG_PREP.get("car_age", {})
    df = CarAgeTransformer(
        reference_year=car_age_cfg.get("reference_year", date.today().year),
        year_column=car_age_cfg.get("year_column", "Model_Year"),
    ).fit_transform(df)

    # 2. Encoding categórico (Transmission, Fuel_Type)
    categorical_cfg = _CFG_PREP.get("categorical_encoding", [])
    df = CategoricalTransformer(
        categoricals=categorical_cfg,
    ).fit_transform(df)

    # 3. Features de razão
    
    ratios_cfg = _CFG_PREP.get("ratio_features", [])
    df = RatioFeatureTransformer(
        ratios=ratios_cfg
    ).fit_transform(df)

    # 4. Seleção de features (sem target)
    df = FeatureSelector(
        features_to_keep=_FEATURES_TO_KEEP
    ).fit_transform(df)

    # 5. Reindexar para garantir todas as colunas (especialmente dummies)
    df = df.reindex(columns=_FEATURES_TO_KEEP, fill_value=0)
    
    # 6. Substituir NaN por valores padrão (evita erro de validação)
    # NaN pode surgir de: HP_per_Liter com Engine_Size=0, Miles_Per_Year com Vehicle_Age=0, etc.
    for col in df.columns:
        if df[col].isna().any():
            # Usar mediana das colunas numéricas do dataset de treinamento como fallback
            if col in _FEATURES_TO_KEEP:
                # Carregar medianas do parquet de features (para inferência consistente)
                df_features = pd.read_parquet(_PARQUET_FEATURES)
                median_val = df_features[col].median()
                df[col] = df[col].fillna(median_val)

    return df


def obter_parquet_features() -> pd.DataFrame:
    """
    Retorna o parquet de features completo (usado pela página de monitoramento).

    Lança
    -----
    FileNotFoundError
        Se o parquet de features não existir (preprocessamento ainda não executado).
    """
    if not _PARQUET_FEATURES.exists():
        raise FileNotFoundError(
            f"Parquet de features não encontrado: {_PARQUET_FEATURES}\n"
            "Execute notebooks/preprocessamento.py antes de iniciar a aplicação."
        )
    return pd.read_parquet(_PARQUET_FEATURES)


def obter_colunas_features() -> list[str]:
    """Retorna os nomes das features esperadas pelo modelo."""
    return list(_FEATURES_TO_KEEP)


def obter_colunas_features_brutas() -> list[str]:
    """Retorna os nomes originais das features (como aparecem no parquet)."""
    return list(_FEATURES_TO_KEEP)

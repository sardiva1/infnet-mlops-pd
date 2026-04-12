"""
preprocessing/pipeline_builder.py — Construtor do Pipeline de Pré-processamento.

Responsabilidade única: ler a configuração do YAML e montar um sklearn.Pipeline
com os transformadores stateless na ordem correta.

Princípio de design — Separação entre política e mecanismo:
  • Política  → config/preprocessing.yaml  (O QUÊ transformar e com quais parâmetros)
  • Mecanismo → este arquivo + transformers/ (COMO executar cada transformação)

⚠  Transformadores stateful (StandardScalerTransformer)
   NÃO são incluídos aqui. Eles devem ser aplicados DENTRO do pipeline de
   modelagem (modelagem.py), APÓS o split treino/holdout, para evitar data leakage.
"""
from __future__ import annotations

import logging
from typing import Any
from datetime import date

from sklearn.pipeline import Pipeline

from src.preprocessing.transformers import (
    CarAgeTransformer,
    CategoricalTransformer,
    RatioFeatureTransformer,
    FeatureSelector,
)


class PreprocessingPipelineBuilder:
    """
    Constrói um sklearn.Pipeline de feature engineering a partir do config YAML.

    A ordem das etapas é fixa e reflete as dependências entre transformações:
    1. CarAgeTransformer         — usa apenas Model_Year, gera Vehicle_Age e remove Model_Year
    2. RatioFeatureTransformer   — razões usam colunas originais
    3. FeatureSelector           — seleciona o subconjunto final (deve ser o último)

    Uso:
        builder = PreprocessingPipelineBuilder(config=preprocessing_cfg, logger=logger)
        pipeline = builder.build()
        df_transformado = pipeline.fit_transform(df)
    """

    def __init__(self, config: dict[str, Any], logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger

    def build(self) -> Pipeline:
        """
        Monta e retorna o sklearn.Pipeline com todas as etapas configuradas.

        Returns:
            sklearn.Pipeline pronto para fit_transform().

        Raises:
            KeyError: Se uma seção obrigatória estiver ausente no config.
        """
        # Extrai configurações
        car_age_config = self.config.get("car_age", {})
        categorical_config = self.config.get("categorical_encoding", [])
        ratio_config = self.config.get("ratio_features", [])
        feature_selection_config = self.config.get("feature_selection", {})  
        
        # Monta as etapas do pipeline
        etapas = [
            ("idade_carro", CarAgeTransformer(
                reference_year=car_age_config.get("reference_year", date.today().year),
                year_column=car_age_config.get("year_column", "Model_Year"),
                logger=self.logger,
            )),
            ("categoricas", CategoricalTransformer(
                categoricals=categorical_config,
                logger=self.logger,
            )),
            ("razoes", RatioFeatureTransformer(
                ratios=ratio_config,
                logger=self.logger,
            )),
            ("selecao", FeatureSelector(
                features_to_keep=feature_selection_config.get("features_to_keep", []),
                logger=self.logger,
            )),
        ]

        if self.logger:
            self.logger.info(
                "PreprocessingPipelineBuilder: pipeline montado com %d etapas: %s",
                len(etapas),
                [nome for nome, _ in etapas],
            )

        return Pipeline(etapas)

"""Pacote raiz src — exporta a API pública do pipeline."""
from src.core.context import PipelineContext
from src.ingestion.downloader import KaggleDownloader
from src.ingestion.parquet_writer import CsvToParquetIngester
from src.quality import (
    GeExpectationResolver,
    GreatExpectationsValidator,
    QualityReportWriter,
    QualityValidator,
)
from src.preprocessing import (
    PreprocessingStep,
    PreprocessingPipelineBuilder,
    BaseFeatureTransformer,
    CarAgeTransformer,
    RatioFeatureTransformer,
    FeatureSelector,
    StandardScalerTransformer,
)

__all__ = [
    # Orquestração
    "PipelineContext",
    # Ingestão
    "KaggleDownloader",
    "CsvToParquetIngester",
    # Qualidade
    "QualityValidator",
    "GeExpectationResolver",
    "GreatExpectationsValidator",
    "QualityReportWriter",
    # Pré-processamento
    "PreprocessingStep",
    "PreprocessingPipelineBuilder",
    "BaseFeatureTransformer",
    "CarAgeTransformer",
    "RatioFeatureTransformer",
    "FeatureSelector",
    "StandardScalerTransformer",
]

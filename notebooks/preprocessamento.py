# %%
# ─────────────────────────────────────────────────────────────────────────────
# Aula MLOps — Pré-processamento e Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
#
# TERCEIRA etapa do pipeline de dados.
#   Entrada : data/processed/car_price.parquet  ← gerado por qualidade.py
#   Saída   : data/features/car_price_features.parquet
#
# Conceito central: SEPARAÇÃO entre política e mecanismo
#   • Política  → config/preprocessing.yaml  (O QUÊ transformar e parâmetros)
#   • Mecanismo → src/preprocessing/         (COMO executar cada transformação)

# %%
# Configura o contexto de execução (caminhos, config, logger)
import sys
from pathlib import Path

# Bootstrap: garante que root_dir esteja no sys.path antes de qualquer import de src/
_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "config")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.core.context import PipelineContext
from src.preprocessing import PreprocessingStep

# PipelineContext.from_notebook resolve a raiz do projeto a partir do __file__
# e garante que src/ e config/ estejam no sys.path.
ctx = PipelineContext.from_notebook(__file__)

# %%
# Executa a etapa completa:
#   1. Carrega data/processed/car_price.parquet
#   2. Constrói o sklearn.Pipeline a partir de config/preprocessing.yaml
#   3. Aplica fit_transform (todas as etapas stateless)
#   4. Persiste o resultado em data/features/car_price_features.parquet
#   5. Loga schema, shape, valores ausentes e métricas de saída
step = PreprocessingStep(ctx)
step.run()
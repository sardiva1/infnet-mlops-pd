# %%
# Configuração do Ambiente
import sys
from pathlib import Path
# %%
# definições
ROOT_DIR = Path(__file__).resolve().parent.parent
SECRETS_PATH = ROOT_DIR / 'secrets.env'
CONFIG_DIR = ROOT_DIR / 'config'
PIPELINE_CONFIG = CONFIG_DIR / 'pipeline.yaml'
DATA_CONFIG = CONFIG_DIR / 'data.yaml'
PATHS_LIST = [str(ROOT_DIR), str(CONFIG_DIR)]

# adicionar o root e o config no meu path do sistema
for _p in PATHS_LIST:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.utils.logger import get_logger
from src.utils.config_loader import load_yaml
from src.ingestion import ingest_csv_to_parquet
from src.dowloader import check_kaggle_credentials, list_remote_files, download_dataset
# %%
# fazer a leitura dos arquivos de configuração
data_cfg = load_yaml(DATA_CONFIG)
pipeline_cfg = load_yaml(PIPELINE_CONFIG)

# %%
# obtendo a configuração do log
log_cfg = pipeline_cfg.get('logging', {})

# criar o logger
logger = get_logger(
    name='ingestao',
    logging_config=log_cfg
)

# %%
if check_kaggle_credentials(secrets_path=SECRETS_PATH):
    logger.info('Kaggle Credential set!')
else:
    logger.error('Kaggle Credentials not set')
# %%
# discovery de dados
kaggle_cfg = data_cfg.get('kaggle', {})
dataset = kaggle_cfg.get('dataset')
file_pattern = kaggle_cfg.get('file_pattern', "*.csv")
expected_files = kaggle_cfg.get('expected_files')

logger.info('Dataset  : %s', dataset)
logger.info('Padrão   : %s', file_pattern)
logger.info('Arquivos : %s', expected_files or '(auto-descoberta)')

if not expected_files:
    expected_files = list_remote_files(
        dataset=dataset,
        file_pattern=file_pattern,
        logging_config=log_cfg
    )
    logger.info('Arquivos encontrados: %s', expected_files)

paths_cfg = pipeline_cfg.get('paths', {})
raw_dir = ROOT_DIR / paths_cfg.get('raw_data_dir', 'data/raw')
raw_dir.mkdir(parents=True, exist_ok=True)
exec_cfg = pipeline_cfg.get('execution', {})
skip_download = exec_cfg.get('skip_download_if_exists', False)
force_download = exec_cfg.get('force_redownload', False)

# dowload dos dados - fase Extract de um ETL / ELT
downloaded = download_dataset(
    dataset=dataset,
    expected_files=expected_files,
    destination_dir=raw_dir,
    skip_if_exists=skip_download,
    force=force_download,
    logging_config=log_cfg
)
logger.info('Arquivos prontos: %d', len(downloaded))

# verificar o conteúdo do diretório raw
for f in sorted(raw_dir.glob('*.csv')):
    logger.info('  %s (%.1f KB)', f.name, f.stat().st_size / 1024)

# %%
# definir o caminho de saída do Parquet
processed_dir = ROOT_DIR / paths_cfg.get('processed_data_dir', 'data/processed')
output_path = processed_dir / paths_cfg.get('output_filename', 'data.parquet')

logger.info('Saída: %s', output_path)

# obtendo configurações de processamento
ingest_cfg = data_cfg.get('ingest', {})
compression = ingest_cfg.get('compression', 'snappy')
chunk_size = ingest_cfg.get('chunk_size_rows', 50_000)
validate = ingest_cfg.get('validate_schema', True)
schema_cfg = data_cfg.get('schema', {})
required_cols = schema_cfg.get('required_columns')
skip_ingest = exec_cfg.get('skip_ingest_if_exists', False)
force_ingest = exec_cfg.get('force_ingest', False)

result_path = ingest_csv_to_parquet(
    raw_dir=raw_dir,
    output_path=output_path,
    compression=compression,
    chunk_size_rows=chunk_size,
    validate_schema=validate,
    skip_if_exists=skip_ingest,
    force=force_ingest,
    logging_config=log_cfg
)
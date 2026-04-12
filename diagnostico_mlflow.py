"""
Diagnóstico de configuração do MLflow.

Execute este script para verificar se o MLflow está configurado corretamente:
    python diagnostico_mlflow.py
"""
import sys
from pathlib import Path
from urllib.parse import urlparse

import mlflow

ROOT_DIR = Path(__file__).resolve().parent

def diagnosticar():
    """Executa diagnóstico completo da configuração MLflow."""
    print("=" * 70)
    print("DIAGNÓSTICO MLFLOW")
    print("=" * 70)
    
    # 1. Verificar configuração
    print("\n1️⃣  CONFIGURAÇÃO:")
    print(f"   Root Dir: {ROOT_DIR}")
    print(f"   Arquivo MLruns.db: {ROOT_DIR / 'mlruns.db'}")
    print(f"   Arquivo MLruns.db existe? {(ROOT_DIR / 'mlruns.db').exists()}")
    
    # 2. URI do tracking
    print("\n2️⃣  TRACKING URI:")
    
    # Simula a lógica do tracker.py
    tracking_uri = "sqlite:///mlruns.db"
    print(f"   URI Original: {tracking_uri}")
    
    if tracking_uri.startswith('sqlite:///'):
        db_path = tracking_uri[10:]  # Remove 'sqlite:///'
        if not Path(db_path).is_absolute():
            db_path = str(ROOT_DIR / db_path)
        resolved_uri = f'sqlite:///{db_path}'
    else:
        resolved_uri = tracking_uri
    
    print(f"   URI Resolvida: {resolved_uri}")
    print(f"   Caminho do DB: {resolved_uri.replace('sqlite:///', '')}")
    
    # 3. Tentar conectar ao MLflow
    print("\n3️⃣  CONEXÃO COM MLFLOW:")
    try:
        mlflow.set_tracking_uri(resolved_uri)
        print(f"   ✓ Tracking URI configurado com sucesso")
        
        # Configurar experimento
        mlflow.set_experiment('diagnostico-test')
        print(f"   ✓ Experimento configurado")
        
        # Tentar criar um run
        with mlflow.start_run(run_name='test_run'):
            mlflow.log_param('test_param', 'value')
            mlflow.log_metric('test_metric', 1.0)
            run_id = mlflow.active_run().info.run_id
            print(f"   ✓ Run criado: {run_id}")
        
        print(f"   ✓ Run fechado com sucesso")
        
    except Exception as e:
        print(f"   ✗ Erro: {e}")
        return False
    
    # 4. Verificar backend store
    print("\n4️⃣  BACKEND STORE:")
    try:
        # Lista experiments
        experiments = mlflow.search_experiments(max_results=5)
        print(f"   ✓ Experiments encontrados: {len(experiments)}")
        for exp in experiments:
            print(f"      - {exp.name} (id={exp.experiment_id})")
    except Exception as e:
        print(f"   ✗ Erro ao listar experiments: {e}")
    
    # 5. Dicas finais
    print("\n5️⃣  PARA VISUALIZAR OS RESULTADOS:")
    print(f"   Execute no terminal:")
    print(f"   mlflow ui --backend-store-uri {resolved_uri}")
    print(f"   Depois acesse: http://localhost:5000")
    
    print("\n" + "=" * 70)
    print("✓ DIAGNÓSTICO CONCLUÍDO COM SUCESSO")
    print("=" * 70)
    return True

if __name__ == '__main__':
    try:
        sucesso = diagnosticar()
        sys.exit(0 if sucesso else 1)
    except Exception as e:
        print(f"\n✗ ERRO FATAL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

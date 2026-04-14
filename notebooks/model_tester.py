"""
production_app/utils/model_tester.py — Testador local para o modelo treinado.

Responsabilidades:
1. Carregar dados de teste
2. Fazer predições em lote
3. Calcular métricas de validação (RMSE, MAE, R², MAPE)
4. Gerar relatório comparativo entre predições e valores reais
5. Identificar erros significativos

Exemplo de uso:
    >>> tester = ModelTester(db_uri="sqlite:///mlruns.db")
    >>> tester.testar_com_parquet("data/features/car_price_features.parquet")
    >>> print(tester.relatorio())
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import mlflow.pyfunc

# ── Bootstrap de paths ────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent  # production_app/utils/
_PROJECT_ROOT = _HERE.parent.parent  # infnet-mlops-pd/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.modeling.metrics import calcular_metricas


class ModelTester:
    """
    Testador local para validar o modelo treinado contra dados reais.

    Atributos
    ---------
    db_uri : str
        URI do banco SQLite do MLflow
    modelo : mlflow.pyfunc.PyFuncModel, opcional
        Modelo carregado (None até chamar testar_*)
    X_teste : pd.DataFrame, opcional
        Features de teste
    y_teste : pd.Series, opcional
        Target de teste
    y_previsto : np.ndarray, opcional
        Predições do modelo
    metricas : dict, opcional
        Métricas calculadas (RMSE, MAE, R², MAPE)
    logger : logging.Logger
        Logger para mensagens diagnósticas
    """

    def __init__(self, db_uri: str, logger: logging.Logger | None = None) -> None:
        """
        Inicializa o testador.

        Parâmetros
        ----------
        db_uri : str
            URI do banco SQLite do MLflow (ex: "sqlite:///mlruns.db")
        logger : logging.Logger, opcional
            Logger customizado
        """
        self.db_uri = db_uri
        self.modelo: mlflow.pyfunc.PyFuncModel | None = None
        self.X_teste: pd.DataFrame | None = None
        self.y_teste: pd.Series | None = None
        self.y_previsto: np.ndarray | None = None
        self.metricas: dict[str, float] | None = None
        
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

    def carregar_modelo(self) -> None:
        """
        Carrega o modelo registrado do MLflow.

        Lança
        -----
        mlflow.exceptions.MlflowException
            Se o modelo não estiver registrado.
        """
        mlflow.set_tracking_uri(self.db_uri)
        try:
            self.modelo = mlflow.pyfunc.load_model("models:/car-pricing-best/latest")
            self.logger.info("✓ Modelo carregado com sucesso")
        except Exception as e:
            self.logger.error(f"✗ Erro ao carregar modelo: {e}")
            raise

    def testar_com_parquet(self, caminho_parquet: str) -> None:
        """
        Testa o modelo com dados do parquet completo (X + y).

        Parâmetros
        ----------
        caminho_parquet : str
            Caminho do parquet com features + target (ex: "data/features/car_price_features.parquet")
        """
        if self.modelo is None:
            self.carregar_modelo()

        caminho = Path(caminho_parquet)
        if not caminho.exists():
            raise FileNotFoundError(f"Parquet não encontrado: {caminho}")

        df = pd.read_parquet(caminho)
        
        # Separar features (tudo exceto Price) e target
        if "Price" not in df.columns:
            raise ValueError("Parquet deve conter coluna 'Price' como target")
        
        self.y_teste = df["Price"]
        self.X_teste = df.drop(columns=["Price"])
        
        self.logger.info(f"✓ Dados carregados: {len(self.X_teste)} amostras")
        self._fazer_predicoes()

    def testar_com_csv(self, caminho_csv: str) -> None:
        """
        Testa o modelo com dados do CSV.

        Parâmetros
        ----------
        caminho_csv : str
            Caminho do CSV com features + target
        """
        if self.modelo is None:
            self.carregar_modelo()

        caminho = Path(caminho_csv)
        if not caminho.exists():
            raise FileNotFoundError(f"CSV não encontrado: {caminho}")

        df = pd.read_csv(caminho)
        
        if "Price" not in df.columns:
            raise ValueError("CSV deve conter coluna 'Price' como target")
        
        self.y_teste = df["Price"]
        self.X_teste = df.drop(columns=["Price"])
        
        self.logger.info(f"✓ Dados carregados: {len(self.X_teste)} amostras")
        self._fazer_predicoes()

    def _fazer_predicoes(self) -> None:
        """Executa as predições e calcula métricas."""
        if self.modelo is None or self.X_teste is None or self.y_teste is None:
            raise RuntimeError("Modelo e dados devem estar carregados antes de fazer predições")

        try:
            self.y_previsto = self.modelo.predict(self.X_teste)
            self.metricas = calcular_metricas(self.y_teste.values, self.y_previsto)
            self.logger.info("✓ Predições realizadas com sucesso")
        except Exception as e:
            self.logger.error(f"✗ Erro ao fazer predições: {e}")
            raise

    def prever_manual(self, features_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Faz uma predição para um único carro com features fornecidas manualmente.

        Parâmetros
        ----------
        features_dict : dict
            Dicionário com as features do carro. Deve conter as mesmas colunas
            que o modelo espera (sem a coluna Price).
            
            Exemplo:
            {
                'Brand': 'Toyota',
                'Model_Year': 2020,
                'Engine_Size': 2.0,
                'Fuel_Type': 'Diesel',
                'Transmission': 'Manual',
                'Mileage': 50000,
                'Doors': 4,
                'Owner_Count': 2,
                'Horsepower': 200,
                'Vehicle_Age': 4,
                'Miles_Per_Year': 12500,
                'Brand_Avg_Price': 46000,
                'HP_per_Liter': 100,
                ...
            }

        Retorna
        -------
        dict com chaves:
            - 'predicao': float - Valor predito em dólares
            - 'features': dict - Features utilizadas
            - 'timestamp': str - Horário da predição
        """
        if self.modelo is None:
            self.carregar_modelo()

        try:
            # Converter para DataFrame (esperado pelo modelo)
            df_features = pd.DataFrame([features_dict])
            
            # Fazer predição
            predicao = self.modelo.predict(df_features)[0]
            
            resultado = {
                'predicao': float(predicao),
                'features': features_dict,
                'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            
            self.logger.info(f"✓ Predição manual: ${predicao:,.2f}")
            return resultado
            
        except Exception as e:
            self.logger.error(f"✗ Erro ao fazer predição manual: {e}")
            raise

    def prever_interativo(self) -> dict[str, Any]:
        """
        Interface interativa para o usuário inserir features e fazer uma predição.

        Retorna
        -------
        dict
            Resultado da predição (mesma estrutura de prever_manual())
        """
        if self.modelo is None:
            self.carregar_modelo()

        print("\n" + "=" * 70)
        print("PREDIÇÃO INTERATIVA - Insira os valores das features")
        print("=" * 70)

        features_dict = {
            'Brand': input("Brand (ex: Toyota): ").strip(),
            'Model_Year': int(input("Model_Year (ex: 2020): ")),
            'Engine_Size': float(input("Engine_Size em litros (ex: 2.0): ")),
            'Fuel_Type': input("Fuel_Type (ex: Diesel): ").strip(),
            'Transmission': input("Transmission (ex: Manual): ").strip(),
            'Mileage': int(input("Mileage em km (ex: 50000): ")),
            'Doors': int(input("Doors (ex: 4): ")),
            'Owner_Count': int(input("Owner_Count (ex: 2): ")),
            'Horsepower': int(input("Horsepower (ex: 200): ")),
        }

        print("\n⏳ Processando predição...")
        return self.prever_manual(features_dict)

    def relatorio_predicao_manual(self, resultado_predicao: dict[str, Any]) -> str:
        """
        Gera um relatório formatado para uma predição manual.

        Parâmetros
        ----------
        resultado_predicao : dict
            Retorno de prever_manual() ou prever_interativo()

        Retorna
        -------
        str
            Relatório formatado
        """
        linhas = [
            "=" * 70,
            "RESULTADO DA PREDIÇÃO",
            "=" * 70,
            f"Timestamp: {resultado_predicao['timestamp']}",
            f"",
            f"VALOR PREDITO: ${resultado_predicao['predicao']:,.2f}",
            f"",
            f"FEATURES UTILIZADAS:",
        ]

        for chave, valor in resultado_predicao['features'].items():
            linhas.append(f"  {chave}: {valor}")

        linhas.append("=" * 70)
        return "\n".join(linhas)

    def relatorio(self) -> str:
        """
        Gera um relatório formatado das métricas de teste.

        Retorna
        -------
        str
            Relatório com métricas e estatísticas
        """
        if self.metricas is None:
            return "⚠️  Nenhuma predição realizada ainda. Execute testar_com_parquet() ou testar_com_csv()"

        rmse = self.metricas.get("rmse", float("nan"))
        mae = self.metricas.get("mae", float("nan"))
        r2 = self.metricas.get("r2", float("nan"))
        mape = self.metricas.get("mape", float("nan"))

        linhas = [
            "=" * 70,
            "RELATÓRIO DE TESTE DO MODELO",
            "=" * 70,
            f"Amostras testadas: {len(self.y_teste)}",
            f"",
            f"MÉTRICAS DE DESEMPENHO:",
            f"  RMSE (Root Mean Squared Error): ${rmse:,.2f}",
            f"  MAE  (Mean Absolute Error):     ${mae:,.2f}",
            f"  R²   (Coeficiente de Determinação): {r2:.4f}",
            f"  MAPE (Mean Absolute Percentage Error): {mape:.2f}%",
            f"",
            f"ESTATÍSTICAS DOS DADOS REAIS:",
            f"  Mínimo:  ${self.y_teste.min():,.2f}",
            f"  Máximo:  ${self.y_teste.max():,.2f}",
            f"  Média:   ${self.y_teste.mean():,.2f}",
            f"  Mediana: ${self.y_teste.median():,.2f}",
            f"  Desvio:  ${self.y_teste.std():,.2f}",
            f"",
            f"ESTATÍSTICAS DAS PREDIÇÕES:",
            f"  Mínimo:  ${self.y_previsto.min():,.2f}",
            f"  Máximo:  ${self.y_previsto.max():,.2f}",
            f"  Média:   ${self.y_previsto.mean():,.2f}",
            f"  Mediana: ${np.median(self.y_previsto):,.2f}",
            f"  Desvio:  ${self.y_previsto.std():,.2f}",
            "=" * 70,
        ]
        return "\n".join(linhas)

    def diagnostico_erros(self, percentil: int = 95) -> pd.DataFrame:
        """
        Identifica as piores predições (erros mais significativos).

        Parâmetros
        ----------
        percentil : int, default=95
            Percentil para filtrar erros elevados (ex: 95 = top 5% de erros)

        Retorna
        -------
        pd.DataFrame
            DataFrame com Index, Valor_Real, Previsto, Erro_Absoluto, Erro_Percentual
        """
        if self.y_previsto is None or self.y_teste is None:
            raise RuntimeError("Predições não disponíveis. Execute testar_com_parquet() primeiro")

        erros_absolutos = np.abs(self.y_previsto - self.y_teste.values)
        erros_percentuais = (erros_absolutos / self.y_teste.values) * 100

        threshold = np.percentile(erros_absolutos, percentil)

        df_erros = pd.DataFrame({
            "Valor_Real": self.y_teste.values,
            "Previsto": self.y_previsto,
            "Erro_Absoluto": erros_absolutos,
            "Erro_Percentual": erros_percentuais,
        })

        df_erros = df_erros[df_erros["Erro_Absoluto"] >= threshold].sort_values(
            "Erro_Absoluto", ascending=False
        )

        return df_erros

    def salvar_predicoes(self, caminho_csv: str) -> None:
        """
        Salva as predições e erros em um CSV.

        Parâmetros
        ----------
        caminho_csv : str
            Caminho de saída para o CSV
        """
        if self.y_previsto is None or self.y_teste is None:
            raise RuntimeError("Predições não disponíveis")

        erros_absolutos = np.abs(self.y_previsto - self.y_teste.values)
        erros_percentuais = (erros_absolutos / self.y_teste.values) * 100

        df_resultado = pd.DataFrame({
            "Valor_Real": self.y_teste.values,
            "Previsto": self.y_previsto,
            "Erro_Absoluto": erros_absolutos,
            "Erro_Percentual": erros_percentuais,
        })

        df_resultado.to_csv(caminho_csv, index=False)
        self.logger.info(f"✓ Predições salvas em: {caminho_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# Script de teste rápido
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Configurar path para acesso aos módulos
    projeto_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(projeto_root))

    # Executar teste
    db_uri = "sqlite:///mlruns.db"
    parquet_path = "data/features/car_price_features.parquet"

    tester = ModelTester(db_uri=db_uri)
    
    try:
        tester.testar_com_parquet(parquet_path)
        print(tester.relatorio())
        
        print("\n" + "=" * 70)
        print("TOP 10 PIORES PREDIÇÕES (Top 5% de erros)")
        print("=" * 70)
        print(tester.diagnostico_erros(percentil=95).head(10).to_string())
        
        # Salvar predições
        tester.salvar_predicoes("outputs/test_predictions.csv")
        
        # ── Exemplo 1: Predição manual com dicionário ──────────────────────
        print("\n" + "=" * 70)
        print("EXEMPLO 1: PREDIÇÃO MANUAL COM DICIONÁRIO")
        print("=" * 70)
        
        exemplo_features = {
            'Brand': 'Toyota',
            'Model_Year': 2020,
            'Engine_Size': 2.0,
            'Fuel_Type': 'Diesel',
            'Transmission': 'Manual',
            'Mileage': 50000,
            'Doors': 4,
            'Owner_Count': 2,
            'Horsepower': 200,
            'Vehicle_Age': 4,
            'Miles_Per_Year': 12500,
            'Brand_Avg_Price': 46000,
            'HP_per_Liter': 100,
        }
        
        resultado = tester.prever_manual(exemplo_features)
        print(tester.relatorio_predicao_manual(resultado))
        
        # ── Exemplo 2: Predição interativa ────────────────────────────────
        # Descomente a linha abaixo para ativar entrada interativa:
        # resultado_interativo = tester.prever_interativo()
        # print(tester.relatorio_predicao_manual(resultado_interativo))
        
    except Exception as e:
        print(f"✗ Erro durante teste: {e}")
        raise

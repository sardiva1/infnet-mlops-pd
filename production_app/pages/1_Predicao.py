"""
pages/1_Predicao.py — Interface Streamlit: predição de preço veicular usado.

O usuário insere as 9 features originais do dataset de preço de carros.
Ao submeter:
  1. A cadeia completa de pré-processamento (utils/pipeline_utils.py) converte as
     entradas brutas nas ~14 features engenheiradas que o modelo espera.
  2. O modelo é carregado diretamente do banco SQLite do MLflow (sem servidor REST).
  3. As métricas de IC (cv_rmse_std, holdout_rmse) são recuperadas via MlflowClient.
  4. O IC de 95% é calculado como: y_hat ± 1,96 × (cv_rmse_std / √n_folds)
  5. Os resultados são exibidos com gauge visual e barra de intervalo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# ── Bootstrap de paths ────────────────────────────────────────────────────────
_PAGE_DIR     = Path(__file__).resolve().parent   # pages/
_APP_DIR      = _PAGE_DIR.parent                  # production_app/
_PROJECT_ROOT = _APP_DIR.parent                   # demo_projeto/

for _p in [str(_APP_DIR), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.pipeline_utils import preprocessar_entradas
from utils.model_utils import (
    carregar_modelo,
    prever_individual,
    obter_params_ic,
    calcular_intervalo_confianca,
    registrar_predicao,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuração da página
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predição de Preço Veicular Usado",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 Predição de Preço Veicular Usado")
st.markdown(
    """
    Informe as features originais do veículo abaixo.
    A aplicação executa o **pipeline completo de feature engineering**
    (razões como milhas por ano, preço médio por marca, razão HP/cilindrada, encoding categórico) e
    carrega o modelo diretamente do **banco SQLite do MLflow** — sem servidor externo.
    """
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — configuração do banco MLflow
# ─────────────────────────────────────────────────────────────────────────────
_URI_PADRAO = f"sqlite:///{_PROJECT_ROOT / 'mlruns.db'}"

with st.sidebar:
    st.header("⚙️ Configurações MLflow")
    db_uri = st.text_input(
        "URI do banco SQLite",
        value=_URI_PADRAO,
        help=(
            "URI do banco SQLite gerado por modelagem.py.\n"
            "Exemplos:\n"
            "  sqlite:///mlruns.db\n"
            "  sqlite:////caminho/absoluto/mlruns.db"
        ),
    )

    st.divider()
    st.markdown(
        """
        **Como gerar o banco:**
        ```bash
        cd infnet-mlops-pd
        python notebooks/ingestao.py
        python notebooks/preprocessamento.py
        python notebooks/modelagem.py
        ```
        Depois execute a aplicação:
        ```bash
        streamlit run production_app/app.py
        ```
        """
    )

# ─────────────────────────────────────────────────────────────────────────────
# Cache do modelo (recarregado apenas quando o URI muda)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Carregando modelo do MLflow...")
def _modelo_em_cache(uri: str):
    """Carrega e armazena em cache o modelo MLflow para evitar recarregamentos."""
    return carregar_modelo(uri)


# ─────────────────────────────────────────────────────────────────────────────
# Formulário de entrada — features originais do veículo
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Características do veículo")
st.caption(
    "Todas as features derivadas (milhas por ano, preço médio por marca, potência por cilindrada, encoding) "
    "são calculadas automaticamente pelo pipeline de pré-processamento."
)

col1, col2, col3 = st.columns(3)

with col1:
    brand = st.selectbox(
        "Marca",
        options=["Ford", "Hyundai", "BMW", "Honda", "Tesla", "Toyota"],
        help="Marca do veículo",
    )
    model_year = st.slider(
        "Ano do modelo",
        min_value=2000,
        max_value=2024,
        value=2020,
        step=1,
        help="Ano de fabrico do veículo",
    )
    engine_size = st.number_input(
        "Cilindrada (L)",
        min_value=0.5,
        max_value=8.0,
        value=2.0,
        step=0.1,
        help="Volume total do motor em litros",
    )

with col2:
    fuel_type = st.selectbox(
        "Tipo de combustível",
        options=["Diesel", "Hybrid", "Electric", "Petrol"],
        help="Tipo de combustível do veículo",
    )
    transmission = st.selectbox(
        "Transmissão",
        options=["Manual", "Automatic"],
        help="Tipo de transmissão",
    )
    mileage = st.number_input(
        "Quilometragem (km)",
        min_value=0,
        max_value=500_000,
        value=50_000,
        step=1,
        help="Quilometros rodados",
    )

with col3:
    doors = st.slider(
        "Número de portas",
        min_value=2,
        max_value=4,
        value=4,
        step=1,
        help="Número de portas do veículo",
    )
    owner_count = st.slider(
        "Número de proprietários anteriores",
        min_value=1,
        max_value=5,
        value=2,
        step=1,
        help="Quantas vezes o veículo foi vendido",
    )
    horsepower = st.number_input(
        "Potência (CV)",
        min_value=50,
        max_value=500,
        value=200,
        step=1,
        help="Potência do motor em cavalos-vapor",
    )

# ─────────────────────────────────────────────────────────────────────────────
# Predição
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
btn_prever = st.button(
    "🔮 Calcular Preço do Veículo",
    type="primary",
    use_container_width=True,
)

if btn_prever:
    entradas_brutas = {
        "Brand":         brand,
        "Model_Year":    model_year,
        "Engine_Size":   engine_size,
        "Fuel_Type":     fuel_type,
        "Transmission":  transmission,
        "Mileage":       mileage,
        "Doors":         doors,
        "Owner_Count":   owner_count,
        "Horsepower":    horsepower,
    }

    # ── Passo 1: pipeline de pré-processamento ────────────────────────────────
    with st.spinner("Executando pipeline de pré-processamento..."):
        try:
            features_df = preprocessar_entradas(entradas_brutas)
            pipeline_ok = True
        except Exception as exc:
            st.error(f"❌ Erro no pré-processamento: {exc}")
            pipeline_ok = False

    # ── Passo 2: carregamento do modelo e predição ────────────────────────────
    if pipeline_ok:
        with st.spinner("Carregando modelo e realizando predição..."):
            try:
                modelo = _modelo_em_cache(db_uri)
                y_hat  = prever_individual(features_df, modelo)
                predicao_ok = True
            except Exception as exc:
                st.error(
                    f"❌ Erro ao carregar o modelo ou realizar predição: {exc}\n\n"
                    f"Verifique se o banco SQLite está em: `{db_uri}`"
                )
                predicao_ok = False

    # ── Passo 3: parâmetros de IC via MLflow ──────────────────────────────────
    if pipeline_ok and predicao_ok:
        with st.spinner("Recuperando parâmetros de IC do MLflow..."):
            try:
                params_ic = obter_params_ic(db_uri)
                inferior, superior = calcular_intervalo_confianca(
                    y_hat=y_hat,
                    cv_rmse_std=params_ic["cv_rmse_std"],
                )
                ic_ok = True
            except Exception as exc:
                st.warning(
                    f"⚠️ Não foi possível recuperar o IC do MLflow: {exc}\n\n"
                    "A predição pontual é exibida abaixo sem intervalo de confiança."
                )
                ic_ok = False

    # ── Passo 4: exibição dos resultados ──────────────────────────────────────
    if pipeline_ok and predicao_ok:
        st.divider()
        st.subheader("📊 Resultados da Predição")

        # ── Registrar predição em MLflow ──────────────────────────────────────
        try:
            run_id_predicao = registrar_predicao(
                db_uri=db_uri,
                features_entrada=entradas_brutas,
                features_engenheiradas=features_df,
                y_hat=y_hat,
                intervalo_inferior=inferior if ic_ok else None,
                intervalo_superior=superior if ic_ok else None,
                modelo_versao=params_ic.get("versao_modelo", "") if ic_ok else "",
            )
            st.success(f"✅ Predição registrada em MLflow (Run ID: `{run_id_predicao[:8]}...`)")
        except Exception as exc:
            st.warning(f"⚠️ Não foi possível registrar a predição em MLflow: {exc}")

        col_res1, col_res2, col_res3 = st.columns([2, 1, 1])

        with col_res1:
            st.metric(
                label="Preço Previsto do Veículo",
                value=f"${y_hat:,.0f}",
                help="Estimativa pontual do modelo carregado do MLflow.",
            )
            if ic_ok:
                from utils.model_utils import _N_FOLDS_CV
                st.markdown(
                    f"""
                    **Intervalo de Confiança de 95%:**
                    &nbsp;&nbsp; ${inferior:,.0f} &nbsp; – &nbsp; ${superior:,.0f}

                    *EP = cv\\_rmse\\_std / √{_N_FOLDS_CV} =
                    {params_ic["cv_rmse_std"]:,.0f} / √{_N_FOLDS_CV} =
                    {params_ic["cv_rmse_std"] / (_N_FOLDS_CV ** 0.5):,.0f}*
                    """
                )

                # Barra visual do IC
                amplitude = superior - inferior
                st.progress(
                    min(int((y_hat - inferior) / (amplitude + 1e-9) * 100), 100),
                    text=f"Predição dentro do intervalo  |  Amplitude: ${amplitude:,.0f}",
                )

        with col_res2:
            if ic_ok:
                st.metric("Limite inferior (IC 95%)", f"${inferior:,.0f}")
                st.metric("Limite superior (IC 95%)", f"${superior:,.0f}")

        with col_res3:
            if ic_ok:
                st.metric("RMSE holdout", f"${params_ic['holdout_rmse']:,.0f}")
                st.metric("cv_rmse_std", f"${params_ic['cv_rmse_std']:,.0f}")
                st.caption(f"Versão do modelo: {params_ic['versao_modelo']}")
                st.caption(f"Run ID: `{params_ic['run_id'][:8]}...`")

        # ── Inspeção das features engenheiradas ───────────────────────────────
        with st.expander("🔍 Inspecionar features engenheiradas"):
            st.caption(
                f"{len(features_df.columns)} features enviadas ao modelo "
                f"(9 originais → {len(features_df.columns)} após pipeline completo)"
            )
            st.dataframe(
                features_df.T.rename(columns={0: "valor"}),
                use_container_width=True,
                height=600,
            )

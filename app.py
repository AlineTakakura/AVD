import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------------------------------------------------------
# 1. Configuração Inicial da Página
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard de Filmes (2024-2026)",
    page_icon="🎬",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 2. Carregamento e Caching dos Dados
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    # Caminho ajustado para a pasta data/raw conforme a boas práticas do projeto
    caminho = "data/raw/recent_movies_2024_2026.csv"
    df = pd.read_csv(caminho)
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "❌ Ficheiro `recent_movies_2024_2026.csv` não encontrado na pasta `data/raw/`."
        "\n\nCertifique-se de que o ficheiro CSV está guardado em `data/raw/recent_movies_2024_2026.csv`."
    )
    st.stop()

# -----------------------------------------------------------------------------
# 3. Barra Lateral (Sidebar) - Filtros Interativos
# -----------------------------------------------------------------------------
st.sidebar.title("🔍 Filtros de Análise")

# Filtro de Ano de Lançamento
anos_disponiveis = sorted(df["release_year"].dropna().unique().astype(int))
anos_selecionados = st.sidebar.multiselect(
    "Ano de Lançamento:",
    options=anos_disponiveis,
    default=anos_disponiveis
)

# Filtro de Classificação Indicativa
ratings_disponiveis = sorted(df["content_rating"].dropna().unique().tolist())
ratings_selecionados = st.sidebar.multiselect(
    "Classificação Indicativa:",
    options=ratings_disponiveis,
    default=ratings_disponiveis
)

# Aplicação dos Filtros no DataFrame
df_filtered = df[
    (df["release_year"].isin(anos_selecionados)) &
    (df["content_rating"].isin(ratings_selecionados))
]

# -----------------------------------------------------------------------------
# 4. Cabeçalho Principal
# -----------------------------------------------------------------------------
st.title("🎬 Dashboard de Filmes e Entretenimento (2024–2026)")
st.markdown("Visualização de métricas de desempenho financeiro e avaliações do mercado cinematográfico recente.")
st.divider()

# -----------------------------------------------------------------------------
# 5. Indicadores Chave de Desempenho (KPIs / Cartões)
# -----------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total de Filmes", f"{len(df_filtered):,}")

with col2:
    media_nota = df_filtered["rating_out_of_10"].mean()
    st.metric("Nota Média (0-10)", f"{media_nota:.2f}" if pd.notnull(media_nota) else "N/A")

with col3:
    bilheteria_media = df_filtered["box_office_usd"].mean()
    st.metric("Bilheteria Média", f"${bilheteria_media:,.0f}" if pd.notnull(bilheteria_media) else "N/A")

with col4:
    orcamento_medio = df_filtered["budget_usd"].mean()
    st.metric("Orçamento Médio", f"${orcamento_medio:,.0f}" if pd.notnull(orcamento_medio) else "N/A")

st.divider()

# -----------------------------------------------------------------------------
# 6. Gráficos Interativos (Plotly Express)
# -----------------------------------------------------------------------------
g_col1, g_col2 = st.columns(2)

with g_col1:
    st.subheader("📊 Relatório: Orçamento vs. Bilheteria")
    fig_scatter = px.scatter(
        df_filtered,
        x="budget_usd",
        y="box_office_usd",
        color="content_rating",
        hover_name="title",
        hover_data=["director", "rating_out_of_10"],
        labels={
            "budget_usd": "Orçamento ($ USD)",
            "box_office_usd": "Bilheteria ($ USD)",
            "content_rating": "Classificação"
        },
        title="Relação entre Investimento e Arrecadação"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with g_col2:
    st.subheader("🏆 Top 10 Filmes por Bilheteria")
    top_movies = df_filtered.nlargest(10, "box_office_usd")
    fig_bar = px.bar(
        top_movies,
        x="box_office_usd",
        y="title",
        orientation="h",
        color="rating_out_of_10",
        labels={
            "box_office_usd": "Bilheteria ($ USD)",
            "title": "Título do Filme",
            "rating_out_of_10": "Nota"
        },
        title="Maiores Arrecadações de Bilheteria"
    )
    fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------------------------------------------------------------
# 7. Visualização da Tabela de Dados Integrada
# -----------------------------------------------------------------------------
st.divider()
st.subheader("📋 Tabela de Dados Filtrada")

with st.expander("Clique para visualizar ou ocultar os dados em formato de tabela"):
    colunas_exibicao = [
        "title", "release_year", "release_month", "genres",
        "rating_out_of_10", "box_office_usd", "budget_usd", "director"
    ]
    st.dataframe(df_filtered[colunas_exibicao], use_container_width=True)
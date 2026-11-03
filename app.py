
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# Configuração da página
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Filmes e Entretenimento | Dashboard",
    page_icon="🎬",
    layout="wide",
)

st.title("Análise de Filmes e Entretenimento")
st.caption(
    "Explore características, avaliações e indicadores dos filmes de 2024 a 2026. "
    "Os filtros da barra lateral são compartilhados pelos dois dashboards."
)


# -----------------------------------------------------------------------------
# Variáveis do dataset
# -----------------------------------------------------------------------------
VARIAVEIS_QUANTITATIVAS = [
    "release_year",
    "release_day",
    "vote_count",
    "runtime_minutes",
    "box_office_usd",
    "budget_usd",
]

VARIAVEL_CLASSE = "rating_out_of_10"
CLASSES_AVALIACAO = ["Abaixo de 7", "7 ou mais"]

COR_ABAIXO = "#7186A6"
COR_SETE_OU_MAIS = "#C58B78"
COR_DESTAQUE = "#654653"
COR_NEUTRA = "#D9D3D8"

ESCALA_CLASSES = alt.Scale(
    domain=CLASSES_AVALIACAO,
    range=[COR_ABAIXO, COR_SETE_OU_MAIS],
)


# -----------------------------------------------------------------------------
# Carregamento e preparação dos dados
# -----------------------------------------------------------------------------
@st.cache_data
def carregar_dados():
    """Carrega o CSV tratado e prepara as colunas usadas nos dashboards."""

    caminho_csv = (
        Path(__file__).resolve().parent
        / "data"
        / "clean"
        / "dataset.csv"
    )

    if not caminho_csv.exists():
        raise FileNotFoundError(
            "O arquivo data/clean/dataset.csv não foi encontrado. "
            "Confira se ele está presente no repositório."
        )

    dados = pd.read_csv(caminho_csv)

    colunas_obrigatorias = [VARIAVEL_CLASSE]

    faltantes = [
        coluna
        for coluna in colunas_obrigatorias
        if coluna not in dados.columns
    ]

    if faltantes:
        raise ValueError(
            f"Colunas obrigatórias ausentes: {', '.join(faltantes)}"
        )

    for coluna in VARIAVEIS_QUANTITATIVAS:
        if coluna in dados.columns:
            dados[coluna] = pd.to_numeric(
                dados[coluna],
                errors="coerce",
            )

    # Normaliza as classes de avaliação.
    nota = dados[VARIAVEL_CLASSE].astype("string").str.strip()
    nota_numerica = pd.to_numeric(nota, errors="coerce")

    dados[VARIAVEL_CLASSE] = nota.where(
        nota.isin(CLASSES_AVALIACAO)
    )

    mascara_numerica = nota_numerica.notna()

    dados.loc[mascara_numerica, VARIAVEL_CLASSE] = np.where(
        nota_numerica[mascara_numerica] >= 7,
        "7 ou mais",
        "Abaixo de 7",
    )

    return dados


try:
    df = carregar_dados()

except Exception as erro:
    st.error("Não foi possível carregar o dataset tratado.")
    st.exception(erro)
    st.stop()


variaveis_disponiveis = [
    coluna
    for coluna in VARIAVEIS_QUANTITATIVAS
    if coluna in df.columns and df[coluna].notna().any()
]

if not variaveis_disponiveis:
    st.error(
        "O dataset não possui as variáveis quantitativas esperadas."
    )
    st.stop()


# -----------------------------------------------------------------------------
# Filtros compartilhados
# -----------------------------------------------------------------------------
with st.sidebar:

    st.header("Filtros")

    st.caption(
        "As seleções abaixo afetam os dois dashboards."
    )

    variaveis_selecionadas = st.multiselect(
        "Variáveis para os gráficos de distribuição e correlação",
        options=variaveis_disponiveis,
        default=variaveis_disponiveis,
        key="variaveis_selecionadas",
    )

    classes_disponiveis = [
        classe
        for classe in CLASSES_AVALIACAO
        if classe in df[VARIAVEL_CLASSE].dropna().unique()
    ]

    classes_selecionadas = st.multiselect(
        "Faixa de avaliação",
        options=classes_disponiveis,
        default=classes_disponiveis,
        key="classes_selecionadas",
    )

    st.subheader("Intervalos das variáveis")

    intervalos = {}

    for variavel in variaveis_disponiveis:

        valores = df[variavel].dropna()

        minimo = float(valores.min())
        maximo = float(valores.max())

        if minimo == maximo:

            intervalos[variavel] = (minimo, maximo)

            st.caption(
                f"{variavel}: valor único ({minimo:g})"
            )

            continue

        valores_inteiros = np.all(
            np.isclose(
                valores.to_numpy(),
                np.round(valores.to_numpy()),
            )
        )

        passo = (
            1.0
            if valores_inteiros
            else max((maximo - minimo) / 100, 0.01)
        )

        intervalos[variavel] = st.slider(
            variavel,
            min_value=minimo,
            max_value=maximo,
            value=(minimo, maximo),
            step=passo,
            key=f"intervalo_{variavel}",
        )

    if st.button(
        "Restaurar filtros",
        use_container_width=True,
    ):

        for chave in list(st.session_state.keys()):

            if chave.startswith("intervalo_"):
                del st.session_state[chave]

        st.session_state.pop(
            "variaveis_selecionadas",
            None,
        )

        st.session_state.pop(
            "classes_selecionadas",
            None,
        )

        st.session_state.pop(
            "grafico_financeiro_selecao",
            None,
        )

        st.rerun()


# -----------------------------------------------------------------------------
# Aplicação dos filtros compartilhados
# -----------------------------------------------------------------------------
df_filtrado = df[
    df[VARIAVEL_CLASSE].isin(classes_selecionadas)
].copy()

for variavel, (inicio, fim) in intervalos.items():

    df_filtrado = df_filtrado[
        df_filtrado[variavel].between(inicio, fim)
    ]

st.caption(
    f"**{len(df_filtrado):,}** de **{len(df):,}** "
    "filmes após a aplicação dos filtros."
)

if df_filtrado.empty:

    st.warning(
        "Nenhum filme corresponde aos filtros selecionados. "
        "Amplie os intervalos ou selecione uma faixa de avaliação."
    )

    st.stop()


# -----------------------------------------------------------------------------
# Abas dos dashboards
# -----------------------------------------------------------------------------
aba1, aba2 = st.tabs(
    [
        "Dashboard 1 — Exploração dos dados",
        "Dashboard 2 — Indicadores e comparações",
    ]
)


# =============================================================================
# DASHBOARD 1 — Correlação e distribuição
# =============================================================================
with aba1:

    st.header("Exploração das variáveis")

    st.write(
        "Analise a relação entre as variáveis quantitativas e observe "
        "como seus valores estão distribuídos entre as faixas de avaliação."
    )

    coluna_correlacao, coluna_distribuicao = st.columns(
        [1, 1.25]
    )

    # -------------------------------------------------------------------------
    # Correlação
    # -------------------------------------------------------------------------
    with coluna_correlacao:

        st.subheader("Correlação de Spearman")

        variaveis_correlacao = [
            variavel
            for variavel in variaveis_selecionadas
            if df_filtrado[variavel].notna().sum() > 1
        ]

        if len(variaveis_correlacao) == 0:

            st.info(
                "Selecione pelo menos uma variável com dados válidos."
            )

        elif len(variaveis_correlacao) == 1:

            st.info(
                "Selecione pelo menos duas variáveis "
                "para calcular a correlação."
            )

        else:

            matriz = df_filtrado[
                variaveis_correlacao
            ].corr(method="spearman")

            matriz_longa = (
                matriz.rename_axis("Variável Y")
                .rename_axis("Variável X", axis="columns")
                .stack()
                .rename("Correlação")
                .reset_index()
            )

            base_heatmap = alt.Chart(
                matriz_longa
            ).encode(
                x=alt.X(
                    "Variável X:N",
                    sort=variaveis_correlacao,
                    title=None,
                    axis=alt.Axis(labelAngle=-40),
                ),
                y=alt.Y(
                    "Variável Y:N",
                    sort=variaveis_correlacao,
                    title=None,
                ),
            )

            heatmap = base_heatmap.mark_rect().encode(
                color=alt.Color(
                    "Correlação:Q",
                    scale=alt.Scale(
                        domain=[-1, 0, 1],
                        range=[
                            COR_ABAIXO,
                            "#F3F0E8",
                            COR_SETE_OU_MAIS,
                        ],
                    ),
                    title="Correlação",
                ),
                tooltip=[
                    "Variável X:N",
                    "Variável Y:N",
                    alt.Tooltip(
                        "Correlação:Q",
                        format=".2f",
                    ),
                ],
            )

            valores_heatmap = base_heatmap.mark_text(
                fontSize=11
            ).encode(
                text=alt.Text(
                    "Correlação:Q",
                    format=".2f",
                ),
                color=alt.value("#26212D"),
            )

            st.altair_chart(
                (heatmap + valores_heatmap).properties(
                    height=360
                ),
                use_container_width=True,
            )

            st.caption(
                "A correlação indica associação entre variáveis, "
                "não causalidade."
            )

    # -------------------------------------------------------------------------
    # Distribuição
    # -------------------------------------------------------------------------
    with coluna_distribuicao:

        st.subheader("Distribuição das variáveis")

        separar_por_classe = st.toggle(
            "Separar por faixa de avaliação",
            value=True,
            key="separar_distribuicao_por_classe",
        )

        if not variaveis_selecionadas:

            st.info(
                "Selecione pelo menos uma variável na barra lateral."
            )

        else:

            for inicio in range(
                0,
                len(variaveis_selecionadas),
                2,
            ):

                colunas_graficos = st.columns(2)

                grupo = variaveis_selecionadas[
                    inicio:inicio + 2
                ]

                for coluna, variavel in zip(
                    colunas_graficos,
                    grupo,
                ):

                    dados_grafico = df_filtrado[
                        [variavel, VARIAVEL_CLASSE]
                    ].dropna(subset=[variavel])

                    with coluna:

                        if dados_grafico.empty:

                            st.info(
                                f"Não há dados válidos para {variavel}."
                            )

                            continue

                        grafico = alt.Chart(
                            dados_grafico
                        ).mark_bar(
                            opacity=0.85
                        ).encode(
                            x=alt.X(
                                f"{variavel}:Q",
                                bin=alt.Bin(maxbins=20),
                                title=variavel,
                            ),
                            y=alt.Y(
                                "count():Q",
                                title="Quantidade de filmes",
                            ),
                            tooltip=[
                                alt.Tooltip(
                                    "count():Q",
                                    title="Filmes",
                                )
                            ],
                        )

                        if separar_por_classe:

                            grafico = grafico.encode(
                                color=alt.Color(
                                    f"{VARIAVEL_CLASSE}:N",
                                    scale=ESCALA_CLASSES,
                                    title="Faixa de avaliação",
                                )
                            )

                        else:

                            grafico = grafico.encode(
                                color=alt.value(COR_DESTAQUE)
                            )

                        st.altair_chart(
                            grafico.properties(height=230),
                            use_container_width=True,
                        )


# =============================================================================
# DASHBOARD 2 — Indicadores e seleção vinculada
# =============================================================================
with aba2:

    st.header("Indicadores e comparações")

    st.write(
        "Selecione uma região no gráfico de orçamento e bilheteria "
        "para atualizar os indicadores e os demais gráficos."
    )

    # -------------------------------------------------------------------------
    # Gráfico financeiro e seleção
    # -------------------------------------------------------------------------
    st.subheader("Orçamento x bilheteria")

    colunas_financeiras = {
        "budget_usd",
        "box_office_usd",
        VARIAVEL_CLASSE,
    }

    # Começa com os dados que passaram pelos filtros da sidebar.
    df_dashboard2 = df_filtrado.copy()

    if colunas_financeiras.issubset(df_filtrado.columns):

        dados_financeiros = df_filtrado[
            [
                "budget_usd",
                "box_office_usd",
                VARIAVEL_CLASSE,
            ]
        ].dropna().copy()

        if dados_financeiros.empty:

            st.info(
                "Não existem pares válidos de orçamento e bilheteria "
                "após os filtros."
            )

        else:

            modo_financeiro = st.radio(
                "Como deseja visualizar os valores?",
                options=[
                    "Valores transformados em ln(USD)",
                    "Valores em USD",
                ],
                horizontal=True,
                key="modo_grafico_financeiro",
            )

            # Prepara os valores apresentados no gráfico.
            if modo_financeiro == "Valores em USD":

                dados_financeiros["orcamento_plot"] = np.exp(
                    dados_financeiros["budget_usd"]
                )

                dados_financeiros["bilheteria_plot"] = np.exp(
                    dados_financeiros["box_office_usd"]
                )

                dados_financeiros["orcamento_tooltip"] = (
                    dados_financeiros["orcamento_plot"]
                )

                dados_financeiros["bilheteria_tooltip"] = (
                    dados_financeiros["bilheteria_plot"]
                )

                titulo_x = "Orçamento (USD)"
                titulo_y = "Bilheteria (USD)"

                escala_x = alt.Scale(
                    type="log",
                    zero=False,
                )

                escala_y = alt.Scale(
                    type="log",
                    zero=False,
                )

                formato_x = "$,.2s"
                formato_y = "$,.2s"

                tooltip_orcamento = "$,.0f"
                tooltip_bilheteria = "$,.0f"

            else:

                dados_financeiros["orcamento_plot"] = (
                    dados_financeiros["budget_usd"]
                )

                dados_financeiros["bilheteria_plot"] = (
                    dados_financeiros["box_office_usd"]
                )

                dados_financeiros["orcamento_tooltip"] = (
                    dados_financeiros["budget_usd"]
                )

                dados_financeiros["bilheteria_tooltip"] = (
                    dados_financeiros["box_office_usd"]
                )

                titulo_x = "ln(Orçamento em USD)"
                titulo_y = "ln(Bilheteria em USD)"

                escala_x = alt.Scale(
                    zero=False
                )

                escala_y = alt.Scale(
                    zero=False
                )

                formato_x = ".2f"
                formato_y = ".2f"

                tooltip_orcamento = ".3f"
                tooltip_bilheteria = ".3f"

            # Seleção por intervalo no gráfico.
            selecao_financeira = alt.selection_interval(
                name="selecao_financeira",
                encodings=["x", "y"],
                clear="dblclick",
            )

            grafico_financeiro = (
                alt.Chart(dados_financeiros)
                .mark_circle(
                    size=90,
                    stroke="#F3F0E8",
                    strokeWidth=0.5,
                )
                .encode(
                    x=alt.X(
                        "orcamento_plot:Q",
                        title=titulo_x,
                        scale=escala_x,
                        axis=alt.Axis(format=formato_x),
                    ),
                    y=alt.Y(
                        "bilheteria_plot:Q",
                        title=titulo_y,
                        scale=escala_y,
                        axis=alt.Axis(format=formato_y),
                    ),
                    color=alt.Color(
                        f"{VARIAVEL_CLASSE}:N",
                        scale=ESCALA_CLASSES,
                        title="Faixa de avaliação",
                    ),
                    opacity=alt.condition(
                        selecao_financeira,
                        alt.value(0.85),
                        alt.value(0.15),
                    ),
                    tooltip=[
                        alt.Tooltip(
                            "orcamento_tooltip:Q",
                            title="Orçamento",
                            format=tooltip_orcamento,
                        ),
                        alt.Tooltip(
                            "bilheteria_tooltip:Q",
                            title="Bilheteria",
                            format=tooltip_bilheteria,
                        ),
                        alt.Tooltip(
                            f"{VARIAVEL_CLASSE}:N",
                            title="Faixa de avaliação",
                        ),
                    ],
                )
                .add_params(selecao_financeira)
            )

            # A chave muda quando o modo financeiro muda,
            # evitando reaproveitar limites de escalas diferentes.
            chave_grafico = (
                "grafico_financeiro_selecao_usd"
                if modo_financeiro == "Valores em USD"
                else "grafico_financeiro_selecao_ln"
            )

            evento_financeiro = st.altair_chart(
                grafico_financeiro.properties(
                    height=430
                ),
                use_container_width=True,
                key=chave_grafico,
                on_select="rerun",
                selection_mode="selecao_financeira",
            )

            st.caption(
                "Arraste o mouse sobre o gráfico para selecionar uma região. "
                "Dê dois cliques para limpar a seleção."
            )

            # Recupera os limites selecionados.
            intervalo_selecionado = (
                evento_financeiro.selection.get(
                    "selecao_financeira",
                    {},
                )
            )

            limites_x = intervalo_selecionado.get("x")
            limites_y = intervalo_selecionado.get("y")

            # Aplica a seleção ao DataFrame original.
            if (
                limites_x is not None
                and limites_y is not None
                and len(limites_x) == 2
                and len(limites_y) == 2
            ):

                minimo_x, maximo_x = limites_x
                minimo_y, maximo_y = limites_y

                # No modo USD, converte os limites para ln(USD),
                # pois os dados originais estão armazenados nessa escala.
                if modo_financeiro == "Valores em USD":

                    minimo_x, maximo_x = np.log(
                        [minimo_x, maximo_x]
                    )

                    minimo_y, maximo_y = np.log(
                        [minimo_y, maximo_y]
                    )

                df_dashboard2 = df_filtrado[
                    df_filtrado["budget_usd"].between(
                        minimo_x,
                        maximo_x,
                    )
                    & df_filtrado["box_office_usd"].between(
                        minimo_y,
                        maximo_y,
                    )
                ].copy()

    else:

        st.info(
            "O gráfico financeiro precisa das colunas "
            "budget_usd e box_office_usd."
        )

    # -------------------------------------------------------------------------
    # Indicadores
    # -------------------------------------------------------------------------
    st.divider()

    st.subheader("Indicadores dos filmes selecionados")

    total_filmes = len(df_dashboard2)

    quantidade_nota_sete = int(
        df_dashboard2[VARIAVEL_CLASSE]
        .eq("7 ou mais")
        .sum()
    )

    percentual_nota_sete = (
        quantidade_nota_sete / total_filmes * 100
        if total_filmes > 0
        else 0
    )

    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric(
        "Total de filmes",
        f"{total_filmes:,}",
    )

    k2.metric(
        "Filmes com nota ≥ 7",
        f"{quantidade_nota_sete:,}",
    )

    k3.metric(
        "Percentual com nota ≥ 7",
        f"{percentual_nota_sete:.1f}%",
    )

    if (
        "vote_count" in df_dashboard2.columns
        and df_dashboard2["vote_count"].notna().any()
    ):

        k4.metric(
            "Média de votos",
            f"{df_dashboard2['vote_count'].mean():,.0f}",
        )

    else:

        k4.metric(
            "Média de votos",
            "—",
        )

    if (
        "runtime_minutes" in df_dashboard2.columns
        and df_dashboard2["runtime_minutes"].notna().any()
    ):

        k5.metric(
            "Duração média",
            f"{df_dashboard2['runtime_minutes'].mean():.1f} min",
        )

    else:

        k5.metric(
            "Duração média",
            "—",
        )

    st.caption(
        f"{total_filmes:,} filmes considerados nos indicadores "
        "após os filtros e a seleção do gráfico."
    )

    # -------------------------------------------------------------------------
    # Verifica se a seleção possui resultados
    # -------------------------------------------------------------------------
    if df_dashboard2.empty:

        st.warning(
            "A seleção atual não contém filmes. "
            "Dê dois cliques no gráfico financeiro para limpar a seleção."
        )

    else:

        # ---------------------------------------------------------------------
        # Filmes por faixa de avaliação
        # ---------------------------------------------------------------------
        col_quantidade, col_duracao = st.columns(2)

        with col_quantidade:

            st.subheader(
                "Filmes por faixa de avaliação"
            )

            contagem = (
                df_dashboard2[VARIAVEL_CLASSE]
                .value_counts()
                .reindex(
                    CLASSES_AVALIACAO,
                    fill_value=0,
                )
                .rename_axis("Faixa de avaliação")
                .reset_index(
                    name="Quantidade de filmes"
                )
            )

            grafico_contagem = alt.Chart(
                contagem
            ).mark_bar(
                cornerRadiusTopLeft=5,
                cornerRadiusTopRight=5,
            ).encode(
                x=alt.X(
                    "Faixa de avaliação:N",
                    sort=CLASSES_AVALIACAO,
                    title=None,
                ),
                y=alt.Y(
                    "Quantidade de filmes:Q",
                    title="Quantidade de filmes",
                ),
                color=alt.Color(
                    "Faixa de avaliação:N",
                    scale=ESCALA_CLASSES,
                    legend=None,
                ),
                tooltip=[
                    "Faixa de avaliação:N",
                    "Quantidade de filmes:Q",
                ],
            )

            st.altair_chart(
                grafico_contagem.properties(
                    height=300
                ),
                use_container_width=True,
            )

        # ---------------------------------------------------------------------
        # Duração por faixa de avaliação
        # ---------------------------------------------------------------------
        with col_duracao:

            st.subheader(
                "Duração por faixa de avaliação"
            )

            dados_duracao = df_dashboard2[
                [
                    VARIAVEL_CLASSE,
                    "runtime_minutes",
                ]
            ].dropna()

            if not dados_duracao.empty:

                grafico_duracao = alt.Chart(
                    dados_duracao
                ).mark_boxplot(
                    size=55
                ).encode(
                    x=alt.X(
                        f"{VARIAVEL_CLASSE}:N",
                        sort=CLASSES_AVALIACAO,
                        title="Faixa de avaliação",
                    ),
                    y=alt.Y(
                        "runtime_minutes:Q",
                        title="Duração (minutos)",
                    ),
                    color=alt.Color(
                        f"{VARIAVEL_CLASSE}:N",
                        scale=ESCALA_CLASSES,
                        legend=None,
                    ),
                )

                st.altair_chart(
                    grafico_duracao.properties(
                        height=300
                    ),
                    use_container_width=True,
                )

            else:

                st.info(
                    "Não há dados válidos de duração "
                    "para os filmes selecionados."
                )

        # ---------------------------------------------------------------------
        # Votos por faixa de avaliação
        # ---------------------------------------------------------------------
        st.divider()

        col_votos, col_ano = st.columns(2)

        with col_votos:

            st.subheader(
                "Quantidade de votos por faixa de avaliação"
            )

            dados_votos = df_dashboard2[
                [
                    VARIAVEL_CLASSE,
                    "vote_count",
                ]
            ].dropna()

            if not dados_votos.empty:

                grafico_votos = alt.Chart(
                    dados_votos
                ).mark_boxplot(
                    size=55
                ).encode(
                    x=alt.X(
                        f"{VARIAVEL_CLASSE}:N",
                        sort=CLASSES_AVALIACAO,
                        title="Faixa de avaliação",
                    ),
                    y=alt.Y(
                        "vote_count:Q",
                        title="Quantidade de votos",
                    ),
                    color=alt.Color(
                        f"{VARIAVEL_CLASSE}:N",
                        scale=ESCALA_CLASSES,
                        legend=None,
                    ),
                )

                st.altair_chart(
                    grafico_votos.properties(
                        height=300
                    ),
                    use_container_width=True,
                )

            else:

                st.info(
                    "Não há dados válidos de votos "
                    "para os filmes selecionados."
                )

        # ---------------------------------------------------------------------
        # Distribuição por ano de lançamento
        # ---------------------------------------------------------------------
        with col_ano:

            st.subheader(
                "Distribuição por ano de lançamento"
            )

            dados_anos = df_dashboard2[
                [
                    "release_year",
                    VARIAVEL_CLASSE,
                ]
            ].dropna()

            if not dados_anos.empty:

                grafico_anos = alt.Chart(
                    dados_anos
                ).mark_bar(
                    opacity=0.85
                ).encode(
                    x=alt.X(
                        "release_year:Q",
                        bin=alt.Bin(step=1),
                        title="Ano de lançamento",
                    ),
                    y=alt.Y(
                        "count():Q",
                        title="Quantidade de filmes",
                    ),
                    color=alt.Color(
                        f"{VARIAVEL_CLASSE}:N",
                        scale=ESCALA_CLASSES,
                        title="Faixa de avaliação",
                    ),
                    tooltip=[
                        alt.Tooltip(
                            "count():Q",
                            title="Quantidade de filmes",
                        )
                    ],
                )

                st.altair_chart(
                    grafico_anos.properties(
                        height=300
                    ),
                    use_container_width=True,
                )

            else:

                st.info(
                    "Não há anos de lançamento válidos "
                    "para os filmes selecionados."
                )

        # ---------------------------------------------------------------------
        # Estatísticas descritivas
        # ---------------------------------------------------------------------
        with st.expander(
            "Consultar estatísticas descritivas"
        ):

            colunas_estatisticas = [
                variavel
                for variavel in variaveis_selecionadas
                if variavel in df_dashboard2.columns
            ]

            if colunas_estatisticas:

                estatisticas = df_dashboard2[
                    colunas_estatisticas
                ].describe().T

                estatisticas["assimetria"] = (
                    df_dashboard2[
                        colunas_estatisticas
                    ].skew()
                )

                estatisticas["curtose"] = (
                    df_dashboard2[
                        colunas_estatisticas
                    ].kurtosis()
                )

                st.dataframe(
                    estatisticas.round(2),
                    use_container_width=True,
                )

            else:

                st.info(
                    "Selecione ao menos uma variável "
                    "para consultar as estatísticas."
                )

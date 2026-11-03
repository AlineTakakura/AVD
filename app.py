
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


# Configuração
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

VARIAVEL_CLASSE = "rating_out_of_10"

VARIAVEIS = [
    "release_year",
    "release_day",
    "vote_count",
    "runtime_minutes",
    "box_office_usd",
    "budget_usd",
]

CLASSES = ["Abaixo de 7", "7 ou mais"]

COR_ABAIXO = "#7186A6"
COR_SETE = "#C58B78"
COR_DESTAQUE = "#654653"

ESCALA_CLASSES = alt.Scale(
    domain=CLASSES,
    range=[COR_ABAIXO, COR_SETE],
)


# Carregamento dos dados
@st.cache_data
def carregar_dados():
    caminho = (
        Path(__file__).resolve().parent
        / "data"
        / "clean"
        / "dataset.csv"
    )

    if not caminho.exists():
        raise FileNotFoundError(
            "Não encontrei data/clean/dataset.csv. "
            "Confira se o arquivo está no repositório."
        )

    dados = pd.read_csv(caminho)

    if VARIAVEL_CLASSE not in dados.columns:
        raise ValueError(
            f"A coluna {VARIAVEL_CLASSE} não existe no CSV."
        )

    for coluna in VARIAVEIS:
        if coluna in dados.columns:
            dados[coluna] = pd.to_numeric(
                dados[coluna],
                errors="coerce",
            )

    nota = dados[VARIAVEL_CLASSE].astype("string").str.strip()
    nota_num = pd.to_numeric(nota, errors="coerce")

    dados[VARIAVEL_CLASSE] = nota.where(nota.isin(CLASSES))

    validas = nota_num.notna()

    dados.loc[validas, VARIAVEL_CLASSE] = np.where(
        nota_num[validas] >= 7,
        "7 ou mais",
        "Abaixo de 7",
    )

    return dados


try:
    df = carregar_dados()
except Exception as erro:
    st.error("Não foi possível carregar o dataset.")
    st.exception(erro)
    st.stop()


variaveis_disponiveis = [
    coluna
    for coluna in VARIAVEIS
    if coluna in df.columns and df[coluna].notna().any()
]

if not variaveis_disponiveis:
    st.error("O dataset não possui as variáveis quantitativas esperadas.")
    st.stop()


# Filtros compartilhados
with st.sidebar:
    st.header("Filtros")
    st.caption("As seleções abaixo afetam os dois dashboards.")

    variaveis = st.multiselect(
        "Variáveis para distribuição e correlação",
        variaveis_disponiveis,
        default=variaveis_disponiveis,
        key="variaveis_selecionadas",
    )

    classes_disponiveis = [
        classe
        for classe in CLASSES
        if classe in df[VARIAVEL_CLASSE].dropna().unique()
    ]

    classes = st.multiselect(
        "Faixa de avaliação",
        classes_disponiveis,
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
            st.caption(f"{variavel}: {minimo:g}")
            continue

        inteiros = np.all(np.isclose(valores, np.round(valores)))

        passo = (
            1.0
            if inteiros
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

    if st.button("Restaurar filtros", use_container_width=True):
        for chave in list(st.session_state.keys()):
            if chave.startswith("intervalo_"):
                del st.session_state[chave]

        st.session_state.pop("variaveis_selecionadas", None)
        st.session_state.pop("classes_selecionadas", None)

        st.rerun()


df_filtrado = df[
    df[VARIAVEL_CLASSE].isin(classes)
].copy()

for variavel, (inicio, fim) in intervalos.items():
    df_filtrado = df_filtrado[
        df_filtrado[variavel].between(inicio, fim)
    ]

st.caption(
    f"**{len(df_filtrado):,}** de **{len(df):,}** filmes após os filtros."
)

if df_filtrado.empty:
    st.warning("Nenhum filme corresponde aos filtros. Amplie os intervalos.")
    st.stop()


# Abas
aba1, aba2 = st.tabs([
    "Dashboard 1 — Exploração dos dados",
    "Dashboard 2 — Indicadores e comparações",
])


# DASHBOARD 1
with aba1:
    st.header("Exploração das variáveis")

    # Correlação de Spearman
    st.subheader("Correlação de Spearman")

    vars_corr = [
        var
        for var in variaveis
        if df_filtrado[var].notna().sum() > 1
    ]

    if len(vars_corr) < 2:
        st.info(
            "Selecione pelo menos duas variáveis para calcular a correlação."
        )

    else:
        matriz = df_filtrado[vars_corr].corr(method="spearman")

        # Mantém somente o triângulo inferior da matriz.
        linhas = []

        for i, var_y in enumerate(vars_corr):
            for j, var_x in enumerate(vars_corr):
                if j <= i:
                    valor = matriz.loc[var_y, var_x]

                    linhas.append({
                        "Variável X": var_x,
                        "Variável Y": var_y,
                        "Correlação": valor,
                    })

        dados_corr = pd.DataFrame(linhas)

        grafico_corr = alt.Chart(dados_corr).encode(
            x=alt.X(
                "Variável X:N",
                sort=vars_corr,
                scale=alt.Scale(domain=vars_corr),
                title=None,
                axis=alt.Axis(
                    labelAngle=-45,
                    labelFontSize=12,
                    labelLimit=220,
                    labelPadding=10,
                    labelOverlap=False,
                ),
            ),
            y=alt.Y(
                "Variável Y:N",
                sort=list(reversed(vars_corr)),
                scale=alt.Scale(domain=list(reversed(vars_corr))),
                title=None,
                axis=alt.Axis(
                    labelFontSize=12,
                    labelLimit=220,
                    labelPadding=8,
                ),
            ),
        )

        mapa = grafico_corr.mark_rect(
            stroke="#F5F3F0",
            strokeWidth=2,
        ).encode(
            color=alt.Color(
                "Correlação:Q",
                scale=alt.Scale(
                    domain=[-1, 0, 1],
                    range=[
                        COR_ABAIXO,
                        "#F3F0E8",
                        COR_SETE,
                    ],
                ),
                legend=alt.Legend(
                    title="Correlação",
                    orient="bottom",
                ),
            ),
            tooltip=[
                alt.Tooltip("Variável X:N"),
                alt.Tooltip("Variável Y:N"),
                alt.Tooltip("Correlação:Q", format=".2f"),
            ],
        )

        textos = grafico_corr.mark_text(
            fontSize=13,
            fontWeight="bold",
        ).encode(
            text=alt.Text("Correlação:Q", format=".2f"),
            color=alt.value("#26212D"),
        )

        grafico_final = (
            (mapa + textos)
            .properties(
                height=520,
                title="Matriz de correlação de Spearman",
            )
            .configure_axis(
                labelFontSize=12,
                titleFontSize=13,
            )
            .configure_view(
                strokeWidth=0,
            )
        )

        st.altair_chart(
            grafico_final,
            use_container_width=True,
        )

        st.caption(
            "A correlação varia de -1 a 1. Valores próximos de 1 "
            "indicam associação positiva; próximos de -1, associação "
            "negativa; e próximos de 0, pouca associação monotônica."
        )

        st.caption("A correlação indica associação, não causalidade.")

    st.divider()

    # Distribuição das variáveis
    st.subheader("Distribuição das variáveis")

    separar = st.toggle(
        "Separar por faixa de avaliação",
        value=True,
    )

    if not variaveis:
        st.info("Selecione pelo menos uma variável na barra lateral.")

    else:
        for inicio in range(0, len(variaveis), 2):
            colunas = st.columns(2)

            for coluna, variavel in zip(
                colunas,
                variaveis[inicio:inicio + 2],
            ):
                dados = df_filtrado[
                    [variavel, VARIAVEL_CLASSE]
                ].dropna(subset=[variavel])

                grafico = alt.Chart(dados).mark_bar(
                    opacity=0.85,
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

                if separar:
                    grafico = grafico.encode(
                        color=alt.Color(
                            f"{VARIAVEL_CLASSE}:N",
                            scale=ESCALA_CLASSES,
                            legend=None,
                        )
                    )
                else:
                    grafico = grafico.encode(
                        color=alt.value(COR_DESTAQUE)
                    )

                with coluna:
                    st.altair_chart(
                        grafico.properties(height=250),
                        use_container_width=True,
                    )

        # Uma única legenda para todos os gráficos de distribuição.
        if separar:
            st.markdown(
                f"""
                <div style="
                    display: flex;
                    justify-content: center;
                    gap: 28px;
                    padding: 8px;
                    font-size: 14px;
                ">
                    <span>
                        <span style="
                            display:inline-block;
                            width:12px;
                            height:12px;
                            border-radius:50%;
                            background:{COR_ABAIXO};
                            margin-right:6px;
                        "></span>
                        Abaixo de 7
                    </span>
                    <span>
                        <span style="
                            display:inline-block;
                            width:12px;
                            height:12px;
                            border-radius:50%;
                            background:{COR_SETE};
                            margin-right:6px;
                        "></span>
                        7 ou mais
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )


# DASHBOARD 2
with aba2:
    st.header("Indicadores e comparações")

    colunas_financeiras = {
        "budget_usd",
        "box_office_usd",
        VARIAVEL_CLASSE,
    }

    if colunas_financeiras.issubset(df_filtrado.columns):
        dados_financeiros = df_filtrado[
            [
                "budget_usd",
                "box_office_usd",
                VARIAVEL_CLASSE,
            ]
        ].dropna().copy()

        if not dados_financeiros.empty:
            modo = st.radio(
                "Visualização dos valores financeiros",
                [
                    "Valores transformados em ln(USD)",
                    "Valores em USD",
                ],
                horizontal=True,
                key="modo_grafico_financeiro",
                help=(
                    "O notebook aplicou np.log() às duas colunas. "
                    "A opção USD recupera os valores aproximados usando exp()."
                ),
            )

            if modo == "Valores em USD":
                dados_financeiros["orcamento_plot"] = np.exp(
                    dados_financeiros["budget_usd"]
                )

                dados_financeiros["bilheteria_plot"] = np.exp(
                    dados_financeiros["box_office_usd"]
                )

                titulo_x = "Orçamento (USD, escala logarítmica)"
                titulo_y = "Bilheteria (USD, escala logarítmica)"

                escala_x = alt.Scale(type="log", zero=False)
                escala_y = alt.Scale(type="log", zero=False)

                formato_x = "$,.2s"
                formato_y = "$,.2s"
                formato_tooltip = "$,.0f"

            else:
                dados_financeiros["orcamento_plot"] = (
                    dados_financeiros["budget_usd"]
                )

                dados_financeiros["bilheteria_plot"] = (
                    dados_financeiros["box_office_usd"]
                )

                titulo_x = "ln(Orçamento em USD)"
                titulo_y = "ln(Bilheteria em USD)"

                escala_x = alt.Scale(zero=False)
                escala_y = alt.Scale(zero=False)

                formato_x = ".2f"
                formato_y = ".2f"
                formato_tooltip = ".3f"

            # Identifica as linhas originais para aplicar a seleção aos indicadores.
            dados_financeiros["_linha"] = dados_financeiros.index

            # Seleção por arraste.
            selecao = alt.selection_interval(
                name="selecao_financeira",
                encodings=["x", "y"],
                clear="dblclick",
            )

            grafico = alt.Chart(
                dados_financeiros
            ).mark_circle(
                size=90,
                opacity=0.8,
                stroke="#F3F0E8",
                strokeWidth=0.5,
            ).encode(
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
                    selecao,
                    alt.value(0.95),
                    alt.value(0.20),
                ),
                tooltip=[
                    alt.Tooltip(
                        "orcamento_plot:Q",
                        title="Orçamento",
                        format=formato_tooltip,
                    ),
                    alt.Tooltip(
                        "bilheteria_plot:Q",
                        title="Bilheteria",
                        format=formato_tooltip,
                    ),
                    alt.Tooltip(
                        f"{VARIAVEL_CLASSE}:N",
                        title="Faixa de avaliação",
                    ),
                ],
            ).add_params(selecao)

            st.subheader("Orçamento x bilheteria")

            st.caption(
                "Arraste o mouse sobre o gráfico para selecionar uma região. "
                "Dê dois cliques para limpar a seleção."
            )

            evento = st.altair_chart(
                grafico.properties(height=430),
                use_container_width=True,
                on_select="rerun",
                selection_mode="selecao_financeira",
                key="grafico_financeiro",
            )

            # Lê os limites selecionados nos dois eixos.
            limites = evento.selection.get(
                "selecao_financeira",
                {},
            )

            limite_x = limites.get("orcamento_plot")
            limite_y = limites.get("bilheteria_plot")

            df_dashboard2 = df_filtrado.copy()

            if limite_x and limite_y:
                selecionados = dados_financeiros[
                    dados_financeiros["orcamento_plot"].between(
                        float(limite_x[0]),
                        float(limite_x[1]),
                    )
                    & dados_financeiros["bilheteria_plot"].between(
                        float(limite_y[0]),
                        float(limite_y[1]),
                    )
                ]

                df_dashboard2 = df_filtrado.loc[
                    selecionados["_linha"]
                ].copy()

                st.caption(
                    f"Região selecionada: {len(df_dashboard2):,} filmes. "
                    "Os indicadores e gráficos abaixo consideram essa seleção."
                )

            else:
                st.caption(
                    "Nenhuma região selecionada: mostrando todos os filmes filtrados."
                )

        else:
            df_dashboard2 = df_filtrado.copy()
            st.info("Não há pares válidos de orçamento e bilheteria.")

    else:
        df_dashboard2 = df_filtrado.copy()
        st.info(
            "O gráfico financeiro precisa de budget_usd e box_office_usd."
        )

    # Indicadores atualizados pela seleção financeira
    total = len(df_dashboard2)

    nota_sete = int(
        df_dashboard2[VARIAVEL_CLASSE].eq("7 ou mais").sum()
    )

    percentual = (
        nota_sete / total * 100
        if total
        else 0
    )

    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric("Total de filmes", f"{total:,}")
    k2.metric("Filmes com nota ≥ 7", f"{nota_sete:,}")
    k3.metric("Percentual com nota ≥ 7", f"{percentual:.1f}%")

    if (
        "vote_count" in df_dashboard2
        and df_dashboard2["vote_count"].notna().any()
    ):
        k4.metric(
            "Média de votos",
            f"{df_dashboard2['vote_count'].mean():,.0f}",
        )
    else:
        k4.metric("Média de votos", "—")

    if (
        "runtime_minutes" in df_dashboard2
        and df_dashboard2["runtime_minutes"].notna().any()
    ):
        k5.metric(
            "Duração média",
            f"{df_dashboard2['runtime_minutes'].mean():.1f} min",
        )
    else:
        k5.metric("Duração média", "—")

    st.divider()

    col_contagem, col_duracao = st.columns(2)

    with col_contagem:
        st.subheader("Filmes por faixa de avaliação")

        contagem = (
            df_dashboard2[VARIAVEL_CLASSE]
            .value_counts()
            .reindex(CLASSES, fill_value=0)
            .rename_axis("Faixa de avaliação")
            .reset_index(name="Quantidade de filmes")
        )

        grafico_contagem = alt.Chart(
            contagem
        ).mark_bar(
            cornerRadiusTopLeft=5,
            cornerRadiusTopRight=5,
        ).encode(
            x=alt.X(
                "Faixa de avaliação:N",
                sort=CLASSES,
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
            grafico_contagem.properties(height=300),
            use_container_width=True,
        )

    with col_duracao:
        st.subheader("Duração por faixa de avaliação")

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
                size=55,
            ).encode(
                x=alt.X(
                    f"{VARIAVEL_CLASSE}:N",
                    sort=CLASSES,
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
                tooltip=[
                    alt.Tooltip(
                        "runtime_minutes:Q",
                        format=".1f",
                    )
                ],
            )

            st.altair_chart(
                grafico_duracao.properties(height=300),
                use_container_width=True,
            )

        else:
            st.info("Não há dados válidos de duração.")

    st.divider()

    col_votos, col_anos = st.columns(2)

    with col_votos:
        st.subheader("Quantidade de votos por faixa de avaliação")

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
                size=55,
            ).encode(
                x=alt.X(
                    f"{VARIAVEL_CLASSE}:N",
                    sort=CLASSES,
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
                grafico_votos.properties(height=300),
                use_container_width=True,
            )

        else:
            st.info("Não há dados válidos de votos.")

    with col_anos:
        st.subheader("Distribuição por ano de lançamento")

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
                opacity=0.85,
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
                grafico_anos.properties(height=300),
                use_container_width=True,
            )

        else:
            st.info("Não há anos válidos para exibir.")

    with st.expander("Consultar estatísticas descritivas"):
        colunas = [
            var
            for var in variaveis
            if var in df_dashboard2.columns
        ]

        if colunas:
            estatisticas = df_dashboard2[colunas].describe().T

            estatisticas["assimetria"] = (
                df_dashboard2[colunas].skew()
            )

            estatisticas["curtose"] = (
                df_dashboard2[colunas].kurtosis()
            )

            st.dataframe(
                estatisticas.round(2),
                use_container_width=True,
            )

        else:
            st.info("Selecione ao menos uma variável.")

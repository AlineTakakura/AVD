import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import load_data


st.set_page_config(
    page_title="EDA — Filmes e Entretenimento",
    page_icon="🎬",
    layout="wide",
)

st.title("Análise Exploratória de Dados — Filmes e Entretenimento")
st.caption(
    "Dashboard baseado no notebook de EDA: distribuições, correlações e comparação "
    "entre filmes com avaliação abaixo de 7 e filmes com avaliação 7 ou mais."
)

# Variáveis usadas no notebook
VARS_QUANT = [
    "release_year",
    "release_day",
    "vote_count",
    "runtime_minutes",
    "box_office_usd",
    "budget_usd",
]
VAR_CLASSE = "rating_out_of_10"

COR_CONTEXTO = "#B0BEC5"  # Abaixo de 7
COR_DESTAQUE = "#2D223A"  # 7 ou mais
CORES_CLASSE = alt.Scale(
    domain=["Abaixo de 7", "7 ou mais"],
    range=[COR_CONTEXTO, COR_DESTAQUE],
)


@st.cache_data
def preparar_dados():
    dados = load_data().copy()

    # Garante que as colunas quantitativas possam ser usadas nos gráficos.
    for coluna in VARS_QUANT:
        if coluna in dados.columns:
            dados[coluna] = pd.to_numeric(dados[coluna], errors="coerce")

    if VAR_CLASSE not in dados.columns:
        raise ValueError(
            f"A coluna obrigatória '{VAR_CLASSE}' não foi encontrada no dataset."
        )

    # O notebook trabalha com as classes "Abaixo de 7" e "7 ou mais".
    # Aceita tanto uma coluna já categorizada quanto uma coluna numérica.
    if pd.api.types.is_numeric_dtype(dados[VAR_CLASSE]):
        dados[VAR_CLASSE] = np.where(
            dados[VAR_CLASSE] >= 7, "7 ou mais", "Abaixo de 7"
        )
    else:
        dados[VAR_CLASSE] = dados[VAR_CLASSE].astype("string").str.strip()
        dados[VAR_CLASSE] = dados[VAR_CLASSE].replace(
            {
                "Abaixo de 7": "Abaixo de 7",
                "7 ou mais": "7 ou mais",
            }
        )

    return dados


try:
    df = preparar_dados()
except Exception as erro:
    st.error(
        "Não foi possível carregar os dados. Confira se `src/data_loader.py` "
        "existe e se a função `load_data()` retorna o dataset de filmes."
    )
    st.exception(erro)
    st.stop()

# Usa somente as variáveis quantitativas que realmente existem no arquivo.
vars_disponiveis = [var for var in VARS_QUANT if var in df.columns]
if not vars_disponiveis:
    st.error("Nenhuma das variáveis quantitativas esperadas foi encontrada no dataset.")
    st.stop()

# Sidebar: filtros
with st.sidebar:
    st.header("Filtros")

    variaveis = st.multiselect(
        "Variáveis quantitativas",
        options=vars_disponiveis,
        default=vars_disponiveis,
        help="Escolha as variáveis que deseja analisar.",
    )

    classes_disponiveis = sorted(
        df[VAR_CLASSE].dropna().astype(str).unique().tolist()
    )
    classes_selecionadas = st.multiselect(
        "Classes de avaliação",
        options=classes_disponiveis,
        default=classes_disponiveis,
        help="Filtre os filmes por faixa de avaliação.",
    )

    st.subheader("Intervalos")
    intervalos = {}
    for var in variaveis:
        valores = df[var].dropna()
        if valores.empty:
            continue

        minimo = float(valores.min())
        maximo = float(valores.max())
        if minimo == maximo:
            intervalos[var] = (minimo, maximo)
            st.caption(f"{var}: todos os valores são {minimo:g}.")
        else:
            passo = 1.0 if pd.api.types.is_integer_dtype(df[var].dropna().dtype) else 0.1
            intervalos[var] = st.slider(
                f"Intervalo de {var}",
                min_value=minimo,
                max_value=maximo,
                value=(minimo, maximo),
                step=passo,
                key=f"intervalo_{var}",
            )

# Aplica os filtros escolhidos
df_filtrado = df[df[VAR_CLASSE].astype(str).isin(classes_selecionadas)].copy()
for var, (inicio, fim) in intervalos.items():
    df_filtrado = df_filtrado[df_filtrado[var].between(inicio, fim)]

st.caption(f"{len(df_filtrado):,} de {len(df):,} filmes após aplicar os filtros.")

# Dashboard 1 — indicadores gerais
st.subheader("Indicadores gerais")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric("Filmes filtrados", f"{len(df_filtrado):,}")
kpi2.metric(
    "Avaliação média",
    f"{df_filtrado[VAR_CLASSE].notna().mean() * 100:.1f}% classificados"
    if not df_filtrado.empty
    else "—",
)

if "runtime_minutes" in df_filtrado.columns and df_filtrado["runtime_minutes"].notna().any():
    kpi3.metric(
        "Duração média",
        f"{df_filtrado['runtime_minutes'].mean():.1f} min",
    )
else:
    kpi3.metric("Duração média", "—")

if "vote_count" in df_filtrado.columns and df_filtrado["vote_count"].notna().any():
    kpi4.metric("Média de votos", f"{df_filtrado['vote_count'].mean():,.0f}")
else:
    kpi4.metric("Média de votos", "—")

st.divider()

# Dashboard 2 — correlação e distribuição
col_correlacao, col_distribuicao = st.columns([1, 1.4])

with col_correlacao:
    st.subheader("Correlação de Spearman")

    vars_corr = [
        var for var in variaveis
        if var in df_filtrado.columns and df_filtrado[var].notna().sum() > 1
    ]

    if len(vars_corr) < 2:
        st.warning("Selecione pelo menos duas variáveis quantitativas com dados.")
    elif len(df_filtrado) < 3:
        st.warning("Há poucos filmes após os filtros para calcular a correlação.")
    else:
        matriz = df_filtrado[vars_corr].corr(method="spearman")
        corr = (
            matriz.rename_axis(index="var_y", columns="var_x")
            .stack()
            .rename("correlacao")
            .reset_index()
        )
        ordem = {var: i for i, var in enumerate(vars_corr)}
        corr = corr[
            corr["var_y"].map(ordem) > corr["var_x"].map(ordem)
        ]

        base = alt.Chart(corr).encode(
            x=alt.X(
                "var_x:N",
                title=None,
                sort=vars_corr[:-1],
                axis=alt.Axis(labelAngle=0, labelFontSize=11),
            ),
            y=alt.Y(
                "var_y:N",
                title=None,
                sort=vars_corr[1:],
                axis=alt.Axis(labelFontSize=11),
            ),
        )
        heatmap = base.mark_rect().encode(
            color=alt.Color(
                "correlacao:Q",
                title="Spearman",
                scale=alt.Scale(
                    scheme="redblue",
                    domain=[-1, 1],
                    reverse=True,
                ),
            ),
            tooltip=[
                "var_x:N",
                "var_y:N",
                alt.Tooltip("correlacao:Q", format=".2f"),
            ],
        )
        textos = base.mark_text(fontSize=13).encode(
            text=alt.Text("correlacao:Q", format=".2f"),
            color=alt.condition(
                "abs(datum.correlacao) > 0.5",
                alt.value("white"),
                alt.value("black"),
            ),
        )
        st.altair_chart(
            (heatmap + textos).properties(height=360),
            use_container_width=True,
        )
        st.caption(
            "A correlação varia de -1 a 1. Valores próximos de zero indicam "
            "uma associação monotônica fraca; correlação não implica causalidade."
        )

with col_distribuicao:
    st.subheader("Distribuição das variáveis")

    separar_por_classe = st.toggle(
        "Comparar por faixa de avaliação",
        value=True,
        help="Mostra as distribuições separadas por 'Abaixo de 7' e '7 ou mais'.",
    )

    if not variaveis:
        st.warning("Selecione pelo menos uma variável quantitativa.")
    elif df_filtrado.empty:
        st.warning("Nenhum filme corresponde aos filtros selecionados.")
    else:
        for inicio in range(0, len(variaveis), 2):
            colunas = st.columns(2)
            for coluna, var in zip(colunas, variaveis[inicio:inicio + 2]):
                dados_grafico = df_filtrado[[var, VAR_CLASSE]].dropna(subset=[var])
                grafico = alt.Chart(dados_grafico).mark_bar(opacity=0.8).encode(
                    x=alt.X(
                        f"{var}:Q",
                        bin=alt.Bin(maxbins=20),
                        title=var,
                    ),
                    y=alt.Y("count():Q", title="Quantidade de filmes"),
                    tooltip=[alt.Tooltip("count():Q", title="Filmes")],
                )

                if separar_por_classe:
                    grafico = grafico.encode(
                        color=alt.Color(
                            f"{VAR_CLASSE}:N",
                            title="Faixa de avaliação",
                            scale=CORES_CLASSE,
                        ),
                        tooltip=[
                            alt.Tooltip(f"{VAR_CLASSE}:N", title="Classe"),
                            alt.Tooltip("count():Q", title="Filmes"),
                        ],
                    )
                else:
                    grafico = grafico.encode(color=alt.value(COR_DESTAQUE))

                with coluna:
                    st.altair_chart(
                        grafico.properties(height=220),
                        use_container_width=True,
                    )

st.divider()

# Relação entre orçamento e bilheteria, destacada no notebook
st.subheader("Orçamento x bilheteria")
if {"budget_usd", "box_office_usd", VAR_CLASSE}.issubset(df_filtrado.columns):
    dados_relacao = df_filtrado[
        ["budget_usd", "box_office_usd", VAR_CLASSE]
    ].dropna()

    usar_log = st.checkbox(
        "Usar escala logarítmica (log10) para orçamento e bilheteria",
        value=False,
        help="Útil porque o notebook identificou forte assimetria à direita nessas variáveis.",
    )

    if usar_log:
        # Valores iguais ou menores que zero não podem ser transformados por log10.
        dados_relacao = dados_relacao[
            (dados_relacao["budget_usd"] > 0)
            & (dados_relacao["box_office_usd"] > 0)
        ].copy()
        dados_relacao["orcamento_plot"] = np.log10(dados_relacao["budget_usd"])
        dados_relacao["bilheteria_plot"] = np.log10(dados_relacao["box_office_usd"])
        titulo_x = "Orçamento (log10 USD)"
        titulo_y = "Bilheteria (log10 USD)"
    else:
        dados_relacao["orcamento_plot"] = dados_relacao["budget_usd"]
        dados_relacao["bilheteria_plot"] = dados_relacao["box_office_usd"]
        titulo_x = "Orçamento (USD)"
        titulo_y = "Bilheteria (USD)"

    if dados_relacao.empty:
        st.warning("Não há pares válidos de orçamento e bilheteria para mostrar.")
    else:
        dispersao = alt.Chart(dados_relacao).mark_circle(size=65, opacity=0.7).encode(
            x=alt.X("orcamento_plot:Q", title=titulo_x),
            y=alt.Y("bilheteria_plot:Q", title=titulo_y),
            color=alt.Color(
                f"{VAR_CLASSE}:N",
                title="Faixa de avaliação",
                scale=CORES_CLASSE,
            ),
            tooltip=[
                alt.Tooltip("budget_usd:Q", title="Orçamento (USD)", format=",.0f"),
                alt.Tooltip("box_office_usd:Q", title="Bilheteria (USD)", format=",.0f"),
                alt.Tooltip(f"{VAR_CLASSE}:N", title="Classe"),
            ],
        ).properties(height=430)
        st.altair_chart(dispersao, use_container_width=True)
        st.caption(
            "O gráfico permite observar a relação entre orçamento e bilheteria "
            "e comparar os filmes por faixa de avaliação. Ele não demonstra, "
            "por si só, que um orçamento maior causa uma bilheteria maior."
        )
else:
    st.info(
        "O gráfico orçamento x bilheteria precisa das colunas "
        "`budget_usd` e `box_office_usd`."
    )

# Tabela descritiva
with st.expander("Ver estatísticas descritivas"):
    colunas_tabela = [var for var in variaveis if var in df_filtrado.columns]
    if colunas_tabela and not df_filtrado.empty:
        descricao = df_filtrado[colunas_tabela].describe().T
        descricao["assimetria"] = df_filtrado[colunas_tabela].skew()
        descricao["curtose"] = df_filtrado[colunas_tabela].kurtosis()
        st.dataframe(descricao.round(2), use_container_width=True)
    else:
        st.info("Selecione variáveis e filtros que retornem dados para exibir a tabela.")

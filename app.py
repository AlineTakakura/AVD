import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import load_data

st.set_page_config(page_title="EDA — Filmes e Entretenimento", page_icon="🎬", layout="wide")

st.title("Análise Exploratória de Dados — Filmes e Entretenimento")
st.caption("Explore os dados dos filmes com filtros que se aplicam a todos os gráficos.")

VARS_QUANT = [
    "release_year", "release_day", "vote_count", "runtime_minutes",
    "box_office_usd", "budget_usd",
]
VAR_CLASSE = "rating_out_of_10"
COR_ABAIXO = "#6B8F91"       # verde-azulado suave para avaliações abaixo de 7
COR_SETE = "#D6A85F"          # dourado suave para avaliações 7 ou mais
COR_DESTAQUE = "#527C91"      # azul-petróleo para gráficos sem divisão por classe
CORES_CLASSE = alt.Scale(domain=["Abaixo de 7", "7 ou mais"], range=[COR_ABAIXO, COR_SETE])


@st.cache_data
def preparar_dados():
    dados = load_data().copy()
    for coluna in VARS_QUANT:
        if coluna in dados.columns:
            dados[coluna] = pd.to_numeric(dados[coluna], errors="coerce")
    if VAR_CLASSE not in dados.columns:
        raise ValueError(f"A coluna obrigatória '{VAR_CLASSE}' não foi encontrada.")

    # Padroniza as duas classes e preserva valores ausentes como ausentes.
    nota = dados[VAR_CLASSE]
    if pd.api.types.is_numeric_dtype(nota):
        dados[VAR_CLASSE] = np.where(nota.isna(), np.nan, np.where(nota >= 7, "7 ou mais", "Abaixo de 7"))
    else:
        nota = nota.astype("string").str.strip()
        nota_num = pd.to_numeric(nota, errors="coerce")
        classe_texto = nota.replace({"Abaixo de 7": "Abaixo de 7", "7 ou mais": "7 ou mais"})
        dados[VAR_CLASSE] = classe_texto.where(classe_texto.isin(["Abaixo de 7", "7 ou mais"]))
        mascara_num = nota_num.notna()
        dados.loc[mascara_num, VAR_CLASSE] = np.where(nota_num[mascara_num] >= 7, "7 ou mais", "Abaixo de 7")
    return dados


try:
    df = preparar_dados()
except Exception as erro:
    st.error("Não foi possível carregar os dados. Confira src/data_loader.py e o dataset.")
    st.exception(erro)
    st.stop()

vars_disponiveis = [var for var in VARS_QUANT if var in df.columns and df[var].notna().any()]
if not vars_disponiveis:
    st.error("Nenhuma variável quantitativa esperada foi encontrada no dataset.")
    st.stop()

# FILTROS: seleção de variáveis afeta os gráficos, enquanto os intervalos filtram os dados.
with st.sidebar:
    st.header("Filtros")
    st.caption("Os filtros de classe e intervalos são aplicados aos dois dashboards.")
    variaveis_graficos = st.multiselect(
        "Variáveis para visualizar",
        options=vars_disponiveis,
        default=vars_disponiveis,
        help="Escolha quais variáveis aparecem nos gráficos de distribuição e correlação. Não desativa os outros filtros.",
    )

    classes_disponiveis = [c for c in ["Abaixo de 7", "7 ou mais"] if c in df[VAR_CLASSE].dropna().unique()]
    classes_selecionadas = st.multiselect(
        "Faixa de avaliação",
        options=classes_disponiveis,
        default=classes_disponiveis,
        help="Desmarque uma classe para removê-la de todos os gráficos.",
    )

    st.subheader("Intervalos numéricos")
    st.caption("Cada intervalo começa e termina nos valores reais disponíveis na base.")
    intervalos = {}
    for var in vars_disponiveis:
        valores = df[var].dropna()
        minimo, maximo = float(valores.min()), float(valores.max())
        if minimo == maximo:
            intervalos[var] = (minimo, maximo)
            continue
        inteiros = np.all(np.isclose(valores.to_numpy(), np.round(valores.to_numpy())))
        passo = 1.0 if inteiros else max((maximo - minimo) / 100, 0.01)
        intervalos[var] = st.slider(
            f"{var}", min_value=minimo, max_value=maximo,
            value=(minimo, maximo), step=passo, key=f"filtro_{var}",
            help=f"Valores disponíveis: {minimo:g} a {maximo:g}.",
        )

    if st.button("Restaurar filtros", use_container_width=True):
        for var in vars_disponiveis:
            st.session_state.pop(f"filtro_{var}", None)
        st.session_state.pop("variaveis_graficos", None)
        st.session_state.pop("classes_avaliacao", None)
        st.rerun()

# Aplica os filtros independentemente das variáveis escolhidas para visualização.
df_filtrado = df[df[VAR_CLASSE].isin(classes_selecionadas)].copy()
for var, (inicio, fim) in intervalos.items():
    df_filtrado = df_filtrado[df_filtrado[var].between(inicio, fim)]

st.caption(f"**{len(df_filtrado):,}** de **{len(df):,}** filmes após os filtros.")
if df_filtrado.empty:
    st.warning("Nenhum filme corresponde aos filtros. Amplie um ou mais intervalos ou selecione outra faixa de avaliação.")
    st.stop()

tab1, tab2 = st.tabs(["Dashboard 1 — Exploração", "Dashboard 2 — Indicadores e comparações"])

with tab1:
    st.subheader("Indicadores gerais")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Filmes filtrados", f"{len(df_filtrado):,}")
    k2.metric("Avaliações 7 ou mais", f"{(df_filtrado[VAR_CLASSE] == '7 ou mais').sum():,}")
    k3.metric("Percentual com avaliação ≥ 7", f"{(df_filtrado[VAR_CLASSE].eq('7 ou mais').mean() * 100):.1f}%")
    if "runtime_minutes" in df_filtrado and df_filtrado["runtime_minutes"].notna().any():
        k4.metric("Duração média", f"{df_filtrado['runtime_minutes'].mean():.1f} min")
    else:
        k4.metric("Duração média", "—")

    st.divider()
    col_corr, col_dist = st.columns([1, 1.3])
    with col_corr:
        st.subheader("Correlação de Spearman")
        vars_corr = [v for v in variaveis_graficos if v in df_filtrado and df_filtrado[v].notna().sum() > 1]
        if len(vars_corr) < 2:
            st.info("Selecione pelo menos duas variáveis para calcular a correlação.")
        else:
            matriz = df_filtrado[vars_corr].corr(method="spearman").rename_axis(index="Variável Y", columns="Variável X")
            matriz_longa = matriz.stack().rename("Correlação").reset_index()
            mapa = alt.Chart(matriz_longa).encode(
                x=alt.X("Variável X:N", sort=vars_corr, title=None, axis=alt.Axis(labelAngle=-35)),
                y=alt.Y("Variável Y:N", sort=vars_corr, title=None),
            )
            heat = mapa.mark_rect().encode(
                color=alt.Color("Correlação:Q", scale=alt.Scale(domain=[-1, 1], range=["#527C91", "#F3F0E8", "#C47F67"]), title="Spearman"),
                tooltip=["Variável X:N", "Variável Y:N", alt.Tooltip("Correlação:Q", format=".2f")],
            )
            texto = mapa.mark_text(fontSize=11).encode(text=alt.Text("Correlação:Q", format=".2f"), color=alt.value("white"))
            st.altair_chart((heat + texto).properties(height=350), use_container_width=True)
            st.caption("A correlação indica associação, não causalidade.")

    with col_dist:
        st.subheader("Distribuição das variáveis")
        separar_classe = st.toggle("Separar por faixa de avaliação", value=True, key="dist_por_classe")
        if not variaveis_graficos:
            st.info("Escolha ao menos uma variável na barra lateral.")
        else:
            for inicio in range(0, len(variaveis_graficos), 2):
                cols = st.columns(2)
                for col, var in zip(cols, variaveis_graficos[inicio:inicio + 2]):
                    dados_grafico = df_filtrado[[var, VAR_CLASSE]].dropna(subset=[var])
                    grafico = alt.Chart(dados_grafico).mark_bar(opacity=0.85).encode(
                        x=alt.X(f"{var}:Q", bin=alt.Bin(maxbins=20), title=var),
                        y=alt.Y("count():Q", title="Quantidade de filmes"),
                        tooltip=[alt.Tooltip("count():Q", title="Filmes")],
                    )
                    if separar_classe:
                        grafico = grafico.encode(color=alt.Color(f"{VAR_CLASSE}:N", scale=CORES_CLASSE, title="Faixa de avaliação"))
                    else:
                        grafico = grafico.encode(color=alt.value(COR_DESTAQUE))
                    with col:
                        st.altair_chart(grafico.properties(height=220), use_container_width=True)

with tab2:
    st.subheader("Indicadores do conjunto filtrado")
    k1, k2, k3, k4 = st.columns(4)
    total = len(df_filtrado)
    qtd_sete = int(df_filtrado[VAR_CLASSE].eq("7 ou mais").sum())
    k1.metric("Total de filmes", f"{total:,}")
    k2.metric("Avaliação 7 ou mais", f"{qtd_sete:,}")
    k3.metric("Percentual com nota ≥ 7", f"{qtd_sete / total * 100:.1f}%" if total else "—")
    if "vote_count" in df_filtrado and df_filtrado["vote_count"].notna().any():
        k4.metric("Média de votos", f"{df_filtrado['vote_count'].mean():,.0f}")
    else:
        k4.metric("Média de votos", "—")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Quantidade por faixa de avaliação")
        contagem = df_filtrado[VAR_CLASSE].value_counts().reindex(["Abaixo de 7", "7 ou mais"], fill_value=0).rename_axis("Faixa").reset_index(name="Filmes")
        barras = alt.Chart(contagem).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
            x=alt.X("Faixa:N", sort=["Abaixo de 7", "7 ou mais"], title=None),
            y=alt.Y("Filmes:Q", title="Quantidade de filmes"),
            color=alt.Color("Faixa:N", scale=CORES_CLASSE, legend=None),
            tooltip=["Faixa:N", "Filmes:Q"],
        )
        st.altair_chart(barras.properties(height=300), use_container_width=True)
    with c2:
        st.subheader("Duração por faixa de avaliação")
        if "runtime_minutes" in df_filtrado and df_filtrado["runtime_minutes"].notna().any():
            dados_duracao = df_filtrado[[VAR_CLASSE, "runtime_minutes"]].dropna()
            caixas = alt.Chart(dados_duracao).mark_boxplot(size=55).encode(
                x=alt.X(f"{VAR_CLASSE}:N", sort=["Abaixo de 7", "7 ou mais"], title="Faixa de avaliação"),
                y=alt.Y("runtime_minutes:Q", title="Duração (minutos)"),
                color=alt.Color(f"{VAR_CLASSE}:N", scale=CORES_CLASSE, legend=None),
                tooltip=[alt.Tooltip("runtime_minutes:Q", title="Duração", format=".1f")],
            )
            st.altair_chart(caixas.properties(height=300), use_container_width=True)
        else:
            st.info("A variável runtime_minutes não possui dados válidos.")

    st.divider()
    st.subheader("Orçamento x bilheteria")
    if {"budget_usd", "box_office_usd", VAR_CLASSE}.issubset(df_filtrado.columns):
        relacao = df_filtrado[["budget_usd", "box_office_usd", VAR_CLASSE]].dropna().copy()
        if relacao.empty:
            st.warning("Não há pares válidos de orçamento e bilheteria após os filtros.")
        else:
            # Os dados parecem estar em log10 quando os valores ficam na faixa de 0 a 10.
            # Nesse caso, não aplicamos log10 novamente.
            ja_log = (
                relacao["budget_usd"].min() >= 0
                and relacao["box_office_usd"].min() >= 0
                and relacao["budget_usd"].max() <= 10
                and relacao["box_office_usd"].max() <= 10
            )
            usar_log = st.checkbox(
                "Exibir em escala log10",
                value=ja_log,
                help="Use esta opção para dados em dólares. Se os valores já estiverem em log10, eles serão exibidos sem uma segunda transformação.",
                key="escala_log_dashboard2",
            )

            if usar_log and not ja_log:
                relacao = relacao[(relacao["budget_usd"] > 0) & (relacao["box_office_usd"] > 0)].copy()
                relacao["orcamento_plot"] = np.log10(relacao["budget_usd"])
                relacao["bilheteria_plot"] = np.log10(relacao["box_office_usd"])
                titulo_x, titulo_y = "Orçamento (log10 USD)", "Bilheteria (log10 USD)"
                formato_tooltip = ",.0f"
            else:
                relacao["orcamento_plot"] = relacao["budget_usd"]
                relacao["bilheteria_plot"] = relacao["box_office_usd"]
                if ja_log:
                    titulo_x, titulo_y = "Orçamento (log10 USD)", "Bilheteria (log10 USD)"
                    formato_tooltip = ".3f"
                else:
                    titulo_x, titulo_y = "Orçamento (USD)", "Bilheteria (USD)"
                    formato_tooltip = ",.0f"

            if relacao.empty:
                st.warning("Não há valores positivos para exibir na escala logarítmica.")
            else:
                x_min, x_max = float(relacao["orcamento_plot"].min()), float(relacao["orcamento_plot"].max())
                y_min, y_max = float(relacao["bilheteria_plot"].min()), float(relacao["bilheteria_plot"].max())
                margem_x = max((x_max - x_min) * 0.08, 0.03 if ja_log else 1)
                margem_y = max((y_max - y_min) * 0.08, 0.03 if ja_log else 1)

                pontos = alt.Chart(relacao).mark_circle(size=105, opacity=0.8, stroke="#F3F0E8", strokeWidth=0.5).encode(
                    x=alt.X(
                        "orcamento_plot:Q", title=titulo_x,
                        scale=alt.Scale(domain=[x_min - margem_x, x_max + margem_x], zero=False),
                        axis=alt.Axis(format=".2f" if ja_log else "~s", grid=True),
                    ),
                    y=alt.Y(
                        "bilheteria_plot:Q", title=titulo_y,
                        scale=alt.Scale(domain=[y_min - margem_y, y_max + margem_y], zero=False),
                        axis=alt.Axis(format=".2f" if ja_log else "~s", grid=True),
                    ),
                    color=alt.Color(
                        f"{VAR_CLASSE}:N", scale=CORES_CLASSE,
                        title="Faixa de avaliação",
                        legend=alt.Legend(orient="right", titleFontSize=13, labelFontSize=12),
                    ),
                    tooltip=[
                        alt.Tooltip("budget_usd:Q", title="Orçamento (valor na base)", format=formato_tooltip),
                        alt.Tooltip("box_office_usd:Q", title="Bilheteria (valor na base)", format=formato_tooltip),
                        alt.Tooltip(f"{VAR_CLASSE}:N", title="Faixa de avaliação"),
                    ],
                )
                grafico = pontos.properties(height=440, title="Cada ponto representa um filme")
                st.altair_chart(grafico, use_container_width=True)
                if ja_log:
                    st.caption("Os dados parecem estar transformados em log10; os eixos mostram os valores logarítmicos existentes. O intervalo dos eixos foi ajustado aos pontos para facilitar a leitura.")
                else:
                    st.caption("O intervalo dos eixos foi ajustado aos dados para evitar espaço vazio. A associação observada não comprova causalidade.")
    else:
        st.info("O gráfico exige as colunas budget_usd e box_office_usd.")

    st.divider()
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Votos por faixa de avaliação")
        if "vote_count" in df_filtrado and df_filtrado["vote_count"].notna().any():
            votos = df_filtrado[[VAR_CLASSE, "vote_count"]].dropna()
            voto_chart = alt.Chart(votos).mark_boxplot(size=55).encode(
                x=alt.X(f"{VAR_CLASSE}:N", sort=["Abaixo de 7", "7 ou mais"], title="Faixa de avaliação"),
                y=alt.Y("vote_count:Q", title="Quantidade de votos"),
                color=alt.Color(f"{VAR_CLASSE}:N", scale=CORES_CLASSE, legend=None),
            )
            st.altair_chart(voto_chart.properties(height=300), use_container_width=True)
        else:
            st.info("A variável vote_count não possui dados válidos.")
    with c4:
        st.subheader("Ano de lançamento dos filmes")
        if "release_year" in df_filtrado and df_filtrado["release_year"].notna().any():
            anos = df_filtrado[["release_year", VAR_CLASSE]].dropna()
            ano_chart = alt.Chart(anos).mark_bar(opacity=0.85).encode(
                x=alt.X("release_year:Q", bin=alt.Bin(step=5), title="Ano de lançamento"),
                y=alt.Y("count():Q", title="Quantidade de filmes"),
                color=alt.Color(f"{VAR_CLASSE}:N", scale=CORES_CLASSE, title="Faixa de avaliação"),
                tooltip=[alt.Tooltip("count():Q", title="Filmes")],
            )
            st.altair_chart(ano_chart.properties(height=300), use_container_width=True)
        else:
            st.info("A variável release_year não possui dados válidos.")

    with st.expander("Ver estatísticas descritivas"):
        colunas = [v for v in variaveis_graficos if v in df_filtrado.columns]
        if colunas:
            descricao = df_filtrado[colunas].describe().T
            descricao["assimetria"] = df_filtrado[colunas].skew(numeric_only=True)
            descricao["curtose"] = df_filtrado[colunas].kurtosis(numeric_only=True)
            st.dataframe(descricao.round(2), use_container_width=True)
        else:
            st.info("Escolha variáveis na barra lateral para mostrar estatísticas.")

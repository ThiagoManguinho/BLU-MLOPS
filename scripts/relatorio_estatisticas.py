"""Relatório de estatísticas de monitoramento a partir das respostas registradas.

Lê `estatisticas/respostas_referencia.csv` e `estatisticas/respostas_producao.csv`
(gerados por `submeter_conjunto.py`) e escreve `estatisticas/relatorio.md`, com
os gráficos ao lado:

1. **Tempo de resposta** de `POST /prever_ocupacao`: estatísticas não
   paramétricas (mínimo, quartis, p90, p95, p99, máximo, IQR), intervalo de
   confiança do p95 por bootstrap, comparação com um SLA de exemplo
   (p95 < 400 ms), histograma, boxplot e gráfico de violino.
2. **Variável de entrada `nitidez`** (contínua): distância de Kolmogorov-Smirnov
   entre referência e produção.
3. **Variável de saída `ocupadas`** (inteira de 0 a 7, discreta): Population
   Stability Index entre referência e produção, uma faixa por valor.

Uso: uv run python scripts/relatorio_estatisticas.py [--sla-ms 400]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _estatisticas import (
    ALFA,
    CATEGORIAS_OCUPADAS,
    ic_bootstrap,
    ks,
    ks_critico,
    leitura_psi,
    percentis,
    psi_categorico,
)

_CORES_PERCENTIS = ((90, "#55A868"), (95, "#DD8452"), (99, "#C44E52"))


# --- Gráficos -------------------------------------------------------------------


def grafico_latencia(dados: dict[str, np.ndarray], pasta: Path) -> Path:
    todos = np.concatenate(list(dados.values()))
    figura, eixo_h = plt.subplots(figsize=(12, 4.2))
    eixo_h.hist(todos, bins=60, color="#4C72B0", alpha=0.8)
    for q, cor in _CORES_PERCENTIS:
        valor_q = np.percentile(todos, q)
        eixo_h.axvline(valor_q, color=cor, linestyle="--", label=f"p{q} = {valor_q:.0f} ms")
    eixo_h.set_title(f"Histograma do tempo de resposta, todos os pedidos (n={todos.size})")
    eixo_h.set_xlabel("ms")
    eixo_h.set_ylabel("pedidos")
    eixo_h.legend()
    figura.tight_layout()
    caminho = pasta / "latencia.png"
    figura.savefig(caminho, dpi=110)
    plt.close(figura)
    return caminho


def grafico_boxplot_violino(dados: dict[str, np.ndarray], pasta: Path) -> Path:
    """Boxplot e gráfico de violino do tempo de resposta, lado a lado.

    O boxplot resume em 5 números (bigodes, quartis e mediana) e marca os
    pontos atípicos. O violino desenha a densidade estimada (KDE): onde ele é
    largo há muitos pedidos com aquele tempo. Ele mostra o que o boxplot
    esconde, como mais de um "pico" de tempos.
    """
    nomes = list(dados)
    valores = [dados[n] for n in nomes]
    posicoes = np.arange(1, len(nomes) + 1)
    figura, (eixo_b, eixo_v) = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)

    eixo_b.boxplot(valores, positions=posicoes, tick_labels=nomes, whis=1.5, widths=0.5,
                   showmeans=True, meanprops={"marker": "D", "markerfacecolor": "white",
                                              "markeredgecolor": "black", "markersize": 5})
    eixo_b.set_title("Boxplot (bigodes = 1,5 IQR; losango = média)")
    eixo_b.set_ylabel("tempo de resposta (ms)")

    partes = eixo_v.violinplot(valores, positions=posicoes, widths=0.8, showextrema=False,
                               bw_method=0.15)
    for corpo in partes["bodies"]:
        corpo.set_facecolor("#4C72B0")
        corpo.set_alpha(0.45)
    for x, v in zip(posicoes, valores, strict=True):
        eixo_v.scatter(np.full(v.size, x) + np.random.default_rng(0).uniform(-0.12, 0.12, v.size),
                       v, s=4, color="black", alpha=0.35, zorder=3)
        for q, cor in _CORES_PERCENTIS:
            eixo_v.hlines(np.percentile(v, q), x - 0.3, x + 0.3, color=cor, linewidth=1.6)
        eixo_v.hlines(np.median(v), x - 0.3, x + 0.3, color="black", linewidth=1.6)
    eixo_v.set_xticks(posicoes, nomes)
    eixo_v.set_title("Violino (preto = mediana; verde/laranja/vermelho = p90/p95/p99)")
    figura.tight_layout()
    caminho = pasta / "latencia_boxplot_violino.png"
    figura.savefig(caminho, dpi=110)
    plt.close(figura)
    return caminho


def modas(valores: np.ndarray, largura_ms: float = 5.0) -> list[float]:
    """Picos da densidade (KDE), do maior para o menor, que tenham pelo menos
    15% da altura do maior. É o que o violino mostra e o boxplot não."""
    from scipy.stats import gaussian_kde

    valores = np.asarray(valores, dtype=float)
    grade = np.arange(valores.min(), valores.max() + largura_ms, 0.5)
    densidade = gaussian_kde(valores, bw_method=0.15)(grade)
    picos = [
        i for i in range(1, grade.size - 1)
        if densidade[i] >= densidade[i - 1] and densidade[i] > densidade[i + 1]
        and densidade[i] >= 0.15 * densidade.max()
    ]
    return [float(grade[i]) for i in sorted(picos, key=lambda i: -densidade[i])]


def grafico_nitidez(referencia: np.ndarray, producao: np.ndarray, pasta: Path) -> Path:
    figura, (eixo_c, eixo_b) = plt.subplots(1, 2, figsize=(12, 4.2))
    for valores, nome in ((referencia, "referência"), (producao, "produção")):
        ordenados = np.sort(valores)
        eixo_c.step(ordenados, np.arange(1, ordenados.size + 1) / ordenados.size, where="post",
                    label=f"{nome} (n={ordenados.size})")
    eixo_c.set_title("CDF empírica da nitidez (o KS é a maior distância vertical)")
    eixo_c.set_xlabel("nitidez (variância do Laplaciano)")
    eixo_c.legend()
    eixo_b.boxplot([referencia, producao], tick_labels=["referência", "produção"])
    eixo_b.set_title("Boxplot da nitidez")
    figura.tight_layout()
    caminho = pasta / "nitidez.png"
    figura.savefig(caminho, dpi=110)
    plt.close(figura)
    return caminho


def grafico_ocupadas(tabela: pd.DataFrame, pasta: Path) -> Path:
    figura, eixo = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(tabela))
    eixo.bar(x - 0.2, tabela["p_ref"], width=0.4, label="referência")
    eixo.bar(x + 0.2, tabela["p_prod"], width=0.4, label="produção")
    eixo.set_xticks(x, tabela["valor"])
    eixo.set_xlabel("vagas ocupadas na imagem")
    eixo.set_ylabel("proporção das imagens")
    eixo.set_title("Distribuição de vagas ocupadas por imagem")
    eixo.legend()
    figura.tight_layout()
    caminho = pasta / "ocupadas.png"
    figura.savefig(caminho, dpi=110)
    plt.close(figura)
    return caminho


# --- Relatório ------------------------------------------------------------------


def _ms(valor: float) -> str:
    return f"{valor:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _num(valor: float, casas: int = 3) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


def ler_respostas(caminho: Path) -> pd.DataFrame:
    tabela = pd.read_csv(caminho)
    return tabela[tabela["status_http"] == 200].copy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pasta", type=Path, default=Path("estatisticas"))
    parser.add_argument("--sla-ms", type=float, default=400.0)
    args = parser.parse_args()

    caminhos = {
        "referência": args.pasta / "respostas_referencia.csv",
        "produção": args.pasta / "respostas_producao.csv",
    }
    faltando = [str(c) for c in caminhos.values() if not c.exists()]
    if faltando:
        print("faltam as respostas: " + ", ".join(faltando), file=sys.stderr)
        print("rode antes scripts/submeter_conjunto.py para os dois conjuntos", file=sys.stderr)
        return 1
    respostas = {nome: ler_respostas(c) for nome, c in caminhos.items()}
    ref, prod = respostas["referência"], respostas["produção"]

    # 1. Tempo de resposta.
    latencias = {nome: t["tempo_resposta_ms"].to_numpy() for nome, t in respostas.items()}
    latencias_todas = {**latencias, "todos": np.concatenate(list(latencias.values()))}
    linhas_lat = []
    for nome, valores in latencias_todas.items():
        resumo = percentis(valores)
        baixo, alto = ic_bootstrap(valores, 95)
        acima = float((valores > args.sla_ms).mean())
        linhas_lat.append((nome, resumo, baixo, alto, acima))
    grafico_latencia(latencias, args.pasta)
    grafico_boxplot_violino(latencias_todas, args.pasta)

    # 2. KS da nitidez (entrada).
    nit_ref = ref["nitidez"].dropna().to_numpy(dtype=float)
    nit_prod = prod["nitidez"].dropna().to_numpy(dtype=float)
    d_ks, p_ks = ks(nit_ref, nit_prod)
    grafico_nitidez(nit_ref, nit_prod, args.pasta)

    # 3. PSI de ocupadas (saída).
    valor_psi, por_categoria = psi_categorico(
        ref["ocupadas"].to_numpy(dtype=int), prod["ocupadas"].to_numpy(dtype=int),
        CATEGORIAS_OCUPADAS,
    )
    tabela_psi = pd.DataFrame(por_categoria)
    grafico_ocupadas(tabela_psi, args.pasta)

    # Relatório em Markdown.
    s = []
    s.append("# Relatório de estatísticas de monitoramento\n")
    s.append(f"Gerado em {datetime.now().astimezone().isoformat(timespec='seconds')} por "
             "`scripts/relatorio_estatisticas.py`, a partir de "
             "`respostas_referencia.csv` e `respostas_producao.csv`. "
             "Só entram pedidos com status 200.\n")

    s.append("## 1. Tempo de resposta de `POST /prever_ocupacao`\n")
    s.append("Medido no cliente, do envio até a resposta completa, um pedido por vez "
             "(sem concorrência), depois de 3 pedidos de aquecimento descartados.\n")
    s.append("| Conjunto | n | mín | p25 | p50 | p75 | **p90** | **p95** | **p99** | máx | IQR "
             f"| IC 95% do p95 | > {args.sla_ms:.0f} ms |")
    s.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|")
    for nome, r, baixo, alto, acima in linhas_lat:
        s.append(
            f"| {nome} | {r['n']} | {_ms(r['min'])} | {_ms(r['p25'])} | {_ms(r['p50'])} | "
            f"{_ms(r['p75'])} | **{_ms(r['p90'])}** | **{_ms(r['p95'])}** | **{_ms(r['p99'])}** | "
            f"{_ms(r['max'])} | {_ms(r['iqr'])} | {_ms(baixo)} a {_ms(alto)} | "
            f"{_num(acima * 100, 1)}% |"
        )
    s.append("\nValores em milissegundos.\n")
    _, todos, baixo_t, alto_t, acima_t = linhas_lat[-1]
    atende = todos["p95"] < args.sla_ms
    s.append(f"**SLA de exemplo (p95 < {args.sla_ms:.0f} ms):** "
             f"{'atendido' if atende else 'NÃO atendido'}. O p95 de todos os pedidos é "
             f"{_ms(todos['p95'])} ms (IC 95% de {_ms(baixo_t)} a {_ms(alto_t)} ms), e "
             f"{_num(acima_t * 100, 1)}% dos pedidos passaram de {args.sla_ms:.0f} ms.\n")
    n_total = todos["n"]
    posicao = 0.99 * (n_total - 1) + 1
    s.append(f"**Como ler o p99 com n = {n_total}:** o p99 fica na posição {_num(posicao, 2)} "
             f"dos tempos ordenados, ou seja, entre o {n_total - int(posicao) + 1}º e o "
             f"{n_total - int(posicao)}º maior tempo. Ele depende de poucos pedidos e oscila "
             "muito entre execuções; o p95 é mais estável, e por isso é o percentil usado em "
             "SLAs.\n")
    s.append("![Tempo de resposta](latencia.png)\n")

    s.append("### Boxplot e gráfico de violino\n")
    s.append("![Boxplot e violino do tempo de resposta](latencia_boxplot_violino.png)\n")
    s.append("| Conjunto | bigode inferior | Q1 | mediana | média | Q3 | bigode superior "
             "| atípicos | picos da densidade (violino) |")
    s.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for nome, valores in latencias_todas.items():
        r = percentis(valores)
        picos = ", ".join(f"{_ms(m)}" for m in modas(valores))
        s.append(
            f"| {nome} | {_ms(r['bigode_inferior'])} | {_ms(r['p25'])} | {_ms(r['p50'])} | "
            f"{_ms(float(np.mean(valores)))} | {_ms(r['p75'])} | {_ms(r['bigode_superior'])} | "
            f"{r['atipicos']} ({_num(r['atipicos'] / r['n'] * 100, 1)}%) | {picos} |"
        )
    s.append("\nValores em milissegundos. Os picos estão em ordem de altura, do mais alto "
             "para o mais baixo.\n")
    s.append("- **Boxplot:** a caixa vai do Q1 ao Q3 (metade central dos pedidos), a linha é a "
             "mediana, e os bigodes vão até o pedido mais extremo dentro de 1,5 IQR além da "
             "caixa. Os pontos acima do bigode são atípicos.")
    s.append("- **Violino:** a largura é a densidade estimada (KDE) dos tempos. Onde ele é largo, "
             "muitos pedidos levaram aquele tempo. Mais de um pico indica grupos de pedidos "
             "com tempos diferentes, algo que o boxplot não mostra.\n")

    s.append("## 2. Conjuntos submetidos\n")
    s.append("| Conjunto | Origem | Imagens válidas (status 200) |")
    s.append("|---|---|---:|")
    s.append(f"| referência | 245 fotos de treino do modelo (`train-maquete/images`) | {len(ref)} |")
    s.append(f"| produção | variações realistas das 3 fotos de `samples/` "
             f"(`scripts/gerar_producao.py`) | {len(prod)} |\n")

    s.append("## 3. Variável de entrada: `nitidez` (contínua → KS)\n")
    s.append("A nitidez (variância do Laplaciano) é um número real não negativo, então é "
             "contínua e o critério é a distância de Kolmogorov-Smirnov.\n")
    s.append("| Conjunto | n | p25 | mediana | p75 |")
    s.append("|---|---:|---:|---:|---:|")
    for nome, valores in (("referência", nit_ref), ("produção", nit_prod)):
        q1, q2, q3 = np.percentile(valores, [25, 50, 75])
        s.append(f"| {nome} | {valores.size} | {_num(q1, 1)} | {_num(q2, 1)} | {_num(q3, 1)} |")
    leitura_ks = ("as distribuições são diferentes (drift)" if p_ks < ALFA
                  else "não há evidência de diferença")
    d_critico = ks_critico(nit_ref.size, nit_prod.size)
    s.append(f"\n**KS: D = {_num(d_ks)}**, p-valor = {p_ks:.3g}. Com α = {_num(ALFA, 2)}, "
             f"{leitura_ks} (o D crítico para estes tamanhos de amostra é "
             f"{_num(d_critico)}). D é a maior distância vertical entre as duas CDFs: "
             f"em algum valor de nitidez, as proporções de imagens abaixo dele diferem em "
             f"{_num(d_ks * 100, 1)} pontos percentuais.\n")
    s.append("![Nitidez](nitidez.png)\n")

    s.append("## 4. Variável de saída: `ocupadas` (discreta → PSI)\n")
    s.append("`ocupadas` é um inteiro de 0 a 7 (a maquete tem 7 vagas), bem abaixo do "
             "limite de 100 valores do enunciado, então é discreta e o critério é o PSI, "
             "com uma faixa por valor.\n")
    s.append("| ocupadas | % referência | % produção | contribuição ao PSI |")
    s.append("|---:|---:|---:|---:|")
    for _, linha in tabela_psi.iterrows():
        s.append(f"| {int(linha['valor'])} | {_num(linha['p_ref'] * 100, 1)}% | "
                 f"{_num(linha['p_prod'] * 100, 1)}% | {_num(linha['contribuicao'], 4)} |")
    s.append(f"\n**PSI = {_num(valor_psi)}** → {leitura_psi(valor_psi)} "
             "(< 0,1 estável; 0,1 a 0,25 mudança moderada; > 0,25 drift).\n")
    s.append("![Ocupadas](ocupadas.png)\n")

    destino = args.pasta / "relatorio.md"
    destino.write_text("\n".join(s), encoding="utf-8")

    print(f"Latência (todos, n={todos['n']}): p90 {_ms(todos['p90'])} ms, "
          f"p95 {_ms(todos['p95'])} ms, p99 {_ms(todos['p99'])} ms")
    print(f"KS nitidez: D = {d_ks:.3f} (p = {p_ks:.3g})")
    print(f"PSI ocupadas: {valor_psi:.3f} ({leitura_psi(valor_psi)})")
    print(f"Relatório em {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

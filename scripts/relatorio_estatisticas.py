"""Relatório de estatísticas de monitoramento a partir das respostas registradas.

Lê `estatisticas/respostas_referencia.csv` e `estatisticas/respostas_producao.csv`
(gerados por `submeter_conjunto.py`) e escreve `estatisticas/relatorio.md`, com
os gráficos ao lado:

1. **Tempo de resposta** de `POST /prever_ocupacao`: estatísticas não
   paramétricas (mínimo, quartis, p90, p95, p99, máximo, IQR), intervalo de
   confiança do p95 por bootstrap e comparação com um SLA de exemplo
   (p95 < 400 ms).
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
from analisar_log import leitura_psi
from scipy.stats import ks_2samp

PERCENTIS = (25, 50, 75, 90, 95, 99)
CATEGORIAS_OCUPADAS = list(range(8))  # 0 a 7 vagas ocupadas
_EPS = 1e-6
_ALFA = 0.05


# --- Funções de cálculo (testadas em tests/test_estatisticas.py) ---------------


def percentis(valores: np.ndarray) -> dict:
    """Resumo não paramétrico: não supõe nenhuma distribuição.

    Usa a interpolação linear do numpy (padrão), a mesma do Excel
    (PERCENTIL.INC) e do pandas.
    """
    valores = np.asarray(valores, dtype=float)
    resumo = {"n": int(valores.size), "min": float(valores.min()), "max": float(valores.max())}
    for q in PERCENTIS:
        resumo[f"p{q}"] = float(np.percentile(valores, q))
    resumo["iqr"] = resumo["p75"] - resumo["p25"]
    return resumo


def ic_bootstrap(
    valores: np.ndarray, q: float = 95, reamostragens: int = 2000, semente: int = 0
) -> tuple[float, float]:
    """Intervalo de 95% do percentil q por bootstrap (percentis da reamostragem)."""
    valores = np.asarray(valores, dtype=float)
    gerador = np.random.default_rng(semente)
    amostras = gerador.choice(valores, size=(reamostragens, valores.size), replace=True)
    estimativas = np.percentile(amostras, q, axis=1)
    return float(np.percentile(estimativas, 2.5)), float(np.percentile(estimativas, 97.5))


def ks(referencia: np.ndarray, producao: np.ndarray) -> tuple[float, float]:
    """Distância D de Kolmogorov-Smirnov (maior diferença entre as CDFs) e p-valor."""
    resultado = ks_2samp(referencia, producao)
    return float(resultado.statistic), float(resultado.pvalue)


def psi_categorico(
    referencia: np.ndarray, producao: np.ndarray, categorias: list[int]
) -> tuple[float, pd.DataFrame]:
    """PSI com uma faixa por categoria.

    PSI = soma (p_prod - p_ref) * ln(p_prod / p_ref). Categorias sem nenhum
    caso recebem 1e-6 para o logaritmo existir.
    """
    referencia = np.asarray(referencia)
    producao = np.asarray(producao)
    p_ref = np.array([(referencia == c).mean() for c in categorias])
    p_prod = np.array([(producao == c).mean() for c in categorias])
    a, b = np.maximum(p_ref, _EPS), np.maximum(p_prod, _EPS)
    contribuicao = (b - a) * np.log(b / a)
    tabela = pd.DataFrame(
        {"valor": categorias, "p_ref": p_ref, "p_prod": p_prod, "contribuicao": contribuicao}
    )
    return float(contribuicao.sum()), tabela


# --- Gráficos -------------------------------------------------------------------


def grafico_latencia(dados: dict[str, np.ndarray], pasta: Path) -> Path:
    todos = np.concatenate(list(dados.values()))
    figura, (eixo_h, eixo_b) = plt.subplots(1, 2, figsize=(12, 4.2))
    eixo_h.hist(todos, bins=40, color="#4C72B0", alpha=0.8)
    for q, cor in ((90, "#55A868"), (95, "#DD8452"), (99, "#C44E52")):
        valor_q = np.percentile(todos, q)
        eixo_h.axvline(valor_q, color=cor, linestyle="--", label=f"p{q} = {valor_q:.0f} ms")
    eixo_h.set_title(f"Tempo de resposta, todos os pedidos (n={todos.size})")
    eixo_h.set_xlabel("ms")
    eixo_h.legend()
    eixo_b.boxplot(list(dados.values()), tick_labels=list(dados), whis=1.5)
    eixo_b.set_title("Boxplot por conjunto")
    eixo_b.set_ylabel("ms")
    figura.tight_layout()
    caminho = pasta / "latencia.png"
    figura.savefig(caminho, dpi=110)
    plt.close(figura)
    return caminho


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

    # 2. KS da nitidez (entrada).
    nit_ref = ref["nitidez"].dropna().to_numpy(dtype=float)
    nit_prod = prod["nitidez"].dropna().to_numpy(dtype=float)
    d_ks, p_ks = ks(nit_ref, nit_prod)
    grafico_nitidez(nit_ref, nit_prod, args.pasta)

    # 3. PSI de ocupadas (saída).
    valor_psi, tabela_psi = psi_categorico(
        ref["ocupadas"].to_numpy(dtype=int), prod["ocupadas"].to_numpy(dtype=int),
        CATEGORIAS_OCUPADAS,
    )
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
    leitura_ks = ("as distribuições são diferentes (drift)" if p_ks < _ALFA
                  else "não há evidência de diferença")
    s.append(f"\n**KS: D = {_num(d_ks)}**, p-valor = {p_ks:.3g}. Com α = {_num(_ALFA, 2)}, "
             f"{leitura_ks}. D é a maior distância vertical entre as duas CDFs: "
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

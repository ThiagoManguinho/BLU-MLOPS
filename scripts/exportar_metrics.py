"""Tira uma foto do GET /metrics e monta a planilha de análise (docs/).

A planilha mostra, com fórmulas do Excel visíveis, como sair do texto do
/metrics (formato Prometheus) para:
- o histograma da latência de cada endpoint (contagem por faixa = diferença
  entre faixas acumuladas vizinhas);
- os quartis e o boxplot aproximados, interpolando dentro da faixa, como faz
  o histogram_quantile do Prometheus;
- a taxa de erros por endpoint (respostas 4xx e 5xx ÷ total).

Abas: leia_me, metrics_bruto (as linhas do /metrics usadas), uma aba lat_*
por endpoint e taxa_erros. Os mesmos números são calculados aqui em Python e
impressos no terminal, para conferir as fórmulas.

Uso: uv run python scripts/exportar_metrics.py [--url URL] [--saida ARQUIVO]
"""

from __future__ import annotations

import argparse
import io
import math
import sys
from datetime import datetime
from pathlib import Path

import httpx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.drawing.image import Image as ImagemXlsx
from openpyxl.styles import Font, PatternFill
from prometheus_client.parser import text_string_to_metric_families

LATENCIA = "bentoml_service_request_duration_seconds"
PEDIDOS = "bentoml_service_request_total"
QUANTIS = (("Q1", 0.25), ("Mediana", 0.5), ("Q3", 0.75), ("p95", 0.95))
# Rotas de infraestrutura que o BentoML também mede, mas não são inferência.
ROTAS_INFRA = ("/metrics", "/healthz", "/livez", "/readyz")
_NEGRITO = Font(bold=True)
_CABECALHO = PatternFill("solid", fgColor="DDEBF7")


def ler_metrics(url: str) -> tuple[str, list[dict]]:
    texto = httpx.get(f"{url}/metrics", timeout=30).text
    linhas = []
    for familia in text_string_to_metric_families(texto):
        if familia.name not in (LATENCIA, PEDIDOS.removesuffix("_total")):
            continue
        for amostra in familia.samples:
            if amostra.labels.get("endpoint", "") in ROTAS_INFRA:
                continue
            linhas.append(
                {
                    "metrica": amostra.name,
                    "endpoint": amostra.labels.get("endpoint", ""),
                    "http_response_code": amostra.labels.get("http_response_code", ""),
                    "le": amostra.labels.get("le", ""),
                    "valor": amostra.value,
                }
            )
    return texto, linhas


def faixas_por_endpoint(linhas: list[dict]) -> dict[str, list[tuple[str, float]]]:
    """Faixas acumuladas da latência das respostas 200 de cada endpoint. Os
    erros (4xx/5xx) respondem em milissegundos e distorceriam o tempo de
    inferência; eles entram na aba taxa_erros."""
    resultado: dict[str, dict[str, float]] = {}
    for linha in linhas:
        if linha["metrica"] == f"{LATENCIA}_bucket" and linha["http_response_code"] == "200":
            por_le = resultado.setdefault(linha["endpoint"], {})
            por_le[linha["le"]] = por_le.get(linha["le"], 0.0) + linha["valor"]
    return {
        endpoint: sorted(por_le.items(), key=lambda item: float(item[0]))
        for endpoint, por_le in resultado.items()
        if por_le.get("+Inf", 0) > 0
    }


def quantil(faixas: list[tuple[str, float]], q: float) -> float:
    """Mesma interpolação linear do histogram_quantile do Prometheus. Se o
    quantil cair na faixa +Inf, devolve o maior limite finito."""
    total = faixas[-1][1]
    posicao = q * total
    anterior_le, anterior_acum = 0.0, 0.0
    for le, acumulado in faixas:
        superior = float(le)
        if acumulado >= posicao:
            if math.isinf(superior):
                return anterior_le
            na_faixa = acumulado - anterior_acum
            return anterior_le + (superior - anterior_le) * (posicao - anterior_acum) / na_faixa
        anterior_le, anterior_acum = superior, acumulado
    return anterior_le


def _cabecalho(aba, linha: int, titulos: list[str]) -> None:
    for coluna, titulo in enumerate(titulos, start=1):
        celula = aba.cell(row=linha, column=coluna, value=titulo)
        celula.font = _NEGRITO
        celula.fill = _CABECALHO


def _boxplot_png(endpoint: str, estatisticas: dict) -> io.BytesIO:
    figura, eixo = plt.subplots(figsize=(6, 2.4))
    eixo.bxp([estatisticas], orientation="horizontal", showfliers=False)
    eixo.set_yticks([1], [endpoint])
    eixo.set_xlabel("latência (s), aproximada pelas faixas do /metrics")
    figura.tight_layout()
    buffer = io.BytesIO()
    figura.savefig(buffer, format="png", dpi=100)
    plt.close(figura)
    buffer.seek(0)
    return buffer


def aba_latencia(
    livro: Workbook, endpoint: str, faixas: list[tuple[str, float]], soma: float
) -> dict:
    nome = ("lat_" + endpoint.strip("/"))[:31]
    aba = livro.create_sheet(nome)
    aba["A1"] = f"Latência de {endpoint} (respostas 200)"
    aba["A1"].font = Font(bold=True, size=13)
    aba["A2"] = (
        "Coluna D vem do /metrics (aba metrics_bruto). As demais são fórmulas: "
        "E = diferença entre faixas acumuladas vizinhas; H:K = interpolação do quantil."
    )
    _cabecalho(
        aba,
        4,
        [
            "le (como no /metrics)",
            "limite inferior (s)",
            "limite superior (s)",
            "acumulado (pedidos <= le)",
            "pedidos na faixa",
            "frequência relativa",
            "acumulado relativo",
            *[f"{rotulo} se cair nesta faixa" for rotulo, _ in QUANTIS],
        ],
    )
    primeira = 5
    ultima = primeira + len(faixas) - 1
    total = f"$D${ultima}"
    # Posição de cada quantil (q × total), usada pelas colunas H:K.
    linha_resumo = ultima + 3
    posicoes = {}
    for i, (rotulo, q) in enumerate(QUANTIS):
        posicoes[rotulo] = f"$C${linha_resumo + 1 + i}"

    for i, (le, _acumulado) in enumerate(faixas):
        r = primeira + i
        aba.cell(row=r, column=1, value=le)
        aba.cell(row=r, column=2, value=0 if i == 0 else f"=C{r - 1}")
        # +Inf não tem limite superior: usa o limite inferior (o maior finito),
        # como o histogram_quantile.
        aba.cell(row=r, column=3, value=f"=B{r}" if le == "+Inf" else float(le))
        aba.cell(
            row=r,
            column=4,
            value=(
                f'=SUMIFS(metrics_bruto!$E:$E,metrics_bruto!$A:$A,"{LATENCIA}_bucket",'
                f'metrics_bruto!$B:$B,"{endpoint}",metrics_bruto!$C:$C,"200",'
                f'metrics_bruto!$F:$F,"le="&A{r})'
            ),
        )
        anterior = "0" if i == 0 else f"D{r - 1}"
        aba.cell(row=r, column=5, value=f"=D{r}-{anterior}")
        aba.cell(row=r, column=6, value=f"=IF({total}=0,0,E{r}/{total})")
        aba.cell(row=r, column=7, value=f"=IF({total}=0,0,D{r}/{total})")
        for j, (rotulo, _q) in enumerate(QUANTIS):
            pos = posicoes[rotulo]
            aba.cell(
                row=r,
                column=8 + j,
                value=(
                    f'=IF(AND({anterior}<{pos},D{r}>={pos}),'
                    f'IF(A{r}="+Inf",B{r},B{r}+(C{r}-B{r})*({pos}-{anterior})/E{r}),"")'
                ),
            )
    for coluna in "FG":
        for r in range(primeira, ultima + 1):
            aba[f"{coluna}{r}"].number_format = "0.0%"

    # Resumo: total, média, quartis e boxplot.
    valores = {rotulo: quantil(faixas, q) for rotulo, q in QUANTIS}
    _cabecalho(aba, linha_resumo - 1, ["estatística", "como calcular", "posição / valor (s)", "Python"])
    aba.cell(row=linha_resumo, column=1, value="total de pedidos")
    aba.cell(row=linha_resumo, column=2, value="acumulado da faixa +Inf (= _count)")
    aba.cell(row=linha_resumo, column=3, value=f"={total}")
    aba.cell(row=linha_resumo, column=4, value=faixas[-1][1])
    letras = "HIJK"
    for i, (rotulo, q) in enumerate(QUANTIS):
        r = linha_resumo + 1 + i
        aba.cell(row=r, column=1, value=f"posição do {rotulo}")
        aba.cell(row=r, column=2, value=f"{q} × total")
        aba.cell(row=r, column=3, value=f"={q}*{total}")
    base = linha_resumo + 1 + len(QUANTIS)
    for i, (rotulo, _q) in enumerate(QUANTIS):
        r = base + i
        aba.cell(row=r, column=1, value=f"{rotulo} (s)")
        aba.cell(row=r, column=2, value=f"soma da coluna {letras[i]} (só uma faixa preenchida)")
        aba.cell(row=r, column=3, value=f"=SUM({letras[i]}{primeira}:{letras[i]}{ultima})")
        aba.cell(row=r, column=4, value=round(valores[rotulo], 6))
    # Bigodes de Tukey, limitados às faixas que de fato têm pedidos (o mínimo
    # e o máximo reais não aparecem no /metrics).
    limites = []
    anterior_le, anterior_acum = 0.0, 0.0
    for le, acumulado in faixas:
        superior = anterior_le if le == "+Inf" else float(le)
        if acumulado > anterior_acum:
            limites.append((anterior_le, superior))
        anterior_le, anterior_acum = superior, acumulado
    iqr = valores["Q3"] - valores["Q1"]
    estatisticas = {
        "label": endpoint,
        "q1": valores["Q1"],
        "med": valores["Mediana"],
        "q3": valores["Q3"],
        "whislo": max(valores["Q1"] - 1.5 * iqr, min(a for a, _ in limites)),
        "whishi": min(valores["Q3"] + 1.5 * iqr, max(b for _, b in limites)),
    }

    q1, q3 = f"C{base}", f"C{base + 2}"
    # MINIFS/MAXIFS são funções "novas" do Excel: no arquivo precisam do prefixo
    # _xlfn., senão o Excel mostra #NOME?.
    extras = [
        ("média (s)", "_sum ÷ _count", (
            f'=SUMIFS(metrics_bruto!$E:$E,metrics_bruto!$A:$A,"{LATENCIA}_sum",'
            f'metrics_bruto!$B:$B,"{endpoint}",metrics_bruto!$C:$C,"200")'
            f'/SUMIFS(metrics_bruto!$E:$E,metrics_bruto!$A:$A,"{LATENCIA}_count",'
            f'metrics_bruto!$B:$B,"{endpoint}",metrics_bruto!$C:$C,"200")'
        ), soma / faixas[-1][1]),
        ("IQR (s)", "Q3 − Q1", f"={q3}-{q1}", iqr),
        ("bigode inferior (s)", "max(Q1 − 1,5·IQR, menor limite com pedidos)",
         f'=MAX({q1}-1.5*({q3}-{q1}),_xlfn.MINIFS(B{primeira}:B{ultima},E{primeira}:E{ultima},">0"))',
         estatisticas["whislo"]),
        ("bigode superior (s)", "min(Q3 + 1,5·IQR, maior limite com pedidos)",
         f'=MIN({q3}+1.5*({q3}-{q1}),_xlfn.MAXIFS(C{primeira}:C{ultima},E{primeira}:E{ultima},">0"))',
         estatisticas["whishi"]),
    ]
    for i, (rotulo, como, formula, valor_python) in enumerate(extras):
        r = base + len(QUANTIS) + i
        aba.cell(row=r, column=1, value=rotulo)
        aba.cell(row=r, column=2, value=como)
        aba.cell(row=r, column=3, value=formula)
        aba.cell(row=r, column=4, value=round(valor_python, 6))
    for r in range(linha_resumo - 1, base + len(QUANTIS) + len(extras)):
        aba.cell(row=r, column=1).font = _NEGRITO

    # Histograma: pedidos por faixa.
    grafico = BarChart()
    grafico.title = f"Histograma da latência: {endpoint}"
    grafico.x_axis.title = "faixa (le, em segundos)"
    grafico.y_axis.title = "pedidos"
    grafico.add_data(Reference(aba, min_col=5, min_row=4, max_row=ultima), titles_from_data=True)
    grafico.set_categories(Reference(aba, min_col=1, min_row=primeira, max_row=ultima))
    grafico.width, grafico.height = 22, 9
    aba.add_chart(grafico, "M4")

    aba.add_image(ImagemXlsx(_boxplot_png(endpoint, estatisticas)), "M24")

    aba.column_dimensions["A"].width = 22
    aba.column_dimensions["B"].width = 42
    for coluna in "CDEFGHIJK":
        aba.column_dimensions[coluna].width = 17
    return {"total": faixas[-1][1], **valores, **{k: estatisticas[k] for k in ("whislo", "whishi")}}


def aba_erros(livro: Workbook, linhas: list[dict]) -> list[tuple]:
    aba = livro.create_sheet("taxa_erros")
    aba["A1"] = "Taxa de erros por endpoint (desde que o serviço subiu)"
    aba["A1"].font = Font(bold=True, size=13)
    _cabecalho(aba, 3, ["endpoint", "total", "2xx", "4xx", "5xx", "taxa de erros (4xx+5xx)", "taxa 5xx"])
    endpoints = sorted({linha["endpoint"] for linha in linhas if linha["metrica"] == PEDIDOS})
    resultado = []
    for i, endpoint in enumerate(endpoints):
        r = 4 + i
        aba.cell(row=r, column=1, value=endpoint)
        soma = (
            f'SUMIFS(metrics_bruto!$E:$E,metrics_bruto!$A:$A,"{PEDIDOS}",'
            f'metrics_bruto!$B:$B,A{r}'
        )
        aba.cell(row=r, column=2, value=f"={soma})")
        for coluna, prefixo in zip((3, 4, 5), "245", strict=True):
            aba.cell(row=r, column=coluna, value=f'={soma},metrics_bruto!$C:$C,"{prefixo}*")')
        aba.cell(row=r, column=6, value=f"=IF(B{r}=0,0,(D{r}+E{r})/B{r})")
        aba.cell(row=r, column=7, value=f"=IF(B{r}=0,0,E{r}/B{r})")
        aba[f"F{r}"].number_format = aba[f"G{r}"].number_format = "0.0%"
        por_codigo = {}
        for linha in linhas:
            if linha["metrica"] == PEDIDOS and linha["endpoint"] == endpoint:
                classe = linha["http_response_code"][:1]
                por_codigo[classe] = por_codigo.get(classe, 0) + linha["valor"]
        total = sum(por_codigo.values())
        erros = por_codigo.get("4", 0) + por_codigo.get("5", 0)
        resultado.append((endpoint, total, erros, erros / total if total else 0.0))
    aba.column_dimensions["A"].width = 28
    for coluna in "BCDEFG":
        aba.column_dimensions[coluna].width = 16
    aba.column_dimensions["F"].width = 24
    return resultado


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:3000")
    parser.add_argument("--saida", type=Path, default=Path("docs/analise_metrics.xlsx"))
    args = parser.parse_args()

    try:
        texto, linhas = ler_metrics(args.url)
    except httpx.HTTPError as erro:
        print(f"não consegui ler {args.url}/metrics: {erro}. O serviço está rodando?", file=sys.stderr)
        return 1
    faixas = faixas_por_endpoint(linhas)
    if not faixas:
        print("o /metrics ainda não tem pedidos registrados; gere carga antes (just carga).", file=sys.stderr)
        return 1

    livro = Workbook()
    leia_me = livro.active
    leia_me.title = "leia_me"
    instante = datetime.now().astimezone().isoformat(timespec="seconds")
    for i, texto_linha in enumerate(
        [
            "Análise do GET /metrics do serviço BLU (BentoML, formato Prometheus)",
            f"Foto tirada em {instante} de {args.url}/metrics",
            "",
            "Como ler o /metrics:",
            f"- {PEDIDOS}: contador de pedidos por endpoint e código HTTP (aba taxa_erros).",
            f"- {LATENCIA}_bucket: histograma ACUMULADO da latência. Cada faixa 'le' conta os",
            "  pedidos que levaram até 'le' segundos (inclui as faixas anteriores).",
            "  A quantidade dentro de uma faixa é a diferença entre faixas vizinhas (abas lat_*).",
            f"- {LATENCIA}_sum e _count: soma das latências e número de pedidos (média = sum/count).",
            "- Quartis e boxplot são APROXIMADOS: só se sabe em que faixa cada pedido caiu, então o",
            "  valor é interpolado dentro da faixa (mesma regra do histogram_quantile do Prometheus).",
            "  O boxplot exato sai do log em arquivo (campo latencia_total_ms).",
            "",
            "Por quanto tempo isso fica guardado:",
            "- No /metrics: só na memória do processo. Os contadores acumulam desde que o serviço",
            "  subiu e voltam a zero quando ele reinicia. Não há histórico no tempo.",
            "- No Prometheus (docker-compose deste repositório): coleta a cada 5 s e guarda 15 dias",
            "  (--storage.tsdb.retention.time=15d).",
            "- Esta planilha é uma foto de um instante: rode de novo para atualizar.",
        ],
        start=1,
    ):
        leia_me.cell(row=i, column=1, value=texto_linha)
    leia_me["A1"].font = Font(bold=True, size=13)
    leia_me.column_dimensions["A"].width = 100

    bruto = livro.create_sheet("metrics_bruto")
    # chave_faixa ("le=0.005") casa as faixas por texto: um "0.005" puro seria
    # lido como número (ou não) conforme o separador decimal do Excel.
    _cabecalho(bruto, 1, ["metrica", "endpoint", "http_response_code", "le", "valor", "chave_faixa"])
    for i, linha in enumerate(linhas, start=2):
        for coluna, chave in enumerate(("metrica", "endpoint", "http_response_code", "le", "valor"), 1):
            bruto.cell(row=i, column=coluna, value=linha[chave])
        bruto.cell(row=i, column=6, value=f"le={linha['le']}" if linha["le"] else "")
    bruto.column_dimensions["A"].width = 48
    bruto.column_dimensions["B"].width = 26

    print("Latência das respostas 200 (aproximada pelas faixas do /metrics):")
    for endpoint, faixas_endpoint in sorted(faixas.items()):
        soma = sum(
            linha["valor"]
            for linha in linhas
            if linha["metrica"] == f"{LATENCIA}_sum"
            and linha["endpoint"] == endpoint
            and linha["http_response_code"] == "200"
        )
        r = aba_latencia(livro, endpoint, faixas_endpoint, soma)
        print(
            f"  {endpoint}: n={r['total']:.0f}  Q1={r['Q1']:.3f}s  mediana={r['Mediana']:.3f}s  "
            f"Q3={r['Q3']:.3f}s  p95={r['p95']:.3f}s  bigodes=[{r['whislo']:.3f}, {r['whishi']:.3f}]"
        )
    print("Taxa de erros:")
    for endpoint, total, erros, taxa in aba_erros(livro, linhas):
        print(f"  {endpoint}: {erros:.0f} de {total:.0f} pedidos ({taxa:.1%})")

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    livro.save(args.saida)
    (args.saida.with_suffix(".metrics.txt")).write_text(texto, encoding="utf-8")
    print(f"\nPlanilha em {args.saida} (e o texto bruto do /metrics em {args.saida.with_suffix('.metrics.txt')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

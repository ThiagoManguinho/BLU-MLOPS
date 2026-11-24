"""Submete um conjunto de imagens ao serviço e registra cada resposta.

Manda cada imagem de uma pasta, uma de cada vez e em sequência, para
`POST /prever_ocupacao`, e anota:
- o **tempo de resposta** medido no cliente (`time.perf_counter`, do envio até
  a resposta completa), que é a base dos percentis p90/p95/p99;
- o que o modelo devolveu (`ocupadas`, `livres`, `total_vagas`);
- o `request_id` da resposta (cabeçalho `X-Request-ID`).

Antes de começar, manda alguns pedidos de aquecimento que não entram na conta:
o primeiro pedido depois que o serviço sobe carrega o modelo e leva segundos.

No fim, procura cada `request_id` no log do servidor (`logs/inferencias.jsonl`,
mais os arquivos rotacionados) e acrescenta a `nitidez` da imagem e a
latência medida pelo próprio servidor. Assim a variável de entrada vem do
monitoramento da Parte 1, e não de um cálculo à parte.

Saída: estatisticas/respostas_<conjunto>.csv

Uso: uv run python scripts/submeter_conjunto.py --conjunto referencia --pasta train-maquete/images
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import httpx
from analisar_log import arquivos_de_log, ler_jsonl

_EXTENSOES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
_ENDPOINT = "/prever_ocupacao"
_COLUNAS = [
    "conjunto",
    "ordem",
    "arquivo",
    "request_id",
    "status_http",
    "tempo_resposta_ms",
    "latencia_servidor_ms",
    "nitidez",
    "ocupadas",
    "livres",
    "total_vagas",
]


def enviar(cliente: httpx.Client, caminho: Path) -> tuple[httpx.Response, float]:
    arquivo = (caminho.name, caminho.read_bytes(), _EXTENSOES[caminho.suffix.lower()])
    inicio = time.perf_counter()
    resposta = cliente.post(_ENDPOINT, files={"imagem": arquivo})
    resposta.read()
    return resposta, (time.perf_counter() - inicio) * 1000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--conjunto", required=True, choices=("referencia", "producao"))
    parser.add_argument("--pasta", type=Path, required=True)
    parser.add_argument("--url", default="http://localhost:3000")
    parser.add_argument("--aquecimento", type=int, default=3)
    parser.add_argument("--log", type=Path, default=Path("logs/inferencias.jsonl"))
    parser.add_argument("--saida", type=Path, default=Path("estatisticas"))
    args = parser.parse_args()

    imagens = sorted(p for p in args.pasta.glob("*") if p.suffix.lower() in _EXTENSOES)
    if not imagens:
        print(f"nenhuma imagem em {args.pasta}", file=sys.stderr)
        return 1

    linhas = []
    with httpx.Client(base_url=args.url, timeout=120) as cliente:
        for _ in range(args.aquecimento):
            enviar(cliente, imagens[0])

        for ordem, caminho in enumerate(imagens, start=1):
            resposta, tempo_ms = enviar(cliente, caminho)
            corpo = resposta.json() if resposta.status_code == 200 else {}
            linhas.append(
                {
                    "conjunto": args.conjunto,
                    "ordem": ordem,
                    "arquivo": caminho.name,
                    "request_id": resposta.headers.get("x-request-id", ""),
                    "status_http": resposta.status_code,
                    "tempo_resposta_ms": round(tempo_ms, 3),
                    "ocupadas": corpo.get("ocupadas", ""),
                    "livres": corpo.get("livres", ""),
                    "total_vagas": corpo.get("total_vagas", ""),
                }
            )
            print(
                f"[{ordem:>4}/{len(imagens)}] {caminho.name:<32} -> {resposta.status_code} "
                f"{tempo_ms:7.1f} ms  ocupadas={corpo.get('ocupadas', '-')}"
            )

    # Junta com o log do servidor pelo request_id.
    log = {
        registro.get("request_id"): registro
        for registro in ler_jsonl(arquivos_de_log(args.log))
    }
    sem_log = 0
    for linha in linhas:
        registro = log.get(linha["request_id"])
        if registro is None:
            sem_log += 1
            linha["nitidez"] = linha["latencia_servidor_ms"] = ""
            continue
        linha["nitidez"] = (registro.get("entrada") or {}).get("nitidez", "")
        linha["latencia_servidor_ms"] = registro.get("latencia_total_ms", "")

    args.saida.mkdir(parents=True, exist_ok=True)
    destino = args.saida / f"respostas_{args.conjunto}.csv"
    with destino.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=_COLUNAS)
        escritor.writeheader()
        escritor.writerows(linhas)

    falhas = sum(1 for linha in linhas if linha["status_http"] != 200)
    print(f"\n{len(linhas)} pedidos gravados em {destino}")
    if falhas:
        print(f"ATENÇÃO: {falhas} pedidos não voltaram 200", file=sys.stderr)
    if sem_log:
        print(
            f"ATENÇÃO: {sem_log} pedidos não foram achados em {args.log}; "
            "confira se o serviço grava o log nessa pasta",
            file=sys.stderr,
        )
    return 1 if falhas or sem_log else 0


if __name__ == "__main__":
    sys.exit(main())

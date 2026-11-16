"""Gera carga no serviço para ver o log e as métricas evoluírem.

Envia N pedidos aos dois endpoints de inferência com as imagens de uma pasta
(padrão: samples/), misturando de propósito:
- arquivos inválidos (`--fracao-erro`), para aparecerem respostas 400;
- com `--degradar`, imagens borradas, escurecidas ou estouradas, geradas aqui
  no cliente a partir das originais. Elas simulam lente suja/fora de foco e
  problemas de iluminação, e devem mover a nitidez e a luminância no log, no
  /metrics e no Grafana (contador `blu_imagens_suspeitas_total`).

Uso: uv run python scripts/gerar_carga.py [--n 100] [--intervalo 0.2] [--degradar]
"""

from __future__ import annotations

import argparse
import io
import random
import sys
import time
from collections import Counter
from pathlib import Path

import httpx
from PIL import Image, ImageEnhance, ImageFilter

_EXTENSOES = {".jpg", ".jpeg", ".png"}
_ENDPOINTS = ("/prever_ocupacao", "/prever_ocupacao_imagem")
_ARQUIVO_INVALIDO = b"isto nao e uma imagem, e um arquivo de texto\n"


def _degradar(dados: bytes, tipo: str) -> bytes:
    imagem = Image.open(io.BytesIO(dados)).convert("RGB")
    if tipo == "borrada":
        imagem = imagem.filter(ImageFilter.GaussianBlur(4))
    elif tipo == "escura":
        imagem = ImageEnhance.Brightness(imagem).enhance(0.45)
    elif tipo == "estourada":
        imagem = ImageEnhance.Brightness(imagem).enhance(1.8)
    saida = io.BytesIO()
    imagem.save(saida, format="JPEG", quality=90)
    return saida.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:3000")
    parser.add_argument("--n", type=int, default=100, help="quantidade de pedidos")
    parser.add_argument("--intervalo", type=float, default=0.2, help="segundos entre pedidos")
    parser.add_argument("--pasta", type=Path, default=Path("samples"))
    parser.add_argument("--fracao-erro", type=float, default=0.1)
    parser.add_argument(
        "--degradar",
        nargs="?",
        type=float,
        const=0.3,
        default=0.0,
        metavar="FRACAO",
        help="fração das imagens degradadas no cliente (só --degradar = 0.3)",
    )
    parser.add_argument("--semente", type=int, default=None)
    args = parser.parse_args()

    aleatorio = random.Random(args.semente)
    imagens = sorted(p for p in args.pasta.glob("*") if p.suffix.lower() in _EXTENSOES)
    if not imagens:
        print(f"nenhuma imagem em {args.pasta}", file=sys.stderr)
        return 1
    conteudo = {p: p.read_bytes() for p in imagens}

    status = Counter()
    tipos = Counter()
    latencias = []
    with httpx.Client(base_url=args.url, timeout=120) as cliente:
        for i in range(1, args.n + 1):
            endpoint = aleatorio.choices(_ENDPOINTS, weights=(0.8, 0.2))[0]
            if aleatorio.random() < args.fracao_erro:
                tipo, nome = "invalido", "arquivo_invalido.txt"
                arquivo = (nome, _ARQUIVO_INVALIDO, "text/plain")
            else:
                caminho = aleatorio.choice(imagens)
                dados = conteudo[caminho]
                tipo = "original"
                if aleatorio.random() < args.degradar:
                    tipo = aleatorio.choice(("borrada", "escura", "estourada"))
                    dados = _degradar(dados, tipo)
                arquivo = (caminho.name, dados, "image/jpeg")

            inicio = time.perf_counter()
            try:
                resposta = cliente.post(endpoint, files={"imagem": arquivo})
                codigo = resposta.status_code
                request_id = resposta.headers.get("x-request-id", "-")
            except httpx.HTTPError as erro:
                codigo, request_id = f"falha ({type(erro).__name__})", "-"
            latencias.append((time.perf_counter() - inicio) * 1000)
            status[codigo] += 1
            tipos[tipo] += 1
            print(f"[{i:>4}/{args.n}] {endpoint:<24} {tipo:<10} -> {codigo}  {request_id}")
            time.sleep(args.intervalo)

    latencias.sort()
    print("\nResumo")
    print("  status HTTP:", dict(status))
    print("  tipos de imagem:", dict(tipos))
    print(
        f"  latência no cliente (ms): mediana {latencias[len(latencias) // 2]:.0f}, "
        f"máx {latencias[-1]:.0f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

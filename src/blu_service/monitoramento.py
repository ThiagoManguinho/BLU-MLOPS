"""Monitoramento do serviço: o que entra, o que sai e quanto demora.

Três peças, todas usadas por `service.py`:

1. **Características** (`caracteristicas_imagem`, `caracteristicas_saida`):
   números que resumem a imagem recebida e a resposta do modelo. São as
   variáveis que permitem montar histogramas, boxplots e comparar com o
   conjunto de referência (KS/PSI) depois.
2. **Log em arquivo** (`MiddlewareRegistro` + `escrever_registro`): uma linha
   JSON por pedido em `logs/inferencias.jsonl`, append-only, com rotação por
   tamanho. O middleware envolve o app inteiro, então até o pedido que o
   BentoML recusa antes de chegar ao endpoint (arquivo que não é imagem) é
   registrado.
3. **Métricas Prometheus próprias** (`blu_*`): aparecem em `GET /metrics` ao
   lado das métricas que o BentoML já expõe (`bentoml_*`).

A métrica de qualidade acompanhada é a **nitidez** (variância do Laplaciano):
a câmera da maquete é fixa e de foco fixo, e o modelo depende das linhas finas
que delimitam as vagas. Foto fora de foco, lente suja ou tremida apagam essas
linhas antes de o modelo errar. A exposição (luminância, pixels
estourados/escuros) e a contagem de vagas (a maquete tem 7) completam o
diagnóstico. Compressão JPEG extrema não derruba a nitidez (os blocos criam
bordas novas); ela aparece nos bytes por pixel (`content_length` / pixels),
que também vão para o log.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image, ImageStat
from prometheus_client import Counter, Gauge, Histogram

from .logic import ResultadoOcupacao

VERSAO_LOG = 1

_LOG_PATH = os.environ.get("BLU_LOG_PATH", "logs/inferencias.jsonl")
_LOG_MAX_BYTES = int(os.environ.get("BLU_LOG_MAX_BYTES", str(10 * 1024 * 1024)))
_LOG_BACKUPS = int(os.environ.get("BLU_LOG_BACKUPS", "5"))

VAGAS_ESPERADAS = int(os.environ.get("BLU_VAGAS_ESPERADAS", "7"))

# A nitidez é medida numa cópia com esta largura, para não depender da
# resolução da câmera (a mesma cena em 1600 px tem bordas "mais suaves" por
# pixel do que em 640 px).
_LARGURA_NITIDEZ = 640

# Limites de "imagem suspeita", tirados do conjunto de referência (as 245 fotos
# de treino, ver referencia/resumo.json). Ajustáveis por variável de ambiente.
# - nitidez < 15: abaixo do p1 da referência (14,5). Um desfoque gaussiano de
#   raio 2 leva as fotos de exemplo de ~130 para ~14.
# - luminância média < 100: abaixo do p1 (106,5); a foto mais escura do treino
#   tem 88,7. Escurecer as fotos de exemplo para 45% leva a ~57-76.
# - fração de pixels estourados (>= 250) > 1%: o máximo da referência é 0,07%.
LIMITE_NITIDEZ_MIN = float(os.environ.get("BLU_LIMITE_NITIDEZ_MIN", "15"))
LIMITE_LUMINANCIA_MIN = float(os.environ.get("BLU_LIMITE_LUMINANCIA_MIN", "100"))
LIMITE_ESTOURADA_MAX = float(os.environ.get("BLU_LIMITE_ESTOURADA_MAX", "0.01"))

# Rotas de infraestrutura que não são pedidos de inferência.
ROTAS_IGNORADAS = ("/metrics", "/healthz", "/livez", "/readyz", "/docs", "/favicon.ico")


# --- Características da entrada e da saída --------------------------------------


def nitidez(imagem: Image.Image) -> float:
    """Variância do Laplaciano 3x3 da imagem em cinza: quanto maior, mais
    bordas definidas. Cai quando a foto está borrada ou muito comprimida."""

    cinza = imagem.convert("L")
    if cinza.width > _LARGURA_NITIDEZ:
        altura = max(1, round(cinza.height * _LARGURA_NITIDEZ / cinza.width))
        cinza = cinza.resize((_LARGURA_NITIDEZ, altura), Image.Resampling.BILINEAR)
    a = np.asarray(cinza, dtype=np.float32)
    if a.shape[0] < 3 or a.shape[1] < 3:
        return 0.0
    laplaciano = (
        a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:] - 4 * a[1:-1, 1:-1]
    )
    return float(laplaciano.var())


def caracteristicas_imagem(imagem: Image.Image) -> dict:
    """Resumo numérico da imagem recebida (variáveis de entrada do log)."""

    histograma = imagem.convert("L").histogram()
    total = sum(histograma) or 1
    media = sum(i * n for i, n in enumerate(histograma)) / total
    variancia = sum(n * (i - media) ** 2 for i, n in enumerate(histograma)) / total
    r, g, b = ImageStat.Stat(imagem.convert("RGB")).mean

    return {
        "formato": imagem.format,
        "modo": imagem.mode,
        "largura": imagem.width,
        "altura": imagem.height,
        "megapixels": round(imagem.width * imagem.height / 1e6, 4),
        "luminancia_media": round(media, 4),
        "luminancia_desvio": round(variancia**0.5, 4),
        "rgb_media": [round(r, 4), round(g, 4), round(b, 4)],
        "fracao_estourada": round(sum(histograma[250:]) / total, 6),
        "fracao_escura": round(sum(histograma[:6]) / total, 6),
        "nitidez": round(nitidez(imagem), 4),
        "histograma_l": histograma,
    }


def _area_poligono(pontos: list[list[float]]) -> float:
    soma = 0.0
    for (x1, y1), (x2, y2) in zip(pontos, pontos[1:] + pontos[:1], strict=True):
        soma += x1 * y2 - x2 * y1
    return abs(soma) / 2


def _media(valores: list[float]) -> float | None:
    return round(sum(valores) / len(valores), 4) if valores else None


def caracteristicas_saida(resultado: ResultadoOcupacao, largura: int, altura: int) -> dict:
    """Resumo da resposta do modelo (variáveis de saída do log)."""

    area_imagem = (largura * altura) or 1
    deteccoes = [
        {
            "estado": vaga.estado,
            "confianca": vaga.confianca,
            "area_rel": round(_area_poligono(vaga.obb) / area_imagem, 6),
            "cx_rel": round(sum(x for x, _ in vaga.obb) / len(vaga.obb) / largura, 4),
            "cy_rel": round(sum(y for _, y in vaga.obb) / len(vaga.obb) / altura, 4),
        }
        for vaga in resultado.vagas
    ]
    confiancas = [d["confianca"] for d in deteccoes]

    return {
        "total_vagas": resultado.total_vagas,
        "livres": resultado.livres,
        "ocupadas": resultado.ocupadas,
        "taxa_ocupacao": resultado.taxa_ocupacao,
        "descartadas_baixa_confianca": resultado.vagas_descartadas_baixa_confianca,
        "confianca_media": _media(confiancas),
        "confianca_min": min(confiancas, default=None),
        "confianca_max": max(confiancas, default=None),
        "confianca_media_livre": _media(
            [d["confianca"] for d in deteccoes if d["estado"] == "LIVRE"]
        ),
        "confianca_media_ocupada": _media(
            [d["confianca"] for d in deteccoes if d["estado"] == "OCUPADA"]
        ),
        "vagas_esperadas": VAGAS_ESPERADAS,
        "vagas_divergentes": resultado.total_vagas != VAGAS_ESPERADAS,
        "deteccoes": deteccoes,
    }


def motivos_suspeita(entrada: dict, saida: dict | None) -> list[str]:
    """Por que esta imagem merece atenção, comparando com os limites tirados
    do conjunto de referência. Lista vazia = nada fora do comum."""

    motivos = []
    if entrada.get("nitidez", LIMITE_NITIDEZ_MIN) < LIMITE_NITIDEZ_MIN:
        motivos.append("borrada")
    if entrada.get("luminancia_media", LIMITE_LUMINANCIA_MIN) < LIMITE_LUMINANCIA_MIN:
        motivos.append("escura")
    if entrada.get("fracao_estourada", 0.0) > LIMITE_ESTOURADA_MAX:
        motivos.append("estourada")
    if saida and saida.get("vagas_divergentes"):
        motivos.append("vagas_divergentes")
    return motivos


# --- Métricas Prometheus próprias -------------------------------------------------

# Faixas escolhidas a partir da distribuição do conjunto de referência
# (nitidez de 8,7 a 493, mediana 149; luminância de 89 a 189, mediana 130),
# com folga dos dois lados para imagens degradadas.
_FAIXAS_NITIDEZ = (5, 10, 15, 25, 50, 75, 100, 150, 200, 300, 400, 500, 750)
_FAIXAS_LUMINANCIA = (40, 60, 80, 100, 120, 140, 160, 180, 200, 220)
_FAIXAS_VAGAS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
_FAIXAS_CONFIANCA = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)

METRICA_NITIDEZ = Histogram(
    "blu_imagem_nitidez",
    "Nitidez da imagem recebida (variância do Laplaciano, cópia com 640 px de largura).",
    labelnames=("endpoint",),
    buckets=_FAIXAS_NITIDEZ,
)
METRICA_LUMINANCIA = Histogram(
    "blu_imagem_luminancia",
    "Luminância média da imagem recebida (0 a 255).",
    labelnames=("endpoint",),
    buckets=_FAIXAS_LUMINANCIA,
)
METRICA_VAGAS = Histogram(
    "blu_vagas_detectadas",
    "Quantidade de vagas detectadas por imagem (a maquete tem 7).",
    labelnames=("endpoint",),
    buckets=_FAIXAS_VAGAS,
)
METRICA_CONFIANCA = Histogram(
    "blu_confianca_deteccao",
    "Confiança de cada detecção aceita pelo limiar, por estado da vaga.",
    labelnames=("estado",),
    buckets=_FAIXAS_CONFIANCA,
)
METRICA_TAXA_OCUPACAO = Gauge(
    "blu_taxa_ocupacao",
    "Taxa de ocupação da última imagem analisada (ocupadas / total detectado).",
    multiprocess_mode="livemostrecent",
)
METRICA_SUSPEITAS = Counter(
    "blu_imagens_suspeitas",
    "Imagens fora da faixa do conjunto de referência, por motivo.",
    labelnames=("motivo",),
)


def atualizar_metricas(endpoint: str, entrada: dict, saida: dict, motivos: list[str]) -> None:
    METRICA_NITIDEZ.labels(endpoint=endpoint).observe(entrada["nitidez"])
    METRICA_LUMINANCIA.labels(endpoint=endpoint).observe(entrada["luminancia_media"])
    METRICA_VAGAS.labels(endpoint=endpoint).observe(saida["total_vagas"])
    for deteccao in saida["deteccoes"]:
        METRICA_CONFIANCA.labels(estado=deteccao["estado"]).observe(deteccao["confianca"])
    METRICA_TAXA_OCUPACAO.set(saida["taxa_ocupacao"])
    for motivo in motivos:
        METRICA_SUSPEITAS.labels(motivo=motivo).inc()


# --- Log em arquivo --------------------------------------------------------------


class FormatadorJsonl(logging.Formatter):
    """Uma linha JSON por evento. O horário vai em ISO 8601 com microssegundos
    e fuso horário (o formato mais detalhado que o `datetime` oferece), mais o
    instante Unix para ordenar e calcular intervalos sem ambiguidade."""

    def format(self, record: logging.LogRecord) -> str:
        instante = datetime.fromtimestamp(record.created).astimezone()
        evento = {
            "ts": instante.isoformat(timespec="microseconds"),
            "ts_unix": round(record.created, 6),
            **getattr(record, "registro", {}),
        }
        return json.dumps(evento, ensure_ascii=False, separators=(",", ":"))


_logger: logging.Logger | None = None


def configurar_log(
    caminho: str | Path | None = None,
    max_bytes: int | None = None,
    backups: int | None = None,
) -> logging.Logger:
    """Cria (ou recria) o logger `blu.inferencias`.

    `RotatingFileHandler` abre o arquivo em modo append: cada evento só
    acrescenta uma linha. Quando o arquivo passa de `max_bytes`, ele vira
    `inferencias.jsonl.1` (o `.1` anterior vira `.2`, e assim por diante até
    `backups`) e um arquivo novo começa. Nenhuma linha é reescrita.
    """

    global _logger
    caminho = Path(caminho or _LOG_PATH)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("blu.inferencias")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    handler = RotatingFileHandler(
        caminho,
        mode="a",
        maxBytes=_LOG_MAX_BYTES if max_bytes is None else max_bytes,
        backupCount=_LOG_BACKUPS if backups is None else backups,
        encoding="utf-8",
    )
    handler.setFormatter(FormatadorJsonl())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    _logger = logger
    return logger


def escrever_registro(registro: dict) -> None:
    (_logger or configurar_log()).info("inferencia", extra={"registro": registro})


# --- Middleware: uma linha de log por pedido -------------------------------------


@dataclass
class _Resposta:
    status: int = 500
    corpo_erro: bytes = b""


class MiddlewareRegistro:
    """Middleware ASGI que registra cada pedido de inferência.

    Ao entrar, cria `scope["state"]["registro"]`, um dicionário que o endpoint
    preenche (pelo `ctx.request.state.registro`) com as características da
    imagem e da resposta. Ao sair, junta status HTTP e latência total e grava
    uma única linha. Como envolve o app inteiro, registra também os pedidos
    recusados pelo BentoML antes do endpoint (ex.: arquivo que não é imagem).
    """

    _LIMITE_CORPO_ERRO = 2000

    def __init__(self, app, modelo: str = "", limiar_confianca: float | None = None):
        self.app = app
        self.modelo = modelo
        self.limiar_confianca = limiar_confianca

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"].startswith(ROTAS_IGNORADAS):
            await self.app(scope, receive, send)
            return

        inicio = time.perf_counter()
        request_id = uuid4().hex
        registro: dict = {}
        scope.setdefault("state", {})["registro"] = registro
        resposta = _Resposta()

        async def enviar(mensagem):
            if mensagem["type"] == "http.response.start":
                resposta.status = mensagem["status"]
                cabecalhos = list(mensagem.get("headers", []))
                cabecalhos.append((b"x-request-id", request_id.encode()))
                mensagem = {**mensagem, "headers": cabecalhos}
            elif (
                mensagem["type"] == "http.response.body"
                and resposta.status >= 400
                and len(resposta.corpo_erro) < self._LIMITE_CORPO_ERRO
            ):
                resposta.corpo_erro += mensagem.get("body", b"")
            await send(mensagem)

        try:
            await self.app(scope, receive, enviar)
        finally:
            cabecalhos = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]}
            entrada = {
                "content_type": cabecalhos.get("content-type", "").split(";")[0] or None,
                "content_length": int(cabecalhos.get("content-length", 0) or 0),
                **registro.get("entrada", {}),
            }
            erro = registro.get("erro")
            if erro is None and resposta.status >= 400:
                erro = {"codigo": resposta.status, "detalhe": _texto_erro(resposta.corpo_erro)}

            escrever_registro(
                {
                    "versao_log": VERSAO_LOG,
                    "request_id": request_id,
                    "pid": os.getpid(),
                    "metodo": scope["method"],
                    "endpoint": scope["path"],
                    "status_http": resposta.status,
                    "latencia_total_ms": round((time.perf_counter() - inicio) * 1000, 3),
                    "latencia_modelo_ms": registro.get("latencia_modelo_ms"),
                    "modelo": self.modelo,
                    "limiar_confianca": self.limiar_confianca,
                    "entrada": entrada,
                    "saida": registro.get("saida"),
                    "alertas": registro.get("alertas", []),
                    "erro": erro,
                }
            )


def _texto_erro(corpo: bytes) -> object:
    """Corpo da resposta de erro: o JSON decodificado quando for JSON, senão o
    texto cru (truncado). Assim o log guarda o motivo do 400/503."""

    if not corpo:
        return None
    texto = corpo[: MiddlewareRegistro._LIMITE_CORPO_ERRO].decode("utf-8", errors="replace")
    try:
        return json.loads(texto)
    except ValueError:
        return texto

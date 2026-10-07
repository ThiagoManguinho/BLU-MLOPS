from __future__ import annotations

import os
from pathlib import Path

import bentoml
from PIL import Image

from .logic import Deteccao

# `bentoml.importing()` marca a ultralytics (e o torch, que ela carrega) como
# dependência de *runtime*: ela precisa existir quando o serviço atende uma
# requisição, mas não quando o `bentoml build` apenas inspeciona a API para
# montar o Bento. Sem isso, o build quebra em qualquer máquina onde o torch
# não possa ser importado — por exemplo, no Windows com Smart App Control
# ligado, que bloqueia as DLLs não assinadas do PyTorch.
with bentoml.importing():
    from ultralytics import YOLO

_MODEL_PATH = os.environ.get("BLU_MODEL_PATH", "best.pt")

_modelo: YOLO | None = None


def _get_modelo() -> YOLO:
    global _modelo
    if _modelo is None:
        caminho = Path(_MODEL_PATH)
        if not caminho.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado em '{caminho}'. Ajuste BLU_MODEL_PATH."
            )
        _modelo = YOLO(str(caminho))
    return _modelo


def modelo_carregado() -> bool:
    return _modelo is not None


def nome_modelo() -> str:
    return f"{Path(_MODEL_PATH).name} (yolov8n-obb, ultralytics)"


def prever(imagem: Image.Image) -> list[Deteccao]:
    """Roda o modelo sobre uma imagem e devolve as detecções brutas (sem
    filtro de confiança, sem agregação)."""
    modelo = _get_modelo()
    resultados = modelo(imagem, verbose=False)
    resultado = resultados[0]

    deteccoes: list[Deteccao] = []
    obb = getattr(resultado, "obb", None)
    if obb is None or obb.cls is None:
        return deteccoes

    nomes = resultado.names
    for cls_id, conf, corners in zip(
        obb.cls.tolist(), obb.conf.tolist(), obb.xyxyxyxy.tolist()
    ):
        classe = nomes[int(cls_id)]
        pontos = [[round(x, 2), round(y, 2)] for x, y in corners]
        deteccoes.append(Deteccao(classe=classe, confianca=round(conf, 4), obb=pontos))

    return deteccoes

"""Testes do monitoramento: características da imagem e da resposta, formato
do log em arquivo e middleware (sem carregar o modelo)."""

import json
import logging
import re

from PIL import Image, ImageDraw, ImageFilter
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from blu_service import monitoramento
from blu_service.logic import ResultadoOcupacao, Vaga


def _imagem_com_linhas() -> Image.Image:
    """Imagem branca com linhas pretas finas, como as marcações das vagas."""
    imagem = Image.new("RGB", (400, 300), "white")
    desenho = ImageDraw.Draw(imagem)
    for x in range(20, 400, 40):
        desenho.line([(x, 0), (x, 300)], fill="black", width=2)
    return imagem


def _resultado(n_vagas: int) -> ResultadoOcupacao:
    vagas = [
        Vaga(
            id=i,
            estado="OCUPADA" if i % 2 else "LIVRE",
            confianca=0.8,
            obb=[[0, 0], [100, 0], [100, 50], [0, 50]],
        )
        for i in range(n_vagas)
    ]
    ocupadas = sum(1 for v in vagas if v.estado == "OCUPADA")
    return ResultadoOcupacao(
        total_vagas=n_vagas,
        livres=n_vagas - ocupadas,
        ocupadas=ocupadas,
        taxa_ocupacao=round(ocupadas / n_vagas, 4) if n_vagas else 0.0,
        vagas=vagas,
    )


def test_nitidez_cai_quando_a_imagem_e_borrada():
    nitida = _imagem_com_linhas()
    borrada = nitida.filter(ImageFilter.GaussianBlur(4))

    assert monitoramento.nitidez(nitida) > 10 * monitoramento.nitidez(borrada)


def test_nitidez_nao_depende_da_resolucao():
    imagem = _imagem_com_linhas().resize((1600, 1200), Image.Resampling.NEAREST)
    pequena = imagem.resize((640, 480), Image.Resampling.BILINEAR)

    assert monitoramento.nitidez(imagem) == monitoramento.nitidez(pequena)


def test_luminancia_de_imagem_preta_e_branca():
    preta = monitoramento.caracteristicas_imagem(Image.new("RGB", (10, 10), "black"))
    branca = monitoramento.caracteristicas_imagem(Image.new("RGB", (10, 10), "white"))

    assert preta["luminancia_media"] == 0
    assert preta["fracao_escura"] == 1.0
    assert branca["luminancia_media"] == 255
    assert branca["fracao_estourada"] == 1.0


def test_histograma_tem_256_faixas_que_somam_os_pixels():
    entrada = monitoramento.caracteristicas_imagem(_imagem_com_linhas())

    assert len(entrada["histograma_l"]) == 256
    assert sum(entrada["histograma_l"]) == 400 * 300


def test_saida_com_as_7_vagas_esperadas_nao_diverge():
    saida = monitoramento.caracteristicas_saida(_resultado(7), 200, 100)

    assert saida["total_vagas"] == 7
    assert saida["vagas_divergentes"] is False
    assert saida["confianca_media"] == 0.8
    assert saida["deteccoes"][0]["area_rel"] == 0.25  # 100x50 numa imagem 200x100
    assert saida["deteccoes"][0]["cx_rel"] == 0.25


def test_saida_com_vagas_faltando_diverge_e_vira_suspeita():
    saida = monitoramento.caracteristicas_saida(_resultado(5), 200, 100)

    assert saida["vagas_divergentes"] is True
    assert "vagas_divergentes" in monitoramento.motivos_suspeita({}, saida)


def test_motivos_de_suspeita_pela_imagem():
    entrada_ruim = {"nitidez": 3.0, "luminancia_media": 40.0, "fracao_estourada": 0.5}
    entrada_boa = {"nitidez": 150.0, "luminancia_media": 130.0, "fracao_estourada": 0.0}

    assert monitoramento.motivos_suspeita(entrada_ruim, None) == ["borrada", "escura", "estourada"]
    assert monitoramento.motivos_suspeita(entrada_boa, None) == []


def test_formatador_gera_uma_linha_json_com_microssegundos_e_fuso():
    registro = logging.LogRecord("blu", logging.INFO, "", 0, "inferencia", None, None)
    registro.registro = {"endpoint": "/prever_ocupacao", "status_http": 200}

    linha = monitoramento.FormatadorJsonl().format(registro)

    assert "\n" not in linha
    evento = json.loads(linha)
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}[+-]\d\d:\d\d", evento["ts"])
    assert evento["endpoint"] == "/prever_ocupacao"


def test_log_so_acrescenta_linhas_e_rotaciona(tmp_path):
    caminho = tmp_path / "inferencias.jsonl"
    monitoramento.configurar_log(caminho, max_bytes=2000, backups=3)
    try:
        for i in range(60):
            monitoramento.escrever_registro({"i": i, "preenchimento": "x" * 50})
    finally:
        monitoramento.configurar_log(tmp_path / "descartado.jsonl")

    arquivos = [caminho, *sorted(tmp_path.glob("inferencias.jsonl.*"))]
    assert len(arquivos) == 4  # o atual + 3 antigos (backupCount)
    for arquivo in arquivos:
        linhas = arquivo.read_text(encoding="utf-8").splitlines()
        indices = [json.loads(linha)["i"] for linha in linhas]
        assert indices == sorted(indices)  # dentro de cada arquivo, só acrescenta
    assert json.loads(caminho.read_text(encoding="utf-8").splitlines()[-1])["i"] == 59


def test_middleware_grava_uma_linha_por_pedido_inclusive_erro(tmp_path):
    caminho = tmp_path / "inferencias.jsonl"
    monitoramento.configurar_log(caminho)

    async def inferir(request):
        request.state.registro["saida"] = {"total_vagas": 7}
        return JSONResponse({"ok": True})

    async def recusar(request):
        return JSONResponse({"erro": "arquivo_invalido"}, status_code=400)

    app = Starlette(
        routes=[
            Route("/prever_ocupacao", inferir, methods=["POST"]),
            Route("/recusado", recusar, methods=["POST"]),
            Route("/metrics", lambda request: JSONResponse({})),
        ]
    )
    cliente = TestClient(monitoramento.MiddlewareRegistro(app, modelo="teste", limiar_confianca=0.25))
    try:
        ok = cliente.post("/prever_ocupacao", content=b"x")
        cliente.post("/recusado", content=b"x")
        cliente.get("/metrics")
    finally:
        monitoramento.configurar_log(tmp_path / "descartado.jsonl")

    eventos = [json.loads(linha) for linha in caminho.read_text(encoding="utf-8").splitlines()]
    assert [e["endpoint"] for e in eventos] == ["/prever_ocupacao", "/recusado"]  # /metrics fora
    assert eventos[0]["status_http"] == 200
    assert eventos[0]["saida"] == {"total_vagas": 7}
    assert eventos[0]["request_id"] == ok.headers["x-request-id"]
    assert eventos[1]["status_http"] == 400
    assert eventos[1]["erro"] == {"codigo": 400, "detalhe": {"erro": "arquivo_invalido"}}
    assert eventos[1]["latencia_total_ms"] >= 0

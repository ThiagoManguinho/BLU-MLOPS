"""Testes de `render.py`: a imagem anotada tem que ser um JPEG válido do
mesmo tamanho da imagem original, sem depender do modelo de verdade."""

from PIL import Image

from blu_service.logic import ResultadoOcupacao, Vaga
from blu_service.render import anotar_imagem


def test_anotar_imagem_devolve_jpeg_valido_do_mesmo_tamanho():
    imagem = Image.new("RGB", (200, 150), color="white")
    resultado = ResultadoOcupacao(
        total_vagas=2,
        livres=1,
        ocupadas=1,
        taxa_ocupacao=0.5,
        vagas=[
            Vaga(id=0, estado="LIVRE", confianca=0.9, obb=[[0, 0], [50, 0], [50, 50], [0, 50]]),
            Vaga(
                id=1,
                estado="OCUPADA",
                confianca=0.8,
                obb=[[100, 0], [150, 0], [150, 50], [100, 50]],
            ),
        ],
    )

    jpeg_bytes = anotar_imagem(imagem, resultado)

    from io import BytesIO

    anotada = Image.open(BytesIO(jpeg_bytes))
    assert anotada.format == "JPEG"
    assert anotada.size == (200, 150)


def test_anotar_imagem_sem_vagas_devolve_imagem_intacta():
    imagem = Image.new("RGB", (10, 10), color="blue")
    resultado = ResultadoOcupacao(total_vagas=0, livres=0, ocupadas=0, taxa_ocupacao=0.0)

    jpeg_bytes = anotar_imagem(imagem, resultado)

    from io import BytesIO

    anotada = Image.open(BytesIO(jpeg_bytes))
    assert anotada.size == (10, 10)

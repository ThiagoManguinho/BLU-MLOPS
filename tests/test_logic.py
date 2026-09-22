"""Testes da regra de negócio (sem carregar o modelo)."""

from blu_service.logic import Deteccao, classificar_ocupacao


def _det(classe: str, confianca: float) -> Deteccao:
    return Deteccao(classe=classe, confianca=confianca, obb=[[0, 0], [1, 0], [1, 1], [0, 1]])


def test_vaga_vira_livre_e_carro_vira_ocupada():
    deteccoes = [_det("vaga", 0.9), _det("carro", 0.9)]

    resultado = classificar_ocupacao(deteccoes, limiar_confianca=0.25)

    assert resultado.total_vagas == 2
    assert resultado.livres == 1
    assert resultado.ocupadas == 1
    assert {v.estado for v in resultado.vagas} == {"LIVRE", "OCUPADA"}


def test_taxa_ocupacao_e_calculada_corretamente():
    deteccoes = [_det("carro", 0.9), _det("carro", 0.9), _det("vaga", 0.9), _det("vaga", 0.9)]

    resultado = classificar_ocupacao(deteccoes, limiar_confianca=0.25)

    assert resultado.taxa_ocupacao == 0.5


def test_deteccao_abaixo_do_limiar_e_descartada_nao_classificada():
    deteccoes = [_det("carro", 0.9), _det("vaga", 0.1)]

    resultado = classificar_ocupacao(deteccoes, limiar_confianca=0.25)

    assert resultado.total_vagas == 1
    assert resultado.vagas_descartadas_baixa_confianca == 1


def test_sem_deteccoes_nao_divide_por_zero():
    resultado = classificar_ocupacao([], limiar_confianca=0.25)

    assert resultado.total_vagas == 0
    assert resultado.taxa_ocupacao == 0.0


def test_classe_desconhecida_e_descartada_sem_quebrar():
    deteccoes = [_det("carro", 0.9), _det("moto", 0.9)]

    resultado = classificar_ocupacao(deteccoes, limiar_confianca=0.25)

    assert resultado.total_vagas == 1
    assert resultado.vagas_descartadas_baixa_confianca == 1

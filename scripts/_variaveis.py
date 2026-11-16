"""Variáveis numéricas do log usadas no resumo da referência e na análise.

Cada item é (nome na tabela, seção da linha do log, chave). As mesmas
variáveis valem para `logs/inferencias.jsonl` e `referencia/referencia.jsonl`,
que têm o mesmo formato.
"""

from __future__ import annotations

VARIAVEIS = [
    # Entrada: como a imagem chegou (iluminação, nitidez, tamanho).
    ("luminancia_media", "entrada", "luminancia_media"),
    ("luminancia_desvio", "entrada", "luminancia_desvio"),
    ("fracao_estourada", "entrada", "fracao_estourada"),
    ("fracao_escura", "entrada", "fracao_escura"),
    ("nitidez", "entrada", "nitidez"),
    ("megapixels", "entrada", "megapixels"),
    ("content_length", "entrada", "content_length"),
    # Saída: o que o modelo respondeu.
    ("total_vagas", "saida", "total_vagas"),
    ("livres", "saida", "livres"),
    ("ocupadas", "saida", "ocupadas"),
    ("taxa_ocupacao", "saida", "taxa_ocupacao"),
    ("confianca_media", "saida", "confianca_media"),
    ("confianca_min", "saida", "confianca_min"),
    ("descartadas_baixa_confianca", "saida", "descartadas_baixa_confianca"),
    # Desempenho.
    ("latencia_modelo_ms", None, "latencia_modelo_ms"),
    ("latencia_total_ms", None, "latencia_total_ms"),
]


def valor(linha: dict, secao: str | None, chave: str) -> float | None:
    origem = linha if secao is None else (linha.get(secao) or {})
    v = origem.get(chave)
    if isinstance(v, bool) or v is None:
        return None
    return float(v)

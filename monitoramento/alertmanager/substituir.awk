# Troca cada ${NOME} da linha pelo valor da variável de ambiente NOME.
# Feito "à mão" (index/substr) em vez de sub/gsub para que senhas com &, \ ou
# outros caracteres especiais sejam copiadas literalmente. Aspas simples são
# dobradas, porque os valores vão entre '...' no YAML.
{
    linha = $0
    saida = ""
    while ((inicio = index(linha, "${")) > 0) {
        resto = substr(linha, inicio + 2)
        fim = index(resto, "}")
        if (fim == 0) break
        nome = substr(resto, 1, fim - 1)
        valor = ENVIRON[nome]
        escapado = ""
        while ((p = index(valor, "'")) > 0) {
            escapado = escapado substr(valor, 1, p) "'"
            valor = substr(valor, p + 1)
        }
        saida = saida substr(linha, 1, inicio - 1) escapado valor
        linha = substr(resto, fim + 1)
    }
    print saida linha
}

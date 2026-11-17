# Como analisar o `GET /metrics` do serviço BLU

O `GET /metrics` é exposto pelo próprio BentoML, no
[formato de texto do Prometheus](https://prometheus.io/docs/instrumenting/exposition_formats/).
Ele não precisa de nada além do serviço rodando:

```bash
curl http://localhost:3000/metrics
```

Este texto explica o que cada série significa e como tirar dela o
**histograma** e o **boxplot da latência** e a **taxa de erros**. Também diz
**por quanto tempo** cada informação fica guardada.

A planilha [`analise_metrics.xlsx`](analise_metrics.xlsx) faz as mesmas contas
com fórmulas do Excel visíveis, sobre uma foto real do `/metrics`. O texto bruto
dessa foto está em [`analise_metrics.metrics.txt`](analise_metrics.metrics.txt).
Os números do exemplo abaixo saem dela: 160 pedidos ao `/prever_ocupacao`,
gerados com `just carga`.

Para gerar uma foto nova: `just serve`, depois `just carga` e por fim
`just planilha`. O script `scripts/exportar_metrics.py` lê o `/metrics`, monta
a planilha e imprime os mesmos números calculados em Python, para conferir as
fórmulas.

## 1. Como o texto é organizado

Cada métrica vem com duas linhas de comentário e depois uma linha por
combinação de rótulos:

```text
# HELP bentoml_service_request_total Total number of runner RPC
# TYPE bentoml_service_request_total counter
bentoml_service_request_total{endpoint="/prever_ocupacao",http_response_code="200",...} 148.0
bentoml_service_request_total{endpoint="/prever_ocupacao",http_response_code="400",...} 12.0
```

- `# TYPE` diz o tipo da métrica:
  - `counter`: só cresce;
  - `gauge`: sobe e desce;
  - `histogram`: contagem por faixas.
- Os rótulos entre chaves (`endpoint`, `http_response_code`, `le`…) separam as
  séries. Os rótulos `runner_name`, `service_name` e `service_version`
  aparecem em todas as linhas e foram omitidos aqui.

### Séries que o BentoML já traz

| Série | Tipo | O que mede |
|---|---|---|
| `bentoml_service_request_total{endpoint, http_response_code}` | counter | Pedidos respondidos, por endpoint e código HTTP. Base da **taxa de erros**. |
| `bentoml_service_request_duration_seconds_bucket{endpoint, http_response_code, le}` | histogram | Quantos pedidos levaram **até** `le` segundos. Base do **histograma** e do **boxplot**. |
| `bentoml_service_request_duration_seconds_sum` / `_count` | histogram | Soma das latências e número de pedidos. A média é `_sum ÷ _count`. |
| `bentoml_service_request_in_progress{endpoint}` | gauge | Pedidos sendo atendidos naquele instante. |
| `bentoml_service_last_request_timestamp_seconds` | gauge | Instante Unix do último pedido. |

### Séries acrescentadas pelo projeto (`blu_*`)

Elas medem a **qualidade da imagem** e a **resposta do modelo**, que o BentoML
não conhece. São geradas em `src/blu_service/monitoramento.py`.

| Série | Tipo | O que mede |
|---|---|---|
| `blu_imagem_nitidez{endpoint}` | histogram | Nitidez (variância do Laplaciano). Abaixo de 15, a foto está borrada. |
| `blu_imagem_luminancia{endpoint}` | histogram | Luminância média (0 a 255). Abaixo de 100, a foto está escura. |
| `blu_vagas_detectadas{endpoint}` | histogram | Vagas detectadas por imagem (a maquete tem 7). |
| `blu_confianca_deteccao{estado}` | histogram | Confiança de cada detecção, por estado (LIVRE/OCUPADA). |
| `blu_taxa_ocupacao` | gauge | Taxa de ocupação da última imagem. |
| `blu_imagens_suspeitas_total{motivo}` | counter | Imagens fora da faixa da referência: `borrada`, `escura`, `estourada`, `vagas_divergentes`. |

Os histogramas `blu_*` são lidos exatamente como o de latência descrito a seguir.

## 2. Histograma da latência

As faixas do `/metrics` são **acumuladas**: `le="0.25"` conta todos os
pedidos que levaram **até** 0,25 s, incluindo os que já estavam em
`le="0.1"`. A quantidade **dentro** de uma faixa é a diferença entre faixas
vizinhas:

```text
pedidos na faixa (a, b] = acumulado(le = b) − acumulado(le = a)
```

O exemplo usa as respostas 200 do `/prever_ocupacao`. Os erros 400 respondem
em milissegundos e distorceriam o tempo de inferência; eles entram na taxa de
erros (seção 4).

| Faixa (s) | `le` | Acumulado (do `/metrics`) | Pedidos na faixa | % |
|---|---|---:|---:|---:|
| 0 – 0,05 | 0.05 | 0 | 0 | 0,0% |
| 0,05 – 0,1 | 0.1 | 29 | 29 − 0 = **29** | 19,6% |
| 0,1 – 0,25 | 0.25 | 63 | 63 − 29 = **34** | 23,0% |
| 0,25 – 0,5 | 0.5 | 144 | 144 − 63 = **81** | 54,7% |
| 0,5 – 1 | 1.0 | 146 | **2** | 1,4% |
| 1 – 2,5 | 2.5 | 147 | **1** | 0,7% |
| 2,5 – 5 | 5.0 | 147 | 0 | 0,0% |
| 5 – 10 | 10.0 | 148 | **1** | 0,7% |
| acima de 10 | +Inf | 148 | 0 | 0,0% |

- O total de pedidos é o acumulado da faixa `+Inf`, igual a `_count`: 148.
- A média é `_sum ÷ _count` = 43,467 ÷ 148 = **0,294 s**.
- Os dois pedidos mais lentos têm causa conhecida, confirmada no log em
  arquivo:
  - o de 1 a 2,5 s é a **primeira inferência** depois que o serviço subiu. O
    modelo é carregado nessa chamada (`latencia_modelo_ms` de 1.632 ms);
  - o de 5 a 10 s (7,5 s) foi enviado no momento em que o Docker subia o
    Prometheus e o Grafana na mesma máquina, com a CPU disputada.

  Os dois são pontos fora da curva esperados, que pesam pouco na média depois
  de alguns pedidos.

Essas faixas são fixas no BentoML: 5 ms, 10 ms, 25 ms, 50 ms, 100 ms, 250 ms,
500 ms, 1 s, 2,5 s, 5 s, 10 s, 30 s, 60 s, 120 s e 180 s. O histograma só
mostra o que cabe nessa resolução. Não dá para saber, por exemplo, como os 81
pedidos se espalham entre 0,25 e 0,5 s.

## 3. Boxplot da latência (aproximado)

O boxplot precisa de Q1, mediana, Q3 e bigodes. O `/metrics` não guarda as
latências individuais, só em que faixa cada pedido caiu. Por isso os quartis
são **estimados por interpolação linear dentro da faixa**, a mesma regra do
`histogram_quantile` do Prometheus.

Para o quantil `q`:

1. posição = `q × total`;
2. ache a primeira faixa cujo acumulado é maior ou igual à posição;
3. valor ≈ `limite_inferior + (limite_superior − limite_inferior) × (posição − acumulado_anterior) ÷ pedidos_na_faixa`.

| Estatística | Posição | Faixa | Conta | Valor |
|---|---:|---|---|---:|
| Q1 | 0,25 × 148 = 37 | 0,1 – 0,25 | 0,1 + 0,15 × (37 − 29) ÷ 34 | **0,135 s** |
| Mediana | 0,50 × 148 = 74 | 0,25 – 0,5 | 0,25 + 0,25 × (74 − 63) ÷ 81 | **0,284 s** |
| Q3 | 0,75 × 148 = 111 | 0,25 – 0,5 | 0,25 + 0,25 × (111 − 63) ÷ 81 | **0,398 s** |
| p95 | 0,95 × 148 = 140,6 | 0,25 – 0,5 | 0,25 + 0,25 × (140,6 − 63) ÷ 81 | **0,490 s** |

Bigodes pela regra de Tukey (1,5 × IQR), limitados às faixas que têm pedidos:

- IQR = Q3 − Q1 = 0,263 s;
- bigode inferior = máx(Q1 − 1,5 × IQR, início da 1ª faixa com pedidos) = máx(−0,259; 0,05) = **0,05 s**;
- bigode superior = mín(Q3 + 1,5 × IQR, fim da última faixa com pedidos) = mín(0,792; 10) = **0,792 s**.

Pontos acima de 0,792 s são outliers. No exemplo, ao menos os dois pedidos
das faixas de 1 a 2,5 s e de 5 a 10 s; os dois da faixa de 0,5 a 1 s podem ou
não passar de 0,792 s, e o `/metrics` não permite saber.

**Limitação:** a interpolação supõe pedidos espalhados por igual dentro da
faixa. O erro máximo é a largura da faixa: entre 0,25 e 0,5 s, até 0,25 s. O
mínimo e o máximo reais não aparecem no `/metrics`.

**Boxplot exato:** use o log em arquivo. Cada linha de
`logs/inferencias.jsonl` tem `latencia_total_ms` (o pedido inteiro, medido no
servidor) e `latencia_modelo_ms` (só a inferência). O `just analisar` desenha
o boxplot exato dessas latências em `analise/latencia_modelo_ms.png`.

## 4. Taxa de erros

O contador `bentoml_service_request_total` separa os pedidos por código HTTP.
A taxa de erros é:

```text
taxa de erros = (pedidos 4xx + pedidos 5xx) ÷ total de pedidos
```

| Endpoint | Total | 2xx | 4xx | 5xx | Taxa de erros |
|---|---:|---:|---:|---:|---:|
| `/prever_ocupacao` | 160 | 148 | 12 | 0 | 12 ÷ 160 = **7,5%** |
| `/prever_ocupacao_imagem` | 43 | 38 | 5 | 0 | 5 ÷ 43 = **11,6%** |

- **4xx** é erro do cliente, por exemplo um arquivo que não é imagem (o
  `just carga` manda 10% assim de propósito).
- **5xx** é falha do serviço, por exemplo o modelo indisponível. Para saúde do
  serviço, a taxa de 5xx é a que mais importa.

### Taxa no tempo (Prometheus)

Os contadores do `/metrics` são **totais desde que o serviço subiu**, então a
conta acima é a taxa média de toda a vida do processo. Para a taxa "agora",
olha-se o quanto o contador cresceu numa janela de tempo. É o que o Prometheus
faz com `rate`, e o que mostra o painel do Grafana:

```promql
sum(rate(bentoml_service_request_total{http_response_code=~"4..|5.."}[5m]))
  / sum(rate(bentoml_service_request_total[5m]))
```

Da mesma forma, o p95 da latência nos últimos 5 minutos:

```promql
histogram_quantile(0.95, sum by (le) (rate(bentoml_service_request_duration_seconds_bucket[5m])))
```

## 5. Por quanto tempo essas informações ficam guardadas

| Onde | Guarda o quê | Por quanto tempo |
|---|---|---|
| **`GET /metrics` (BentoML)** | Totais acumulados desde que o processo subiu, sem histórico no tempo. | **Só enquanto o processo vive.** Os valores ficam em memória e em arquivos temporários do modo multiprocesso do `prometheus_client`. Ao reiniciar o serviço, tudo **volta a zero**. |
| **Prometheus** (`just up` ou `just monitor-up`) | Uma amostra de cada série a cada 5 s, ou seja, o histórico no tempo. | **15 dias** (`--storage.tsdb.retention.time=15d` no `docker-compose.yml`), no volume `prometheus-dados`. Sobrevive a reinícios do serviço: o `rate()` reconhece quando um contador volta a zero. |
| **Grafana** | Nada: consulta o Prometheus na hora de desenhar. | Depende do Prometheus. |
| **Log em arquivo** (`logs/inferencias.jsonl`) | Uma linha por pedido, com o que entrou e o que saiu. | Até 6 arquivos de 10 MB (o atual e mais 5). Com cerca de 2,5 KB por linha, são uns 25 mil pedidos; depois disso o mais antigo é descartado. Configurável por `BLU_LOG_MAX_BYTES` e `BLU_LOG_BACKUPS`. |
| **Esta planilha** | Uma foto de um instante. | Fica como está; rode `just planilha` para atualizar. |

Por isso o `/metrics` sozinho só responde "como está desde que o serviço
subiu". Para "como estava ontem às 14h", é preciso o Prometheus coletando de
forma contínua, ou o log em arquivo.

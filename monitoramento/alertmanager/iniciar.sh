#!/bin/sh
# Monta a configuração do Alertmanager a partir das variáveis de ambiente e
# inicia o Alertmanager.
#
# Sem configuração nenhuma, os e-mails vão para o Mailpit (caixa de e-mail
# local do docker-compose, em http://localhost:8025). Para mandar e-mail de
# verdade, preencha o .env.alertas (modelo em .env.alertas.example).
set -eu

: "${SMTP_SERVIDOR:=mailpit:1025}"
: "${SMTP_REMETENTE:=blu-alertas@blu.local}"
: "${SMTP_USUARIO:=}"
: "${SMTP_SENHA:=}"
: "${SMTP_TLS:=false}"
: "${ALERTA_EMAIL_PARA:=equipe-blu@blu.local}"
export SMTP_SERVIDOR SMTP_REMETENTE SMTP_USUARIO SMTP_SENHA SMTP_TLS ALERTA_EMAIL_PARA

awk -f /etc/alertmanager/substituir.awk /etc/alertmanager/alertmanager.yml.modelo \
    > /tmp/alertmanager.yml

echo "Alertmanager: e-mails para ${ALERTA_EMAIL_PARA} via ${SMTP_SERVIDOR}"
exec /bin/alertmanager \
    --config.file=/tmp/alertmanager.yml \
    --storage.path=/alertmanager

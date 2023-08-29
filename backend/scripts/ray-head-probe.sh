#!/bin/sh

# Lógica de verificação de prontidão e vivacidade
# Este script executa o comando "ray status" para verificar a integridade

# Executar o comando "ray status" e redirecionar saída para o null
ray status --address $RAY_ADDRESS_PROBE > /dev/null 2>&1

# Verificar o código de saída do comando
if [ $? -eq 0 ]; then
  # O comando foi executado com sucesso (integridade verificada)
  exit 0
else
  # O comando falhou (integridade não verificada)
  exit 1
fi

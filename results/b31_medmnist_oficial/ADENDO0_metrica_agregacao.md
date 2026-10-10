# Adendo 0 — B3.1: métrica e agregação no código oficial (antes do B3.0 terminar)

**Data:** 2026-10-05, antes de ver qualquer resultado do B3.0 e de qualquer run do B3.1. O `PREREGISTRO.md` (commit `315b880`) não muda; este adendo só **declara** como o código oficial mede e agrega, para fechar na Parte I a mesma questão aberta na Parte II (C0b, `VERIFICACOES.md` §3; B2.8s).

## Métrica (código oficial, `external/AdaAggRL/exp_environments.py`)

- A acurácia registrada (`history['acc']`, usada na primária e nas AUCs) é `test(self.net, self.testloader)`, avaliada **no test set global inteiro**, num único `DataLoader` (batch 64, `shuffle=False`, `drop_last=True`). Não é uma média de acurácias por cliente.
- **MNIST:** test set oficial (10.000), aproximadamente balanceado por classe.
- **BloodMNIST:** test set oficial (3.421; 3.392 avaliados por causa do `drop_last`), **desbalanceado por classe**: 243 a 666 exemplos por classe. A métrica é a acurácia na distribuição natural do teste, sem balanceamento.
- A recompensa do TD3 é a diferença da **perda somada** sobre o mesmo `testloader`.

## Agregação (código oficial)

- **AdaAggRL (`fixed` e `td3`):** a agregação pondera os modelos dos clientes por `k`, calculado da ação e dos scores (estado) de cada cliente, com min-max, limiar a₅, penalidade de memória e normalização. **Não usa o tamanho da amostra dos clientes.**
- **FedAvg do B3.0** (condição `fedavg` do runner): `average`, ou seja, **média uniforme** dos modelos dos clientes, sem ponderação por amostra.
- **Tamanhos dos clientes:** a partição `_build_groups_by_q` faz grupos por classe dominante, de tamanhos diferentes, e divide cada grupo em partes iguais. Os clientes têm tamanhos diferentes entre grupos, mas nenhuma das agregações acima usa esse tamanho.

## Consequência para a leitura do B3.1

- A comparação fixed × td3 é entre dois pesos **sem tamanho de amostra**, avaliados na mesma métrica (acurácia no teste global). A confusão "ponderação por amostra × uniforme" encontrada no framework próprio **não se aplica** a ela.
- O teto do B3.0 (FedAvg sem ataque) também usa **média uniforme**, coerente com os sistemas comparados. No BloodMNIST, o teste é desbalanceado, então a margem M herda a escala da acurácia nessa distribuição natural; isso fica declarado.

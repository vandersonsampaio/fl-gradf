# Adendo 1 — B2.4r: verificação do centro falhou; centro refeito na grade

**Data:** 2026-10-02 20:36:26 -0300, antes de qualquer run da grade. Previsto no §3 do `PLANO.md`.

## Verificação (PLANO §3)

- O runner novo (`scripts/passo2_oficial/run_b24r.py`, que só troca `run_oficial.A_FIXED` e chama `run_oficial.run`, o mesmo runner do Passo 2) foi rodado com a₅ = 0,475, semente 100, 25 rodadas.
- **Não reproduziu** o `fixed` EB do Passo 2: máx. |Δacc| = 8,4 p.p. nas 25 rodadas.
- O estado inicial é idêntico (acurácia e perda do reset inicial iguais), com a mesma versão do torch (2.3.0+cu121) e o mesmo dispositivo (cuda:0). A divergência começa **na rodada 1**.
- **Diagnóstico:** o código oficial na GPU **não é bit-reprodutível entre execuções** (operações cuDNN não determinísticas). A semente fixa a inicialização, mas não a trajetória.

## Consequência (regra do PLANO §3)

- O **centro (a₅ = 0,475) é rodado de novo** nas sementes 100–104, dentro da mesma grade. A grade passa a ter **25 runs** (5 valores de a₅ × 5 sementes), ~35 h de GPU com 5 processos.
- A análise usa o centro refeito. O centro do Passo 2 aparece só como referência descritiva (`centro_passo2`).
- Critérios, métricas e leitura **não mudam**.
- **Nota para o P2:** o não-determinismo da GPU vale para todos os experimentos no código oficial (Passo 2, B2.1–B2.5). Lá o pareamento por semente pareia a configuração, não a trajetória. Os testes pareados continuam válidos, porque cada run é uma amostra da mesma distribuição.

## Código no congelamento
dc95bbbd2312ed4cb73ec2855b22ac83980f19f9dde8f5baf4d367c051675c99  scripts/passo2_oficial/run_b24r.py
715e40e4a405c8e9a9ef07e6267acf84e82a7249f28b23c722e20bd9fde28f2d  scripts/passo2_oficial/run_grid_b24r.sh
794d9adc9d1b808bedf27ef7b2f08ce896daab4318c3224998ee771256396ba2  scripts/passo2_oficial/analisar_b24r.py

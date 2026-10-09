# Plano — B3.3: varredura curta do limiar a₅ no BloodMNIST (AdaAggRL oficial, EB)

**Status:** FINAL antes de qualquer run (hash em `PLANO.sha256`). **Exploratório.** O código ainda não existe: ele será escrito, testado (§4) e terá o hash registrado em adendo **antes** do disparo, sem mudar nada deste plano.
**Data:** 2026-10-09
**Antecedentes:**
- **B2.4r** (MNIST, EB; `results/b24r_limiar_oficial/`): o limiar é causal no código publicado. a₅ ≤ 0,1 colapsa ou custa caro, e a₅ ∈ [0,25; 0,95] forma um platô a menos de 0,4 p.p. do centro. Nenhuma constante supera o centro (Q2: não).
- **B3.1** (BloodMNIST; `results/b31_medmnist_oficial/RESULTADO.md`): fixed ≈ td3 (±3 p.p.), num regime degradado e de resets periódicos. A acurácia é ~31% (contra 77,8% sem ataque), há um reset a cada ~23 rodadas e a massa de peso nos atacantes é ~0,016 (contra ≤ 0,0001 no MNIST).

## 1. Pergunta

No BloodMNIST, sob EB, **um limiar mais alto que o centro** (que exclui mais clientes por rodada) supera a ação central? Ou seja: o regime degradado do B3.1 é, ao menos em parte, um limiar mal calibrado para este dataset, que uma constante simples corrigiria?

A pergunta é sobre constantes, não sobre aprendizado. Se a resposta for sim, o TD3 publicado também não encontra a constante melhor (B3.2: a política fica parada na inicialização), e o P2 ganha a mesma leitura do B2.7b: "a contribuição possível do RL equivale, no máximo, a ajustar uma constante".

## 2. Desenho

- **Código:** oficial (`external/AdaAggRL`, commit `27b9c18`), venv `external/.venv_adaaggrl`, sem nenhuma edição.
  - Adaptação ao BloodMNIST: a mesma do B3.1 (`bloodmnist_shim.py`, extrator e dados com os hashes do Adendo 1 do B3.1).
  - Runner novo `run_b33.py`, que reusa `run_b3.run` e troca só `R.A_FIXED` no processo, como o `run_b24r.py` faz com o `run_oficial`.
- **Regime:** idêntico ao B3.1 (BloodMNIST, q = 0,5, 96 clientes, 10 por rodada, 20 atacantes, lr 0,05, 500 rodadas). Ataque: **EB** apenas.
- **Condições:** ação fixa [0,475; 0,475; 0,475; 0,475; a₅] com **a₅ ∈ {0,475 (centro); 0,75; 0,95}**. O Box oficial é [0; 0,95].
  - Só valores **acima** do centro entram: no B2.4r os valores abaixo de 0,25 colapsam ou custam caro, e a hipótese aqui é de filtragem insuficiente (massa nos atacantes maior que no MNIST).
- **O centro é rodado de novo nas sementes novas.** Ele não é reaproveitado do B3.1 (sementes 135–144), para manter o pareamento por semente e a mesma execução na GPU, que não é bit-reprodutível (B2.4r ADENDO1).
- **Sementes:** **145–149** (novas). Total: 3 × 5 = **15 runs**.
- **Execução:**
  - Fila **intercalada** por semente (centro, 0,75, 0,95 para cada semente), 6 processos na GPU. Assim cada leva mistura as três condições.
  - Janela de execução 7h–18h seg–sex com `scripts/janela_execucao.sh`.
  - Custo pelo throughput do B3.1 (~0,69 run/h): **~22 h de GPU**.
- **Registro:** o mesmo do B3.1 para `fixed` (acurácia, perda, ação executada, massa nos atacantes reais, resets, `a_fixed` no JSON). Checkpoints parciais a cada 25 rodadas. Um diretório por a₅ (`raw/a5_<valor>/`), porque o nome do arquivo não inclui a₅.

## 3. Métricas

- **Primária:** mediana da acurácia nas rodadas 401–500 (a mesma do B3.1).
- **Sensibilidade por AUC:** média da acurácia nas rodadas 1–500 e nas 251–500, como no B3.1.
- **Mecanismo (descritivas):** nº de resets por run e intervalo mediano entre resets; massa média de peso nos atacantes reais (rodadas com atacante).

## 4. Verificação antes da grade

- **Teste do runner** (2 rodadas, semente 100, fora da grade e descartado): para a₅ ∈ {0,475; 0,75; 0,95}, o JSON deve registrar `a_fixed` = [0,475]×4 + [a₅] e a ação executada em cada passo deve ser igual a `a_fixed`.
- Não há verificação de reprodução contra o B3.1, porque o centro é rodado de novo nas mesmas sementes da grade.

## 5. Critério (fixado agora; exploratório; n = 5)

Para cada a₅ ∈ {0,75; 0,95}: Δ = métrica(a₅) − métrica(centro), pareado por semente, com IC95 t (4 g.l.).

- **"Alguma constante supera o centro":** em pelo menos um a₅, **Δ > 2 p.p. com IC95 > 0 na primária.**
- **Ressalva de resets:** se o critério for atendido na primária mas Δ ≤ 0 na AUC 1–500, o resultado é reportado como **"sensível a resets"** e não é afirmado sem essa ressalva.
- **Sem correção de multiplicidade** (2 comparações), declarado. Wilcoxon só descritivo: com n = 5, o menor p bilateral possível é 0,0625.
- **Resets contam como resultado** (consequência da ação) e nunca são excluídos.
- **Leitura dos casos restantes:** Δ ≤ 2 p.p. ou IC95 que inclui 0 → **"nenhuma constante supera o centro"** nesta varredura. Δ < −2 p.p. com IC95 < 0 é reportado como "limite mais alto piora".

## 6. Cláusula de término

- **B3.3 é a única varredura do limiar no BloodMNIST.** Nenhum valor de a₅, ataque ou semente é acrescentado depois de ver o resultado, qualquer que seja ele.
- **Critério não atendido:** a pergunta do §1 fica fechada como "não há constante simples acima do centro que corrija o regime" (dentro da resolução de n = 5) e não se abre outra varredura.
- **Critério atendido:** o achado fica **exploratório**. Qualquer continuação (por exemplo, confirmar com sementes novas ou testar o TD3 com o limiar deslocado) exige um pré-registro novo e uma decisão do autor; não acontece automaticamente.
- Runs com JSON final completo nunca são rodados de novo (B3.1 §6). Só runs que falharem sem JSON final (erro de ambiente) são repetidos com a mesma semente, e isso é registrado no `grid.log` e no RESULTADO.
- **Falha de execução:** se mais de 3 dos 15 runs falharem sem JSON final mesmo depois de uma repetição, a grade é interrompida e o autor é consultado.

## 7. Leitura

- **Critério atendido:** o regime degradado do B3.1 é, em parte, um limiar mal calibrado para o BloodMNIST. Uma constante simples ganha do centro, e o TD3 publicado (parado na inicialização, B3.2) não a encontra. Isso reforça no P2 a leitura de que o aprendizado equivaleria, no máximo, a ajustar uma constante.
- **Critério não atendido:** a degradação não vem de o limiar estar baixo demais. Junto com a equivalência do B3.1, isso indica que nem a ação central nem os limiares mais altos defendem o BloodMNIST sob EB no ambiente oficial. O P2 reporta isso como limite do mecanismo publicado nesse dataset.

## 8. Regras

- Análise pré-escrita (`analisar_b33.py`, hash no adendo), rodada uma única vez com a grade completa.
- Nenhuma acurácia é inspecionada antes do fim da grade. O monitoramento reporta só saúde e progresso.
- **Ordem:**
  1. este plano commitado;
  2. código (`run_b33.py`, `run_grid_b33.sh`, `analisar_b33.py`) e teste do §4;
  3. adendo com os hashes;
  4. commit;
  5. disparo.
- Logs e checkpoints parciais não são versionados.

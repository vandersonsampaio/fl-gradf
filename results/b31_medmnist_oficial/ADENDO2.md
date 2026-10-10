# Adendo 2 — B3.1: limiar de reset e escala da recompensa

**Data:** 2026-10-05, ~17h40. **Cronologia declarada:** a grade do B3.1 foi disparada às 17h27 (Adendo 1); este adendo foi escrito ~15 min depois, **antes de qualquer dado do B3.1 ser inspecionado** (só existem as primeiras rodadas da 1ª leva, não olhadas). Ele **não muda nada** no desenho, nos critérios nem no código; só declara uma propriedade do ambiente oficial observada no B3.0.

## Declaração

**O limiar de reset e a escala da recompensa foram mantidos do código oficial.**
- Limiar: o episódio reinicia quando a recompensa < −80 (`exp_environments.py`, l. 271), constante oficial e não alterada.
- Recompensa: diferença da perda **somada** sobre o `testloader` do test set inteiro (batch 64, `drop_last=True`): **53 batches** no BloodMNIST (3.392 imagens avaliadas) contra **156** no MNIST.

**No BloodMNIST, o teste menor reduz a escala da recompensa em ~3×,** e o reset pode não disparar num colapso. Isso foi observado no **B3.0, EB, semente 131**: acurácia de 7,1%, abaixo do acaso para 8 classes (12,5%), ou seja, colapso para uma classe minoritária, com só 14 resets, contra ~113 nas outras duas sementes.

## Consequências para a leitura

- O efeito vale **igualmente para as duas condições** (fixed e td3): mesmo ambiente, mesmo limiar.
- Ele **muda a dinâmica em relação ao MNIST**: menos resets por colapso, e colapsos que podem persistir. Muda também **a escala da recompensa que o TD3 vê** (~⅓ da do MNIST). Isso deve ser lembrado ao comparar o B3.1 com o B2.1/B2.2.
- **O limiar não foi ajustado:** ajustá-lo depois de ver o B3.0 seria uma decisão guiada pelos dados.
- A sensibilidade por AUC (pré-registro §5) e a contagem de resets por condição continuam valendo e cobrem esse comportamento.

# Provenance of the frozen hashes

The plans, pre-registrations and addenda of the experiments in `results/` record SHA-256 hashes of themselves and of the code that ran (in `*.sha256` files and in the "code at freeze time" sections of the addenda). All of them were computed on the **original Portuguese versions**.

After the experiments, the code comments, docstrings and printed messages, and the non-frozen documents, were translated into English. To keep the frozen records verifiable:

- **Frozen documents** (`PLANO.md`, `PREREGISTRO.md`, `ADENDO*.md`, `NOTAS.md`) were **not edited**. Each has an English translation next to it (`*.en.md`); if the two ever disagree, the Portuguese original prevails. The `*.sha256` files themselves were also left untouched.
- **Frozen code** (scripts whose hash is recorded) had only comments, docstrings and human-readable strings translated; the program logic is unchanged (checked token by token, ignoring comments and string literals). Their hashes therefore no longer match the working tree. Verify them against the last commit before the translation, **`ab42d47`**, e.g.:

```bash
git show ab42d47:scripts/passo2_oficial/analisar_b31.py | sha256sum
# compare with the hash recorded in results/b31_medmnist_oficial/ADENDO1.md
```

## Notes

1. `scripts/b27_horizonte.py`: its hash in `results/b27_horizonte/PLANO.sha256` matches commit `8ab747f` (the plan's freeze), not `ab42d47`, because the script was corrected afterwards (oracle ceiling fix, documented in `results/b27_horizonte/RESULTADO.md`).
2. Two records did not match the committed file even before the translation, for reasons unrelated to it:
   - `results/c0_espaco_restante/NOTAS.md`: `NOTAS.sha256` records the file as frozen at its §5 (the Gate C0 criterion), before §6 (the decision) was appended on the same day; checked: the committed text up to the end of §5 (before the `---` that precedes §6) hashes to the recorded value.
   - `results/frente1_passo2_oficial/PREREGISTRO.md`: `PREREGISTRO.sha256` holds two hashes, the original one and the one after Addendum 1 (recorded with the relative path `PREREGISTRO.md`); the committed file matches the Addendum 1 hash.
3. `data/models/extract_feature_bloodmnist.pt` and `data/raw/medmnist/bloodmnist.npz` are generated or downloaded data, not versioned; their hashes refer to the local files (`scripts/passo2_oficial/treinar_extrator_bloodmnist.py` regenerates the extractor; the `.npz` comes from Zenodo record 10519652).

## Frozen files

| file | hash recorded in | verify against | current state |
|---|---|---|---|
| `data/models/extract_feature_bloodmnist.pt` | `results/b30_medmnist_sanity/CODIGO.sha256`, `results/b31_medmnist_oficial/ADENDO1.md` | not in git (data artifact); verify the local file directly | unchanged |
| `data/raw/medmnist/bloodmnist.npz` | `results/b30_medmnist_sanity/CODIGO.sha256`, `results/b31_medmnist_oficial/ADENDO1.md` | not in git (data artifact); verify the local file directly | unchanged |
| `results/a0_analises/PLANO.md` | `results/a0_analises/PLANO.sha256` | current file, or `ab42d47` | unchanged (English version in `*.en.md`) |
| `results/b21_replicacao_oficial/PREREGISTRO.md` | `results/b21_replicacao_oficial/PREREGISTRO.sha256` | current file, or `ab42d47` | unchanged (English version in `*.en.md`) |
| `results/b23_steelman_oficial/PLANO.md` | `results/b23_steelman_oficial/PLANO.sha256` | current file, or `ab42d47` | unchanged (English version in `*.en.md`) |
| `results/b23b_steelman_normalizado/PLANO.md` | `results/b23b_steelman_normalizado/PLANO.sha256` | current file, or `ab42d47` | unchanged (English version in `*.en.md`) |
| `results/b24r_limiar_oficial/ADENDO1.md` | `results/b24r_limiar_oficial/ADENDO1.sha256` | current file, or `ab42d47` | unchanged (English version in `*.en.md`) |
| `results/b24r_limiar_oficial/PLANO.md` | `results/b24r_limiar_oficial/PLANO.sha256` | current file, or `ab42d47` | unchanged (English version in `*.en.md`) |
| `results/b25_ipm_oficial/ADENDO1.md` | `results/b25_ipm_oficial/ADENDO1.sha256` | current file, or `ab42d47` | unchanged (English version in `*.en.md`) |
| `results/b25_ipm_oficial/PLANO.md` | `results/b25_ipm_oficial/PLANO.sha256` | current file, or `ab42d47` | unchanged (English version in `*.en.md`) |
| `results/b26_decomposicao/PREREGISTRO.md` | `results/b26_decomposicao/PREREGISTRO.sha256` | current file, or `ab42d47` | unchanged (English version in `*.en.md`) |
| `results/b27_horizonte/PLANO.md` | `results/b27_horizonte/PLANO.sha256` | current file, or `ab42d47` | unchanged (English version in `*.en.md`) |
| `results/b27b_td3_constante/ADENDO1.md` | `results/b27b_td3_constante/ADENDO1.sha256` | current file, or `ab42d47` | unchanged (English version in `*.en.md`) |
| `results/b27b_td3_constante/ADENDO2.md` | `results/b27b_td3_constante/ADENDO2.sha256` | current file, or `ab42d47` | unchanged (English version in `*.en.md`) |
| `results/b27b_td3_constante/PLANO.md` | `results/b27b_td3_constante/PLANO.sha256` | current file, or `ab42d47` | unchanged (English version in `*.en.md`) |
| `results/b28_oraculo_por_regra/PLANO.md` | `results/b28_oraculo_por_regra/PLANO.sha256` | current file, or `ab42d47` | unchanged (English version in `*.en.md`) |
| `results/b28s_sensibilidade_metrica/PLANO.md` | `results/b28s_sensibilidade_metrica/PLANO.sha256` | current file, or `ab42d47` | unchanged (English version in `*.en.md`) |
| `results/b31_medmnist_oficial/ADENDO0_metrica_agregacao.md` | `results/b31_medmnist_oficial/ADENDO1.sha256` | current file, or `ab42d47` | unchanged (English version in `*.en.md`) |
| `results/b31_medmnist_oficial/ADENDO1.md` | `results/b31_medmnist_oficial/ADENDO1.sha256` | current file, or `ab42d47` | unchanged (English version in `*.en.md`) |
| `results/b31_medmnist_oficial/ADENDO2.md` | `results/b31_medmnist_oficial/ADENDO2.sha256` | current file, or `ab42d47` | unchanged (English version in `*.en.md`) |
| `results/b31_medmnist_oficial/PREREGISTRO.md` | `results/b31_medmnist_oficial/PREREGISTRO.sha256` | current file, or `ab42d47` | unchanged (English version in `*.en.md`) |
| `results/b33_limiar_bloodmnist/PLANO.md` | `results/b33_limiar_bloodmnist/PLANO.sha256` | current file, or `ab42d47` (recorded with a relative path) | unchanged (English version in `*.en.md`) |
| `results/c0_espaco_restante/NOTAS.md` | `results/c0_espaco_restante/NOTAS.sha256` | see note 2 | unchanged (English version in `NOTAS.en.md`) |
| `results/c0b_espaco_h150/ADENDO1.md` | `results/c0b_espaco_h150/ADENDO1.sha256` | current file, or `ab42d47` | unchanged (English version in `*.en.md`) |
| `results/c0b_espaco_h150/PLANO.md` | `results/c0b_espaco_h150/PLANO.sha256` | current file, or `ab42d47` | unchanged (English version in `*.en.md`) |
| `results/c0b_iii_metrica/PLANO.md` | `results/c0b_iii_metrica/PLANO.sha256` | current file, or `ab42d47` | unchanged (English version in `*.en.md`) |
| `results/d2_variancia_p1/PLANO.md` | `results/d2_variancia_p1/PLANO.sha256` | current file, or `ab42d47` | unchanged (English version in `*.en.md`) |
| `results/frente1_passo2_oficial/PREREGISTRO.md` | `results/frente1_passo2_oficial/PREREGISTRO.sha256` | current file, or `ab42d47` (Addendum 1 hash, recorded with a relative path; see note 2) | unchanged (English version in `*.en.md`) |
| `scripts/a0_analises.py` | `results/a0_analises/PLANO.sha256` | `ab42d47` | translated (comments/strings) |
| `scripts/b26_decomposicao.py` | `results/b26_decomposicao/PREREGISTRO.sha256` | `ab42d47` | translated (comments/strings) |
| `scripts/b27_horizonte.py` | `results/b27_horizonte/PLANO.sha256` | `8ab747f` | translated (comments/strings) |
| `scripts/b27b_explicativo.py` | `results/b27b_td3_constante/ADENDO2.md` | `ab42d47` | translated (comments/strings) |
| `scripts/b27b_td3_constante.py` | `results/b27b_td3_constante/ADENDO1.md` | `ab42d47` | translated (comments/strings) |
| `scripts/b28_oraculo_por_regra.py` | `results/b28_oraculo_por_regra/PLANO.sha256` | `ab42d47` | translated (comments/strings) |
| `scripts/b28s_sensibilidade_metrica.py` | `results/b28s_sensibilidade_metrica/PLANO.sha256` | `ab42d47` | translated (comments/strings) |
| `scripts/c0b_espaco_h150.py` | `results/c0b_espaco_h150/ADENDO1.md` | `ab42d47` | translated (comments/strings) |
| `scripts/c0b_iii_metrica.py` | `results/c0b_iii_metrica/PLANO.sha256` | `ab42d47` | translated (comments/strings) |
| `scripts/d2_variancia_p1.py` | `results/d2_variancia_p1/PLANO.sha256` | `ab42d47` | translated (comments/strings) |
| `scripts/fila_cpu_d2_iii.sh` | `results/c0b_iii_metrica/PLANO.sha256` | `ab42d47` | translated (comments/strings) |
| `scripts/frente1_ablacao_adaaggrl.py` | `results/b27_horizonte/PLANO.sha256` | `ab42d47` | translated (comments/strings) |
| `scripts/janela_execucao.sh` | `results/b31_medmnist_oficial/ADENDO1.md` | `ab42d47` | translated (comments/strings) |
| `scripts/passo2_oficial/a0_regimes_td3.py` | `results/a0_analises/PLANO.sha256` | `ab42d47` | translated (comments/strings) |
| `scripts/passo2_oficial/analisar.py` | `results/b21_replicacao_oficial/PREREGISTRO.sha256` | `ab42d47` | translated (comments/strings) |
| `scripts/passo2_oficial/analisar_b21.py` | `results/b21_replicacao_oficial/PREREGISTRO.sha256`, `results/b23b_steelman_normalizado/PLANO.sha256` | `ab42d47` | translated (comments/strings) |
| `scripts/passo2_oficial/analisar_b23.py` | `results/b23_steelman_oficial/PLANO.sha256`, `results/b23b_steelman_normalizado/PLANO.sha256` | `ab42d47` | translated (comments/strings) |
| `scripts/passo2_oficial/analisar_b23b.py` | `results/b23b_steelman_normalizado/PLANO.sha256` | `ab42d47` | translated (comments/strings) |
| `scripts/passo2_oficial/analisar_b24r.py` | `results/b24r_limiar_oficial/ADENDO1.md` | `ab42d47` | translated (comments/strings) |
| `scripts/passo2_oficial/analisar_b25.py` | `results/b25_ipm_oficial/PLANO.sha256` | `ab42d47` | translated (comments/strings) |
| `scripts/passo2_oficial/analisar_b30.py` | `results/b30_medmnist_sanity/CODIGO.sha256` | `ab42d47` | translated (comments/strings) |
| `scripts/passo2_oficial/analisar_b31.py` | `results/b31_medmnist_oficial/ADENDO1.md` | `ab42d47` | translated (comments/strings) |
| `scripts/passo2_oficial/bloodmnist_shim.py` | `results/b30_medmnist_sanity/CODIGO.sha256`, `results/b31_medmnist_oficial/ADENDO1.md` | `ab42d47` | translated (comments/strings) |
| `scripts/passo2_oficial/run_b21.py` | `results/b21_replicacao_oficial/PREREGISTRO.sha256`, `results/b23_steelman_oficial/PLANO.sha256`, `results/b23b_steelman_normalizado/PLANO.sha256` | `ab42d47` | translated (comments/strings) |
| `scripts/passo2_oficial/run_b23.py` | `results/b23_steelman_oficial/PLANO.sha256` | `ab42d47` | translated (comments/strings) |
| `scripts/passo2_oficial/run_b23b.py` | `results/b23b_steelman_normalizado/PLANO.sha256` | `ab42d47` | translated (comments/strings) |
| `scripts/passo2_oficial/run_b24r.py` | `results/b24r_limiar_oficial/ADENDO1.md` | `ab42d47` | translated (comments/strings) |
| `scripts/passo2_oficial/run_b25.py` | `results/b25_ipm_oficial/PLANO.sha256` | `ab42d47` | translated (comments/strings) |
| `scripts/passo2_oficial/run_b3.py` | `results/b30_medmnist_sanity/CODIGO.sha256`, `results/b31_medmnist_oficial/ADENDO1.md` | `ab42d47` | translated (comments/strings) |
| `scripts/passo2_oficial/run_grid_b21.sh` | `results/b21_replicacao_oficial/PREREGISTRO.sha256` | `ab42d47` | translated (comments/strings) |
| `scripts/passo2_oficial/run_grid_b23.sh` | `results/b23_steelman_oficial/PLANO.sha256` | `ab42d47` | translated (comments/strings) |
| `scripts/passo2_oficial/run_grid_b23b.sh` | `results/b23b_steelman_normalizado/PLANO.sha256` | `ab42d47` | translated (comments/strings) |
| `scripts/passo2_oficial/run_grid_b24r.sh` | `results/b24r_limiar_oficial/ADENDO1.md` | `ab42d47` | translated (comments/strings) |
| `scripts/passo2_oficial/run_grid_b25.sh` | `results/b25_ipm_oficial/PLANO.sha256` | `ab42d47` | translated (comments/strings) |
| `scripts/passo2_oficial/run_grid_b30.sh` | `results/b30_medmnist_sanity/CODIGO.sha256` | `ab42d47` | translated (comments/strings) |
| `scripts/passo2_oficial/run_grid_b31.sh` | `results/b31_medmnist_oficial/ADENDO1.md` | `ab42d47` | translated (comments/strings) |
| `scripts/passo2_oficial/run_oficial.py` | `results/b21_replicacao_oficial/PREREGISTRO.sha256`, `results/b23_steelman_oficial/PLANO.sha256`, `results/b23b_steelman_normalizado/PLANO.sha256` | `ab42d47` | translated (comments/strings) |
| `scripts/passo2_oficial/test_b25_ipm.py` | `results/b25_ipm_oficial/PLANO.sha256` | `ab42d47` | translated (comments/strings) |
| `scripts/passo2_oficial/treinar_extrator_bloodmnist.py` | `results/b30_medmnist_sanity/CODIGO.sha256`, `results/b31_medmnist_oficial/ADENDO1.md` | `ab42d47` | translated (comments/strings) |
| `scripts/run_grid_b26.sh` | `results/b26_decomposicao/PREREGISTRO.sha256` | `ab42d47` | translated (comments/strings) |
| `scripts/run_grid_b27.sh` | `results/b27_horizonte/PLANO.sha256` | `ab42d47` | translated (comments/strings) |
| `scripts/run_grid_b27a.sh` | `results/b27b_td3_constante/ADENDO1.md` | `ab42d47` | translated (comments/strings) |
| `scripts/run_grid_b27b.sh` | `results/b27b_td3_constante/ADENDO2.md` | `ab42d47` | translated (comments/strings) |
| `scripts/run_grid_b28s.sh` | `results/b28s_sensibilidade_metrica/PLANO.sha256` | `ab42d47` | translated (comments/strings) |
| `scripts/run_grid_c0b.sh` | `results/c0b_espaco_h150/ADENDO1.md` | `ab42d47` | translated (comments/strings) |
| `scripts/run_grid_c0b_iii.sh` | `results/c0b_iii_metrica/PLANO.sha256` | `ab42d47` | translated (comments/strings) |
| `scripts/run_grid_d2.sh` | `results/d2_variancia_p1/PLANO.sha256` | `ab42d47` | translated (comments/strings) |

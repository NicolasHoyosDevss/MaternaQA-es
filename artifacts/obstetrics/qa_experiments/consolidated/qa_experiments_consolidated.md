# QA Synthetic Experiments Consolidated Report

| Experimento | Generador | Verificador | QA | Acceptance | Faith | Rel | Roundtrip | Overlap | Avg answer words | Meta refs en respuestas |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A_gpt54_gen_gpt55_eval | gpt-5.4 | gpt-5.5 | 17 | 100.0% | 0.988 | 0.983 | 0.978 | 0.701 | 66.9 | 8 |
| B_gpt54mini_gen_gpt55_eval | gpt-5.4-mini | gpt-5.5 | 17 | 94.1% | 0.938 | 0.945 | 0.934 | 0.563 | 35.6 | 0 |
| C_gpt52_gen_gpt55_eval | gpt-5.2 | gpt-5.5 | 17 | 100.0% | 0.992 | 0.995 | 0.992 | 0.702 | 47.1 | 0 |
| D_gpt54_gen_gpt54mini_eval | gpt-5.4 | gpt-5.4-mini | 17 | 94.1% | 0.923 | 0.936 | 0.918 | 0.596 | 54.2 | 8 |

## Notas
- `Meta refs en respuestas` cuenta respuestas que mencionan explícitamente el fragmento/contexto/texto.

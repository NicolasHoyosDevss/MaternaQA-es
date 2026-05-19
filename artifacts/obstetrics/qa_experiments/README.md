# QA experiment artifacts

Artefactos de experimentos previos para generación y evaluación de QA sintético.

## Estructura

```text
artifacts/obstetrics/qa_experiments/
├── model_comparison/
│   ├── A_gpt54_gen_gpt55_eval/
│   ├── B_gpt54mini_gen_gpt55_eval/
│   ├── C_gpt52_gen_gpt55_eval/
│   └── D_gpt54_gen_gpt54mini_eval/
├── ragas_diagnostics/
└── consolidated/
```

## Uso

- `model_comparison/`: corridas A/B/C/D usadas para elegir estrategia de generación.
- `ragas_diagnostics/`: pruebas parciales y diagnósticos de Ragas.
- `consolidated/`: reportes consolidados de los experimentos.

Estos archivos no son el dataset final de entrenamiento; se conservan como evidencia experimental.

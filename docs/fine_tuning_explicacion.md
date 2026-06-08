# Fine-tuning de MaternaQA-es en 5 minutos

Este proyecto hizo fine-tuning supervisado de modelos abiertos para responder preguntas clinicas en espanol sobre embarazo y salud materna. La ruta usada fue QLoRA con Hugging Face TRL, PEFT y bitsandbytes: no se reentrena todo el modelo, sino que se carga el modelo base en 4 bits y se entrenan pequenos adapters LoRA.

## Idea principal

El objetivo fue adaptar modelos instruct a un dataset propio de obstetricia llamado `MaternaQA-es`. Para el entrenamiento principal se uso la variante `sft_grounded`, donde cada ejemplo incluye contexto fuente mas pregunta, y el modelo aprende a responder apoyandose en esa evidencia.

En una frase para exponer:

> Entrenamos adapters QLoRA sobre modelos Gemma/MedGemma usando preguntas y respuestas clinicas con contexto fuente, para adaptar el comportamiento del modelo sin hacer full fine-tuning pesado.

## Donde esta implementado

| Parte | Ruta |
|---|---|
| Script de entrenamiento | `scripts/train_qlora_trl.py` |
| Dataset de entrenamiento | `datasets/obstetrics/qa/publication/sft_grounded/` |
| Dataset alternativo sin contexto | `datasets/obstetrics/qa/publication/sft_closed_book/` |
| Salida Gemma 4 | `outputs/gemma4-grounded/` |
| Salida MedGemma | `outputs/medgemma-grounded/` |
| Inferencia con adapters | `scripts/inference_qlora.py` |
| Evaluacion de predicciones | `scripts/evaluate_model_predictions.py` |

## Como funciona el flujo

1. Se cargan los archivos `train.jsonl`, `validation.jsonl` y `test.jsonl` desde `datasets/obstetrics/qa/publication/sft_grounded/`.
2. Cada ejemplo viene como conversacion `messages`; el script lo convierte en memoria a `prompt` y `completion`.
3. TRL calcula la perdida solo sobre la `completion`, es decir, sobre la respuesta esperada del asistente.
4. El modelo base se carga cuantizado en 4 bits con bitsandbytes.
5. PEFT agrega adapters LoRA sobre capas lineales del modelo.
6. `SFTTrainer` entrena solo esos adapters y guarda el resultado en `outputs/`.

## Dataset usado

| Split | Uso | Pares |
|---|---|---:|
| `train` | Entrenamiento | 5093 |
| `validation` | Validacion durante entrenamiento | 306 |
| `test` | Evaluacion final retenida | 328 |

La variante principal fue `sft_grounded`: el modelo recibe `Contexto fuente` y pregunta. Esto es importante porque el entrenamiento busca respuestas guiadas por evidencia, no solo memoria general del modelo.

## Modelos entrenados

| Adapter | Modelo base | Output |
|---|---|---|
| `gemma4-grounded` | `google/gemma-4-E2B-it` | `outputs/gemma4-grounded/` |
| `medgemma-grounded` | `google/medgemma-1.5-4b-it` | `outputs/medgemma-grounded/` |

Cada carpeta de salida contiene el adapter LoRA, tokenizer, configuracion y predicciones del test:

- `adapter_model.safetensors`: pesos entrenados del adapter.
- `adapter_config.json`: configuracion LoRA y modelo base asociado.
- `tokenizer.json` y `tokenizer_config.json`: tokenizer guardado.
- `test_predictions.jsonl`: predicciones generadas sobre el split de test.
- `checkpoint-*`: checkpoints intermedios del entrenamiento.

## Parametros tecnicos principales

| Parametro | Valor | Explicacion corta |
|---|---:|---|
| Metodo | QLoRA | Entrena adapters LoRA sobre modelo base cuantizado. |
| Cuantizacion | 4-bit NF4 | Reduce memoria de GPU para entrenar modelos grandes. |
| Double quantization | Activada | Mejora eficiencia de la cuantizacion. |
| LoRA rank `r` | 16 | Tamano interno del adapter. |
| `lora_alpha` | 16 | Escala de actualizacion LoRA. |
| `lora_dropout` | 0.05 | Regularizacion para evitar sobreajuste. |
| Batch por GPU | 1 | Conservador para workstation con VRAM limitada. |
| Gradient accumulation | 8 | Simula batch efectivo mayor sin usar tanta memoria. |
| Learning rate | `2e-4` | Tasa comun para LoRA/SFT. |
| Epocas | 2 | Dos pasadas sobre el dataset de entrenamiento. |
| Max length | 1024 tokens | Longitud maxima de cada ejemplo. |
| Scheduler | Cosine | Reduce gradualmente el learning rate. |
| Optimizer | `adamw_8bit` | Optimizador eficiente en memoria. |
| Gradient checkpointing | Activado | Ahorra VRAM a cambio de mas computo. |
| Packing | Desactivado | Evita mezclar muestras y contaminar ejemplos clinicos. |
| Seed | 3407 | Reproducibilidad. |

El script elige `bfloat16` si la GPU lo soporta; si no, usa `float16`.

## Resultados del entrenamiento

Ambos entrenamientos reales terminaron en 2 epocas y 1274 pasos.

| Adapter | Pasos | Eval loss final | Accuracy token final |
|---|---:|---:|---:|
| `gemma4-grounded` | 1274 | 0.7771 | 0.8017 |
| `medgemma-grounded` | 1274 | 0.8172 | 0.7944 |

Estos valores salen de `trainer_state.json` dentro de los checkpoints finales `checkpoint-1274`.

## Comando base

Ejemplo del entrenamiento real con MedGemma:

```bash
python scripts/train_qlora_trl.py \
  --model-name google/medgemma-1.5-4b-it \
  --dataset-variant sft_grounded \
  --output-dir outputs/medgemma-grounded
```

Ejemplo equivalente con Gemma 4:

```bash
python scripts/train_qlora_trl.py \
  --model-name google/gemma-4-E2B-it \
  --dataset-variant sft_grounded \
  --output-dir outputs/gemma4-grounded
```

## Guion corto para exponer

Primero, el proyecto construye un dataset clinico de obstetricia en formato pregunta-respuesta. Para el fine-tuning usamos la variante grounded, donde cada pregunta viene con contexto fuente, porque queremos que el modelo responda con evidencia y no solo con conocimiento general.

Segundo, el entrenamiento se hizo con QLoRA. Esto significa que el modelo base se carga en 4 bits para ahorrar memoria, y no se modifican todos sus parametros. En cambio, se entrenan adapters LoRA pequenos. Es una forma eficiente de adaptar modelos grandes en una GPU limitada.

Tercero, el script central es `scripts/train_qlora_trl.py`. Usa TRL para el supervised fine-tuning, PEFT para LoRA y bitsandbytes para la cuantizacion. Los parametros principales fueron 2 epocas, learning rate `2e-4`, batch 1 con acumulacion de gradiente 8, longitud maxima 1024 y LoRA con `r=16`, `alpha=16`, `dropout=0.05`.

Cuarto, se entrenaron dos adapters: uno sobre `google/gemma-4-E2B-it` y otro sobre `google/medgemma-1.5-4b-it`. Las salidas quedan en `outputs/gemma4-grounded/` y `outputs/medgemma-grounded/`, donde estan los pesos del adapter, la configuracion, tokenizer, checkpoints y predicciones del test.

Finalmente, esto no produce un modelo completo nuevo, sino adapters reutilizables que se montan encima del modelo base para inferencia y evaluacion. Esa es la ventaja tecnica: menor costo, menor memoria y una adaptacion especifica al dominio materno-clinico en espanol.

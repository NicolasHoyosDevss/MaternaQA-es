<div align="center">

<img src="./public/app-icon.png" alt="MaternaQA-es logo" width="140" height="140" />

# MaternaQA-es

**MaternaQA-es es un dataset público en español dedicado a Q+A sobre atención médica materna. Se creó para apoyar la investigación en PLN, el ajuste de modelos de lenguaje grande (LLM Fine-tuning), los sistemas de recuperación de información (RAG) y las aplicaciones de IA centradas en la atención médica para las comunidades de habla hispana.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white&labelColor=1e293b)](https://python.org)
[![Dataset](https://img.shields.io/badge/Dataset-5%2C727%20pares%20QA-EC4899?style=for-the-badge&logo=databricks&logoColor=white&labelColor=1e293b)](https://huggingface.co/datasets/JhonHander/obstetrics-qa-synthetic-es)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Datasets-FFD21E?style=for-the-badge&logo=huggingface&logoColor=white&labelColor=1e293b)](https://huggingface.co/datasets/JhonHander)
[![PDFs](https://img.shields.io/badge/Fuentes-63%20PDFs%20cl%C3%ADnicos-22D3EE?style=for-the-badge&logo=adobeacrobatreader&logoColor=white&labelColor=1e293b)](./pdfs/obstetrics)
[![Licencia](https://img.shields.io/badge/Licencia-MIT-FACC15?style=for-the-badge&logo=opensourceinitiative&logoColor=black&labelColor=1e293b)](./LICENSE)

[Dataset Q+A](#dataset-qa) · [Metodología](#metodología-de-construcción) · [Resultados](#resultados-del-dataset) · [Uso](#uso-rápido) · [Documentación](#documentación)

</div>

---

> [!IMPORTANT]
> Este repositorio documenta y conserva el procesamiento usado para construir **MaternaQA-es**, un dataset de Q+A en español para investigación en NLP clínico sobre embarazo, maternidad, parto, posparto y atención perinatal. El foco no es presentar una herramienta de software, sino dejar trazable cómo se creó, validó y publicó el dataset.

## ¿Qué es MaternaQA-es?

**MaternaQA-es** es un recurso de datos para entrenar, evaluar y analizar modelos de lenguaje en tareas de preguntas y respuestas clínicas en español. El dataset se construyó a partir de documentos clínicos extensos relacionados con salud materna y perinatal, transformados en fragmentos auditables y posteriormente en pares pregunta-respuesta sintéticos con control de calidad.

El repositorio incluye:

- Los **PDFs fuente** usados como base documental.
- Los **artefactos intermedios** de extracción, limpieza, segmentación, auditoría y evaluación.
- Las **versiones publicables** del dataset Q+A.
- Scripts reproducibles para reconstruir o extender el procesamiento.
- Documentación metodológica para paper, entrega académica y publicación del dataset.

## Dataset Q+A

El principal entregable es un dataset sintético de **5.727 pares pregunta-respuesta** en español, derivados de contenido clínico sobre embarazo y maternidad.

| Split | Pares Q+A | Chunks fuente | PDFs fuente |
|:------|----------:|--------------:|------------:|
| Entrenamiento | 5.093 | 1.744 | 52 |
| Validación | 306 | 101 | 2 |
| Test | 328 | 108 | 3 |
| **Total** | **5.727** | **1.953** | **57** |

Las variantes listas para publicación están en:

```text
datasets/obstetrics/qa/publication/
├── sft_closed_book/   # pregunta → respuesta
├── sft_grounded/      # contexto + pregunta → respuesta
└── qa_flat_jsonl/     # registros planos con metadatos
```

| Variante | Formato | Uso recomendado |
|----------|---------|-----------------|
| `sft_closed_book` | Pregunta → Respuesta | Fine-tuning sin contexto explícito; evalúa internalización del dominio. |
| `sft_grounded` | Contexto + Pregunta → Respuesta | Fine-tuning o evaluación con evidencia documental. |
| `qa_flat_jsonl` | Registro plano con metadatos | Auditoría, análisis exploratorio y documentación científica. |

[📁 Ver archivos del dataset](./datasets/obstetrics/qa/publication) · [🤗 Ver en Hugging Face](https://huggingface.co/datasets/JhonHander/obstetrics-qa-synthetic-es)

## Corpus documental base

Además del dataset Q+A, el repositorio conserva el corpus limpio usado como punto de partida para la generación de preguntas y respuestas.

| Métrica | Valor |
|---------|------:|
| PDFs procesados | 63 |
| Páginas extraídas | 5.856 |
| Páginas limpias mantenidas | 5.176 (88,4 %) |
| Chunks finales auditados | 2.268 |
| Chunks publicados como LM dataset | 2.223 |
| Tokens promedio por chunk | 879 |
| Fuga de datos entre splits | 0 |

Los chunks incluyen metadatos como `source_pdf`, `pages`, `section_type`, `content_role`, `clinical_score`, `topics` y `token_estimate`. La división train/validation/test se hizo a nivel de documento para evitar contaminación entre splits.

[📁 Ver corpus LM](./datasets/obstetrics/lm) · [🤗 Ver en Hugging Face](https://huggingface.co/datasets/JhonHander/obstetrics-lm-es)

## Metodología de construcción

La construcción de MaternaQA-es sigue una cadena reproducible de preparación y validación de datos:

1. **Curaduría de fuentes** — selección de PDFs clínicos en español relacionados con embarazo, parto, posparto, recién nacido y atención perinatal.
2. **Extracción textual** — lectura de páginas, detección de documentos que requieren OCR y preservación de metadatos de origen.
3. **Limpieza y filtrado** — descarte de páginas cortas, fragmentadas, no clínicas o dominadas por referencias.
4. **Segmentación documental** — creación de chunks clínicamente útiles con límites de longitud y metadatos trazables.
5. **Enriquecimiento temático** — asignación de puntajes clínicos y tópicos para describir la cobertura del corpus.
6. **Generación sintética de Q+A** — creación de preguntas y respuestas con salidas estructuradas de OpenAI.
7. **Control de calidad** — filtros de grounding, auditoría de contenido, evaluación con Ragas y revisión de reportes.
8. **Preparación para publicación** — exportación de variantes SFT y JSONL plano para investigación y entrenamiento.

> [!NOTE]
> La intención académica del repositorio es explicar **qué datos se construyeron, cómo se construyeron y qué garantías de calidad tienen**, no presentar el código como el resultado principal.

## Resultados del dataset

### Evaluación Ragas

Muestreo estratificado evaluado con métricas de fidelidad y relevancia.

| Split | Muestra evaluada | Fidelidad | Relevancia |
|:------|-----------------:|----------:|-----------:|
| Entrenamiento | 300 / 5.093 | **0,7726** | **0,6466** |
| Validación | 100 / 306 | **0,7826** | **0,6812** |
| Test | 100 / 328 | **0,7132** | **0,5583** |

### Grounding

- Solapamiento promedio contexto-respuesta: **0,6836**.
- Solo **27 pares** (0,54 %) quedaron marcados con grounding bajo.

### Tipos de pregunta

El dataset contiene preguntas de seis tipos para cubrir distintos niveles de complejidad:

| Tipo | Intención |
|------|-----------|
| `factual` | Recuperar información clínica puntual. |
| `definicion` | Explicar conceptos, condiciones o procedimientos. |
| `comparacion` | Diferenciar entidades clínicas o decisiones de manejo. |
| `razonamiento` | Justificar relaciones causales, riesgos o recomendaciones. |
| `aplicacion` | Aplicar conocimiento a situaciones clínicas descritas. |
| `hipotetico` | Explorar escenarios condicionales o variantes de caso. |

### Cobertura temática

Los tópicos anotados incluyen:

`prenatal_care` · `postpartum` · `preterm_labor` · `labor_induction` · `vaginal_birth` · `cesarean` · `hemorrhage` · `preeclampsia` · `diabetes_gestational` · `infection` · `fetal_monitoring` · `newborn_care` · `ultrasound` · `genetics` · `contraception` · `infertility` · `menopause` · `gynecologic_oncology`

## Uso rápido

### 1. Instalar dependencias

```bash
git clone https://github.com/NicolasHoyosDevss/Fine-Tunning-Benchmark.git MaternaQA-es
cd MaternaQA-es
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> [!NOTE]
> Para regenerar Q+A se requiere `OPENAI_API_KEY`. Para publicar o descargar recursos privados en Hugging Face se requiere `HF_TOKEN`.

### 2. Cargar una variante del dataset

```python
from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files={
        "train": "datasets/obstetrics/qa/publication/sft_grounded/train.jsonl",
        "validation": "datasets/obstetrics/qa/publication/sft_grounded/validation.jsonl",
        "test": "datasets/obstetrics/qa/publication/sft_grounded/test.jsonl",
    },
)

train = dataset["train"]
validation = dataset["validation"]
test = dataset["test"]
```

### 3. Generar Q+A desde un split del corpus

```bash
python scripts/generate_synthetic_qa.py \
  --input datasets/obstetrics/lm/train_lm.jsonl \
  --raw-output datasets/obstetrics/qa/final/train/raw.jsonl \
  --sft-output datasets/obstetrics/qa/final/train/sft.jsonl \
  --progress-file datasets/obstetrics/qa/final/train/progress.json \
  --report-output datasets/obstetrics/qa/final/train/generation_report.json \
  --model gpt-5.4-mini \
  --min-pairs 2 \
  --max-pairs 5
```

### 4. Entrenamiento experimental opcional

El repositorio también incluye un script para smoke tests de fine-tuning con QLoRA sobre las variantes publicadas.

```bash
python scripts/train_qlora_trl.py \
  --model-name google/gemma-4-E2B-it \
  --dataset-variant sft_grounded \
  --output-dir outputs/smoke-test \
  --max-steps 10 \
  --train-limit 64 \
  --eval-limit 32
```

## Estructura del repositorio

```text
MaternaQA-es/
├── pdfs/obstetrics/              # documentos clínicos fuente
├── artifacts/obstetrics/         # artefactos intermedios y reportes
│   ├── corpus/
│   ├── metadata/
│   ├── reports/
│   ├── tables/
│   └── qa_experiments/
├── datasets/obstetrics/
│   ├── lm/                       # corpus limpio segmentado
│   └── qa/
│       ├── final/                # Q+A por split antes de publicación
│       └── publication/          # variantes finales del dataset
├── scripts/                      # procesamiento, evaluación y entrenamiento experimental
├── docs/                         # metodología y notas técnicas
├── papers/                       # planeación del paper de dataset
├── public/app-icon.png
└── requirements.txt
```

## Scripts principales

| Script | Propósito |
|--------|-----------|
| `extract_pdfs.py` | Extrae texto y metadatos desde los PDFs fuente. |
| `clean_text.py` | Limpia páginas y descarta contenido no útil. |
| `build_lm_dataset.py` | Segmenta documentos y construye el corpus limpio. |
| `audit_dataset.py` | Genera reportes de calidad, duplicados y splits. |
| `generate_synthetic_qa.py` | Genera pares pregunta-respuesta con salidas estructuradas. |
| `evaluate_qa_with_ragas.py` | Evalúa muestras Q+A con métricas Ragas. |
| `prepare_qa_publication_variants.py` | Exporta variantes listas para publicación. |
| `train_qlora_trl.py` | Ejecuta fine-tuning experimental con TRL, PEFT y QLoRA. |

## Documentación

| Documento | Descripción |
|-----------|-------------|
| [Metodología del artículo](./docs/article_methodology.md) | Descripción académica de la construcción del dataset. |
| [Flujo de generación y evaluación Q+A](./docs/qa_dataset_evaluation_workflow.md) | Detalles sobre generación, evaluación y variantes del dataset. |
| [README del dataset publicado](./datasets/obstetrics/qa/publication/README.md) | Esquema de archivos, campos y usos recomendados. |
| [Planeación del paper](./papers/README.md) | Posicionamiento, contribuciones, limitaciones y estrategia de escritura. |
| [Notas de investigación](./docs/research_notes/) | Decisiones técnicas y seguimiento del trabajo. |

## Consideraciones éticas y de uso

- MaternaQA-es es un recurso de investigación; no reemplaza criterio clínico ni guías médicas oficiales.
- Las respuestas son sintéticas y deben interpretarse como datos para entrenamiento/evaluación, no como recomendaciones médicas directas.
- La trazabilidad por `source_pdf`, páginas y chunks permite auditar el origen documental de cada muestra.
- Antes de usar el dataset en sistemas reales, se requiere validación clínica independiente.

---

<div align="center">

**MinCiencias** · Ministerio de Ciencia, Tecnología e Innovación de Colombia

Construido para investigación en NLP clínico en español sobre embarazo y maternidad.

</div>

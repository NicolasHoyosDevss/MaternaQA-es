# Metodología del Dataset de Obstetricia para Fine-Tuning

Este documento resume la metodología del repositorio en un formato útil para redactar el artículo del proyecto. La intención es explicar cómo se construyó un dataset clínico de obstetricia trazable, auditable y apto para entrenamiento supervisado de modelos de lenguaje.

## 1. Propósito

El objetivo fue construir un dataset en español para adaptar modelos de lenguaje al dominio de obstetricia y ginecología. El enfoque priorizó tres criterios:

- trazabilidad entre cada ejemplo y su documento fuente;
- calidad clínica y redacción autocontenida;
- separación clara entre construcción del dataset, evaluación automática y revisión humana.

El resultado principal es un conjunto de pares pregunta-respuesta derivados de literatura clínica, guías, manuales, protocolos y artículos científicos, acompañado por variantes listas para fine-tuning supervisado.

## 2. Fuentes

Las fuentes primarias se almacenan en `pdfs/obstetrics/`. El corpus incluye documentos heterogéneos del dominio obstétrico:

- guías de práctica clínica;
- protocolos de atención;
- manuales y libros;
- capítulos académicos;
- artículos científicos;
- documentos institucionales de apoyo clínico.

Cada PDF se registra en un manifiesto documental con metadatos técnicos y señales de inclusión: nombre de archivo, ruta, tamaño, número de páginas, tipo documental, estado de inclusión, idioma, páginas con posible OCR pendiente y razón de exclusión cuando aplica.

## 3. Extracción de Texto

La extracción opera a nivel de página con `scripts/extract_pdfs.py`.

El extractor principal es `PyMuPDF`, porque ofrece una lectura rápida y estable de bloques de texto. Cuando una página produce poco contenido, `pdfplumber` actúa como respaldo para recuperar texto con layout. Las páginas con muy pocos caracteres después de ambos métodos se marcan como `needs_ocr`, pero no se les aplica OCR automático dentro del pipeline actual.

Las salidas principales son:

```text
artifacts/obstetrics/corpus/raw_pages.jsonl
artifacts/obstetrics/metadata/inventory.json
```

## 4. Limpieza de Artículos Científicos y Documentos Clínicos

La limpieza se realiza con `scripts/clean_text.py`. El objetivo no es normalizar agresivamente el contenido médico, sino remover ruido documental que afectaría el entrenamiento.

El pipeline elimina o marca:

- encabezados y pies repetidos;
- números de página y etiquetas administrativas;
- índices y tablas de contenido;
- secciones dominadas por referencias bibliográficas;
- texto fragmentado por mala extracción;
- páginas demasiado cortas;
- páginas que requieren OCR;
- documentos excluidos por manifiesto.

Cada página conserva información de auditoría: `is_kept`, `drop_reason`, sección detectada, método de extracción, tipo documental y métricas básicas.

## 5. Chunking y Corpus LM

Con `scripts/build_lm_dataset.py`, las páginas limpias se agrupan en chunks con tamaño controlado y solapamiento. El chunking busca equilibrar contexto suficiente para generar QA clínico con unidades manejables para entrenamiento y evaluación.

Cada chunk incluye:

- `chunk_id` estable;
- PDF fuente y páginas;
- texto limpio;
- tipo documental;
- sección aproximada;
- rol de contenido;
- temas clínicos inferidos;
- estimación de tokens;
- puntaje clínico.

Los splits de LM se hacen a nivel de documento, no de página. Esto reduce el riesgo de que contenido casi idéntico del mismo PDF aparezca en entrenamiento y evaluación.

## 6. Generación de Preguntas y Respuestas

La generación de QA se ejecuta con `scripts/generate_synthetic_qa.py` a partir de chunks aceptados. El prompt obliga a producir preguntas autocontenidas y respuestas clínicas en español, evitando expresiones como "según el texto" o "de acuerdo con el fragmento".

Para cada chunk, el generador puede crear preguntas de distintos tipos:

- factual;
- definición;
- comparación;
- razonamiento;
- aplicación clínica;
- hipotético.

Cada par conserva un `contexto_fuente` breve que respalda directamente la respuesta. Este campo es clave para auditoría, evaluación grounded y publicación en formato plano.

## 7. Control Operativo con LLM-as-a-Judge

Durante la generación y evaluación se usan jueces LLM para estimar:

- fidelidad al contexto;
- relevancia de la respuesta;
- consistencia roundtrip;
- veredicto operativo de aceptación o rechazo;
- razón corta del juicio.

Esta capa es útil para filtrar y diagnosticar errores, pero no se considera evaluación formal suficiente por sí sola. Su función principal es mejorar la calidad del artefacto antes de la evaluación reportable.

## 8. Evaluación Formal con Ragas

La evaluación formal se ejecuta con `scripts/evaluate_qa_with_ragas.py` sobre los archivos `raw.jsonl` de cada split.

Las métricas reportables son:

- `faithfulness`: qué tan respaldada está la respuesta por el contexto;
- `answer_relevancy`: qué tan bien responde la pregunta.

No se usa `context_precision` como métrica central porque el experimento no evalúa un sistema de retrieval con múltiples contextos candidatos. Cada QA se construye desde un contexto fuente conocido.

Para controlar costo y sesgo por documentos largos, la evaluación puede hacerse con muestreo estratificado por PDF y semilla fija.

## 9. Variantes Publicables

`scripts/prepare_qa_publication_variants.py` produce variantes limpias desde `datasets/obstetrics/qa/final/`:

| Variante | Uso |
|---|---|
| `sft_closed_book` | Entrenar y evaluar adaptación paramétrica sin contexto explícito. |
| `sft_grounded` | Entrenar respuestas guiadas por evidencia con contexto fuente. |
| `qa_flat_jsonl` | Auditar, explorar y publicar registros planos. |

Los campos vacíos o internos de evaluación se remueven en las variantes publicables para separar datos finales de diagnósticos de construcción.

## 10. Entrenamiento Esperado

El repositorio incluye `scripts/train_qlora_trl.py` para ejecutar fine-tuning supervisado con TRL, PEFT y QLoRA. La ruta esperada usa:

- `train.jsonl` para ajuste;
- `validation.jsonl` para selección y seguimiento;
- `test.jsonl` como evaluación final retenida.

La variante `sft_grounded` es la más alineada con asistentes clínicos que deben responder usando evidencia provista. La variante `sft_closed_book` permite medir cuánto conocimiento del dominio queda internalizado por el modelo.

## 11. Limitaciones y Controles Pendientes

El pipeline ya registra páginas con necesidad de OCR, pero no ejecuta OCR real. También conviene complementar las métricas automáticas con revisión clínica humana, especialmente sobre:

- preguntas con puntajes bajos;
- respuestas largas;
- temas de alto riesgo obstétrico;
- documentos con extracción irregular;
- ejemplos procedentes de tablas complejas.

El dataset debe tratarse como artefacto de investigación. Cualquier modelo entrenado con él requiere validación clínica antes de uso asistencial.

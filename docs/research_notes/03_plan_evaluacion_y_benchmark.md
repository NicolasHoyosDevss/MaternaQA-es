# Plan de evaluación y benchmark propio

Como no existe un benchmark público perfectamente alineado con **obstetricia en español + estilo Q+A del proyecto**, la evaluación debe combinar un **benchmark interno** con auditorías automáticas.

## Propuesta de benchmark v1

| Componente | Recomendación |
|---|---|
| Tamaño inicial | 200–400 preguntas |
| Fuente | Documentos retenidos, pero separados del entrenamiento |
| Formato | Pregunta, respuesta de referencia, chunk fuente, tipo, dificultad |
| Tipos | factual, definición, razonamiento, aplicación clínica, comparación, seguridad |
| Dificultad | básico, intermedio, avanzado |
| Revisión | Idealmente por al menos una persona con criterio clínico |

## Regla crítica de separación

El benchmark no debe salir de los mismos documentos usados para entrenar si se quiere medir generalización documental. La opción preferida es reservar PDFs completos para benchmark/evaluación.

En el pipeline automático conviene separar **train / validation / test** por documento completo y aproximar el reparto por **volumen de chunks**, no por número de PDFs. En corpus pequeños, una validación del 5% puede ser razonable para no quitar demasiado entrenamiento, siempre que el benchmark final se construya aparte con suficientes ejemplos y cobertura clínica.

## Métricas recomendadas

| Métrica | Qué responde |
|---|---|
| Exactitud clínica | ¿La respuesta es correcta? |
| Groundedness | ¿Está sustentada por la fuente? |
| Completitud | ¿Responde lo importante sin omitir riesgos? |
| Seguridad | ¿Evita recomendaciones peligrosas o sobreconfianza? |
| Utilidad | ¿Ayuda a un usuario clínico o académico? |

## Diseño experimental mínimo

1. Modelo base instruct sin fine-tuning.
2. Modelo ajustado con SFT Q+A del proyecto.
3. Evaluación ciega sobre el benchmark interno.
4. Comparación por tipo de pregunta y dificultad.
5. Reporte de errores cualitativos: omisión, alucinación, sobre-generalización, lenguaje no seguro.

## Benchmarks externos complementarios

Para el paper, el benchmark propio debe seguir siendo la evaluación principal de obstetricia en español. Como evaluación complementaria se pueden usar:

- **CareQA** y **HEAD-QA / HEAD-QA v2** para razonamiento médico en español;
- subconjuntos de **obstetricia y ginecología** de benchmarks más amplios cuando exista etiquetado por materia;
- esos benchmarks externos deben mantenerse fuera del entrenamiento si se quieren reportar como evaluación independiente.

## Cuándo agregar más datos

Agregar PDFs adicionales cuando aporten al menos una de estas cosas:

- nuevos temas clínicos;
- mejor cobertura de guías/protocolos;
- representación de subdominios poco cubiertos;
- documentos más recientes o de mayor autoridad.

No agregar más datos solo para aumentar conteos.

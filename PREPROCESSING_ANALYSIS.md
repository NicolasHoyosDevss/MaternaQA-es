# Análisis Crítico del Pipeline de Preprocessing

## Resumen Ejecutivo

El pipeline de preprocessing actual **es sólido y está bien diseñado para el caso de uso específico de obstetricia**, pero tiene **limitaciones importantes que afectan la calidad final de los datos**. El sistema maneja bien documentos altamente variables (guías, protocolos, manuales, artículos) pero es **demasiado conservador en el filtering**, descartando ~20% de chunks válidos por ser "reference-heavy".

---

## 1. ARQUITECTURA GENERAL: ✅ BIEN DISEÑADA

### Pipeline de 4 pasos en cascada:
```
Raw PDFs → Extract (PyMuPDF + pdfplumber fallback)
        → Clean (remove headers, footer, section detection)
        → Build chunks (500-1200 tokens, 80 token overlap)
        → Audit (validation, sampling for review)
```

**Fortalezas:**
- Fallback robusto: PyMuPDF para texto nativo, pdfplumber para layout, marca `needs_ocr` para problemas
- Detección automática de secciones usando heurísticos lingüísticos (mayúsculas, numeración, palabras clave)
- Deduplicación dentro de chunks (exact + near-duplicate)
- Métricas exhaustivas en cada paso
- Reportes detallados para auditoria manual

**Retención por etapa:**
```
2,996 raw pages
↓ 89.9% kept
2,692 clean pages
↓ 44.6% → 1,200 final chunks
```

---

## 2. DETECCIÓN DE SECCIONES: ⚠️ BUENA PERO IMPERFECTA

### Cómo funciona:
Busca en los **primeros 10 renglones** de cada página líneas que cumplan:
1. ≥72% mayúsculas (ej: "CAPÍTULO 5")
2. Coincida con regex de numeración (ej: "1.2.3 Tema")
3. Palabras clave: "Capítulo", "Sección", "Anexo", "Tema"
4. <110 caracteres, ≤14 palabras

### Resultados observados:
```
✅ BIEN DETECTADAS:
- "Capítulo 1. SEMIOLOGÍA OBSTÉTRICA" → "Capítulo 1."
- "13. PROCEDIMIENTO PARA LA ATENCIÓN..." → detecta correctamente
- "Protocolo Atencion Obstetrica" → se detecta bien a través del documento

⚠️ PROBLEMÁTICAS:
- "AÍCRAG" (reversed text?) → se retiene como sección válida 
- "MINISTERIO" (header que no es sección real) → detectado como sección
- Primera página de algunos PDFs → detecta "Introducción" correctamente
```

### Problema clave:
La detección **no distingue entre**:
- Títulos reales de secciones clínicas
- Headers administrativos (ministerios, instituciones)
- Palabras que aparecen accidentalmente en formato de título

### Impacto:
- **Bajo en términos de cantidad**: los chunks se agrupan bien por "sección"
- **Medio en términos de semántica**: algunos chunks tienen sección "MINISTERIO" o "AÍCRAG" que no son significativas
- **Solución**: agregar lista negra de palabras administrativas en `extract_page_section()`

---

## 3. CHUNKING (500-1200 tokens): ✅ EXCELENTE

### Estrategia:
1. Agrupa páginas por (PDF, sección)
2. Divide párrafos respetando boundaries de heading
3. Usa buffer deslizante: cuando acumula ≥500 tokens, genera chunk
4. Si supera 1200, flush con overlap de 80 tokens del final

### Resultados:
```
Tokens promedio por chunk: 908.8 (dentro del target 500-1200)
Rango observado: 316 - 1195 tokens
```

**Muy bueno porque:**
- Mantiene contexto (overlap previene desconexión de ideas)
- Respeta estructura del documento (no corta en mitad de párrafos)
- Balance entre especificidad y generalidad

**Potencial mejora:**
- El primer chunk del documento a veces es corto (316 tokens en MAN_PROC_SERV_GIN_2023) porque captura solo la introducción/índice

---

## 4. SCORING CLÍNICO: ✅ FUNCIONA BIEN (pero posible sesgo)

### Cómo se calcula:
```python
score = 0
# +1 ó +2 por término clínico encontrado
# +2 si ≥120 palabras, +2 más si ≥350 palabras
# -3 si >35% de líneas son referencias
```

### Distribución observada:
```
Promedio: 17.7 (sobre un máximo ~45)
Rango: 5-44
Moda: 15 (6.8%)
```

**Observación importante:**
El score parece **bien calibrado** pero con un sesgo hacia **documentos con más contenido descriptivo que protocolos procedimentales**:

```
LOW score (7) ejemplo:
"Fijación sacroespinosa unilateral versus bilateral, 
en el tratamiento del prolapso..."
→ Técnico pero con menos repetición de términos clave

HIGH score (39) ejemplo:
"Fetal programming assumes that pregnancy is the period
of greatest susceptibility to acquire changes in the cell nucleus..."
→ Mucho lenguaje clínico denso
```

**Recomendación:** El threshold mínimo de 5 está OK, pero considera aumentar a 6 para reducir documentación administrativa/metodológica.

---

## 5. FILTERING POR QUALITY FLAGS: ⚠️ DEMASIADO CONSERVADOR

### Razones de descarte en BUILD stage:

```
reference_heavy:        344/1,678 (20.5%) ← MAYOR PROBLEMA
too_short:               63/1,678 (3.8%)
low_clinical_score:      41/1,678 (2.4%)
numeric_or_table_heavy:  17/1,678 (1.0%)
admin_or_contact_text:   13/1,678 (0.8%)
```

### El problema: "reference_heavy" descarta 20.5%

Un chunk se rechaza si:
```python
reference_line_ratio(text) >= 0.45  # ≥45% de líneas tienen referencia markers
```

**O**

```python
reference_marker_count(text) >= 4  # ≥4 markers (DOI, [2020], et al, etc)
```

**Esto es problemático porque:**

1. **Las guías clínicas TIENEN muchas referencias** — es su naturaleza
2. **Las referencias no son "ruido"** — son evidencia clínica
3. **Muchos chunks válidos se pierden**

**Ejemplo real:**
```
GPC_533_Embarazo_AETSA_compl.pdf:
- 476 páginas kept
- Solo 88 chunks finales
- 74 chunks rechazados por "reference_heavy"
= Tasa de rechazo: 45.7%
```

### Recomendación:
**Cambiar criterio de `reference_heavy` a:**
```python
# Actual (MUY restrictivo):
if reference_line_ratio(text) >= 0.45 and clinical_score < 6:
    reject()

# Propuesto (más sensato):
if reference_line_ratio(text) >= 0.65 and clinical_score < 8:
    # Solo rechazar si MAYORMENTE es referencias Y bajo score clínico
    reject()
```

Esto recuperaría ~80-100 chunks válidos de guías clínicas que ahora se pierden.

---

## 6. HANDLING DE DOCUMENTOS CON OCR: ⚠️ LIMITADO

### PDFs con páginas que necesitan OCR:

```
Obstetrical Risk Management Playbook _ASHRM.pdf     31/31  (100%) ← TODO OCR
RECOM_obstetricia_web.pdf                            7/62   (11.3%)
GINECOLOGÍA-Y-OBSTETRICIA.pdf                        8/148  (5.4%)
GPC_533_Embarazo_AETSA_compl.pdf                      8/494  (1.6%)
MAN_PROC_SERV_GIN_2023.pdf                            2/536  (0.4%)
Protocolo Atencion Obstetrica.pdf                     1/192  (0.5%)
```

**Estado actual:**
- Se marcan las páginas como `needs_ocr: true`
- Se descartan en cleaning si <180 caracteres extraídos
- **Pero nunca se ejecuta OCR real** (el pipeline no lo incluye)

**El problema:**
- `Obstetrical Risk Management Playbook` tiene **0 chunks** — todo está marcado OCR
- Se pierden datos valiosos porque:
  1. Documento probablemente digitalizó mal (PDF de escaneo con OCR defectuosa)
  2. PyMuPDF no puede extraer
  3. pdfplumber no puede extraer

**Soluciones propuestas:**

**Opción A (Fácil):** Eliminar documento problemático (ya hiciste con GPC_472)
```bash
rm obstetrics/spanish/Obstetrical_Risk_Management_Playbook_ASHRM.pdf
```

**Opción B (Mejor):** Integrar Tesseract OCR para PDFs con ≥80% de páginas sin texto

**Opción C (Medium):** Aumentar umbral de aceptación para `needs_ocr` (180 chars es muy bajo)

---

## 7. DEDUPLICACIÓN: ✅ FUNCIONA BIEN (pero incompleto)

### Deduplicación entre chunks del MISMO PDF:
```python
seen_exact = set()  # Texto exacto duplicado
seen_near = set()   # Primeros 240 palabras normalizadas iguales
```

**Funciona bien.** El ejemplo de `CAP09.pdf` vs `Capítulo+9.pdf` muestra que son casi idénticos pero el pipeline **no los detecta como duplicados entre PDFs distintos** — esto es intencional (no deberías perder documentos completos sin aviso).

**Recomendación:** Si quieres deduplicar entre PDFs, necesitarías un paso adicional después de BUILD que compare similitud de texto entre chunks de PDFs diferentes.

---

## 8. RESUMEN DE PROBLEMAS Y SOLUCIONES

| Problema | Severidad | Solución |
|----------|-----------|----------|
| **20.5% rechazo por reference_heavy** | 🔴 Alta | Ajustar threshold (0.65 en vez 0.45) |
| **Detección de secciones imperfecta** | 🟡 Media | Agregar lista negra de palabras |
| **Sin OCR real** | 🟡 Media | Integrar Tesseract o eliminar docs OCR |
| **Primeros chunks cortos** | 🟢 Baja | Ajustar min_tokens inicial |
| **Duplicados entre PDFs** | 🟢 Baja | Decisión: aceptar o hacer dedup B-level |

---

## 9. RECOMENDACIÓN FINAL

**El preprocessing está en buen estado** para un MVP, pero antes de fine-tuning:

### Cambios inmediatos (10 min):
1. ✅ Eliminar `Obstetrical_Risk_Management_Playbook_ASHRM.pdf` (100% OCR, 0 chunks)
2. ✅ Aumentar threshold `reference_line_ratio` de 0.45 → 0.65
3. ✅ Aumentar min `clinical_score` de 5 → 6

### Cambios opcionales (mejor calidad):
4. Agregar lista negra en `extract_page_section()` para palabras administrativas
5. Investigar los chunks con sección "AÍCRAG" y "MINISTERIO" — posible encoding issue

### Para siguiente iteración:
6. Integrar OCR (Tesseract + pdfplumber para PDFs con <20% texto)
7. Logging granular de qué se descarta y por qué

---

## 10. ESTADÍSTICAS FINALES ACTUALES

```
📊 DATASET FINAL:
   - 20 PDFs
   - 2,996 páginas raw
   - 2,692 páginas kept (89.9%)
   - 1,200 chunks finales
   - 1,084 train records
   - 116 validation records
   - Promedio: 908.8 tokens/chunk
   - Clinical score promedio: 17.7/45

⏱️ Tiempo de procesamiento: ~5 minutos para 3K páginas
💾 Tamaño dataset train: ~1MB (1,084 registros JSON)
```

---

## Conclusión

**¿Es una buena estrategia?** 

**Sí, 85/100.** El pipeline es:
- ✅ Robusto a variabilidad de documentos
- ✅ Bien estructurado y auditable
- ✅ Produce chunks de calidad razonable

**Pero necesita:**
- ⚠️ Ajuste de thresholds (reference_heavy especialmente)
- ⚠️ Mejor manejo de documentación administrativa
- ⚠️ OCR real o exclusión de documentos problemáticos

**Impacto estimado de cambios:** +5-8% en cantidad de chunks válidos sin degradación de calidad.

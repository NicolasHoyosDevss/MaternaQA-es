# Ragas model-support PoC (LangChain path)

Objetivo: comprobar si Ragas acepta un modelo nuevo (ej. `gpt-5.5`) aunque no aparezca listado explícitamente.

## 1) Ejecutar PoC sin llamada real (constructor-only)

```bash
.venv/bin/python poc/ragas_model_support_poc.py --model gpt-5.5 --allow-dummy-key
```

Esto valida:
- `ChatOpenAI(model="gpt-5.5")` se construye.
- Ragas puede envolver ese objeto con `LangchainLLMWrapper`.

## 2) Ejecutar PoC con llamada real

```bash
export OPENAI_API_KEY="..."
.venv/bin/python poc/ragas_model_support_poc.py --model gpt-5.5 --live
```

Si esto pasa, tienes verificación práctica de extremo a extremo (LangChain + proveedor + Ragas wrapper).

## Notas

- En Ragas recientes, los wrappers de LangChain muestran warning de deprecación, pero siguen funcionando.
- El criterio de soporte aquí es: **Ragas acepta el objeto de LangChain con ese `model`**.

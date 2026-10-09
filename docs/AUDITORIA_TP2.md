# Auditoría del TP2 — informe final

> Fecha: 2026-10-07. Rama: `borrar`. Consigna: "TP N.º 2 — Representación vectorial de
> texto: embeddings y búsqueda semántica", con el alcance que la cátedra confirmó por correo
> (sin Postgres/Supabase; parte avanzada y Partes E/F fuera).
>
> Niveles de evidencia: **[EJECUTADO]** lo corrí y vi el resultado · **[INSPECCIONADO]**
> leí el código, no lo ejecuté · **[NO PROBADO]** con el motivo.
>
> Entornos usados: Windows 11 con Python 3.13 (Smart App Control bloquea scikit-learn
> `metrics`, spaCy y sentence-transformers) y un contenedor Docker `python:3.12-slim` con
> `requirements.txt` exacto (Python 3.12, como Colab). **El notebook no se ejecutó en Colab
> todavía.**

## 1. Veredicto

⚠️ **Casi listo, pero no se puede entregar todavía.** El pipeline está completo, probado y
ejecutado de punta a punta. Faltan tres cosas que dependen del grupo:

1. **Validar las consultas.** Hoy la evaluación usa `queries_propuesta.json`, una propuesta
   hecha por IA y marcada como PROVISIONAL. La consigna dice que es "la única parte que no
   puede automatizarse".
2. **Escribir las conclusiones** (sección 7 del notebook) y **cerrar el informe** con los
   números de la corrida definitiva.
3. **Ejecutar en Colab.** Los cambios ya están subidos a la rama `borrar`, que es la que
   clona el notebook. Falta la prueba en Colab.

> **Para quien continúe:** los pasos concretos están en
> [`TRASPASO.md`, "Continuar el TP2"](../TRASPASO.md#continuar-el-tp2-qué-falta-y-cómo-hacerlo).

## 2. Lo que ya estaba bien

- **Las cuatro preguntas sobre TF-IDF**, con buen diseño experimental:
  - el `fit` sobre el fondo y el `transform` sobre sondas fijas;
  - la comparación contra un clasificador trivial;
  - el control positivo del detector de idioma.

  P3 reproduce exactamente los números documentados. **[EJECUTADO]**
- **La separación crudo/tokenizado** ("el preprocesamiento pertenece al modelo"), muy bien
  documentada.
- **Word2Vec/FastText propio contra SBW**: parámetros justificados, vecinos de 8 palabras y
  la medición del "parentesco formal". TF-IDF y SBW reproducen exactamente el MRR documentado
  (0,881 y 0,578). **[EJECUTADO]**
- **El scraper y sus tests** (`test_parsers.py`: todo OK en Windows y Linux). **[EJECUTADO]**

## 3. Lo que corregí (y verifiqué después)

| Prioridad | Problema | Archivo | Corrección | Verificación |
|---|---|---|---|---|
| 🔴 | La conclusión de idioma era **falsa**: "las 200 son castellano". *Morning Star* (124205) está en gallego y `lingua` la etiqueta castellano con 0,998 | `experimentos.py` | Segundo instrumento: marcadores léxicos del gallego. La conclusión ahora sale de los datos | Lo detecta. Sus 2 términos TF-IDF top son *dun* y *dunha* **[EJECUTADO]** |
| 🔴 | `.env` versionado aunque esté en `.gitignore`: riesgo de publicar la contraseña de Supabase | `.env` | `git rm --cached .env`. El archivo local queda | `git check-ignore` ahora lo cubre. Tenía solo valores por defecto: **no hubo secretos expuestos** **[EJECUTADO]** |
| 🟠 | La ablación promocional no quitaba los términos con tilde (11 de 37; 24 % de las apariciones). El "+0,005" publicado era incorrecto y venía de un solo split | `experimentos.py` | Se quitan las formas reales del vocabulario, y se agregan 20 splits con la diferencia pareada | −0,000 ± 0,009. La conclusión cualitativa se mantiene **[EJECUTADO]** |
| 🟠 | Entrenamiento no reproducible (`workers=4`): la conclusión sobre ponderar por IDF dependía de la corrida | `embeddings.py` | `workers=1` | Mismo SHA-256 en dos procesos distintos. Ojo: el efecto IDF en Word2Vec pasó de "no cambia" a "mejora" **[EJECUTADO]** |
| 🟠 | Sin camino desde el CSV: sin Postgres no corría nada | `corpus.py` | `fuente="csv"` por defecto, con la misma semántica que el ETL (NULL, jsonb) y import diferido de psycopg | 200 docs, tipos y filtros testeados **[EJECUTADO]** |
| 🟠 | `vector_promedio` sin manejo explícito de nulos, sin normalizar y con dtype inconsistente | `embeddings.py`, nuevo `vectores.py` | Nulo explícito y documentado; validación de NaN/Inf/dimensión y normalización L2 sin dividir por cero | Tests adversariales **[EJECUTADO]** |
| 🟡 | Estabilidad: marcaba "estable" el tramo inicial plano (piso de ruido) y, en una corrección intermedia mía, el último punto por vacuidad | `experimentos.py` | Estable solo si todos los saltos desde ese punto están bajo el umbral | Ningún punto marcado; coincide con la conclusión **[EJECUTADO]** |
| 🟡 | Faltaba `lingua` en `requirements.txt` | `requirements.txt` | Agregado, más sentence-transformers y matplotlib, con las versiones probadas | `pip check` sin conflictos **[EJECUTADO]** |
| 🟡 | spaCy bloqueado en Windows solo para leer una lista de stopwords | `preprocesamiento.py` | Respaldo que lee la misma lista sin cargar las DLL | Tests en Windows OK **[EJECUTADO]** |
| 🟡 | Documentación con cifras viejas: 255k tokens, 5000×, MRR, IDF, idioma, promocional, `.env`, ramas | README, CONTEXTO, TRASPASO | Actualizadas con los números de esta corrida | **[INSPECCIONADO]** |

## 4. Lo que implementé

| Requisito de la consigna | Dónde | Estado |
|---|---|---|
| Corpus desde el CSV, versión cruda y tokenizada, decisiones documentadas con ejemplos | notebook §1 | ✅ **[EJECUTADO]** |
| Word2Vec propio contra SBW-vectors-300-min5, vecinos de ≥4 palabras lado a lado, parámetros justificados, negative sampling | notebook §3 | ✅ **[EJECUTADO]** |
| Vector de documento por promedio + qué se pierde, **demostrado** (orden: 1,000; negación: 0,986) | notebook §3 | ✅ **[EJECUTADO]** |
| SBERT `distiluse`: dimensión 512, límite 128 tokens, **172 de 200 truncados** contados con el tokenizador (contando palabras daría 118) | `busqueda.py`, notebook §4 | ✅ **[EJECUTADO]** |
| Variante SBERT por fragmentos (control del truncamiento) | `busqueda.py` | ✅ extra |
| Vectores normalizados, con manejo explícito de nulos y NaN (0 NaN, 0 Inf, 0 nulos en las 7 representaciones) | `vectores.py`, notebook §4 | ✅ **[EJECUTADO]** |
| Rankings lado a lado de 3 consultas | notebook §5 | ✅ **[EJECUTADO]** |
| Distribución de similitudes entre pares al azar (media, desvío, rango, p5, p95) | notebook §5 | ✅ **[EJECUTADO]** |
| Proyección 2D: PCA con varianza de PC1+PC2 (2 % a 26 %) y t-SNE (perplexity 30), con advertencia | notebook §5 | ✅ **[EJECUTADO]** |
| `queries.json`: ≥10 consultas y relevancia manual | `queries_propuesta.json` (16) | ⚠️ **provisional, falta validar** |
| Consulta sin solapamiento léxico, **verificada matemáticamente** (3 consultas, 0 palabras en común, puntaje TF-IDF de los relevantes = 0) | notebook §6, test | ✅ **[EJECUTADO]** |
| precision@k (TF-IDF, cada modelo, azar), con empates exactos, techo, R-precision, bootstrap y comparación pareada | `evaluacion.py`, notebook §6 | ✅ **[EJECUTADO]**, números provisionales |
| Caso concreto de fallo, con diagnóstico | notebook §6 | ✅ datos; ⚠️ la hipótesis va en el informe |
| Embeddings guardados en disco | `data/embeddings/` | ✅ **[EJECUTADO]** |
| Credenciales fuera del notebook | — | ✅ no usa ninguna **[EJECUTADO]** (búsqueda de patrones) |
| Notebook ejecutado con salidas | `TP2_Beltramo_Cortinas_Cura_Maragliano.ipynb` | ✅ de punta a punta en kernel limpio, ~130 s, sin errores, con una celda final que verifica 15 etapas **[EJECUTADO en Docker; NO PROBADO en Colab]** |
| Herramientas para regenerar y ejecutar el notebook | `tools/generar_notebook.py`, `tools/ejecutar_notebook.py` | ✅ **[EJECUTADO]** |
| Informe ≤3 páginas con los 4 puntos obligatorios + declaración de IA | `docs/informe_borrador.md` | ⚠️ borrador con números provisionales; falta el PDF |

## 5. Tests

| Test | Resultado | Observaciones |
|---|---|---|
| `tests/test_tp2.py` en Linux (runner propio) | **35/35 OK** | incluye SBERT real |
| `pytest tests/test_tp2.py` en Linux | **35 passed** | |
| `tests/test_tp2.py` en Windows | **34 OK, 1 salteado** | SBERT no importa por Smart App Control: salteado, no aprobado |
| `tests/test_parsers.py` en Windows y Linux | **TODO OK** | |
| `pytest tests/` | ⚠️ error de recolección | `test_parsers.py` hace `sys.exit()` al importarse. No es un bug nuevo; documentado en el README |

Cubren:

- **Adversariales:**
  - NaN, Inf y dimensión incorrecta;
  - vectores nulos;
  - consulta nula (empate total);
  - k=0 y k mayor que N;
  - puntajes NaN;
  - largos distintos;
  - empate total (un `argsort` ingenuo daría P@5 = 1,0 donde corresponde 0,03);
  - empate parcial exacto;
  - invariancia al orden de los documentos.
- **Consultas:**
  - ids inexistentes;
  - libro relevante y dudoso a la vez;
  - menos de 10 consultas;
  - consulta sin relevantes;
  - ninguna consulta sin solapamiento.
- **Texto:**
  - texto vacío, solo stopwords o solo símbolos;
  - consulta vacía, con caracteres raros o en otro idioma;
  - palabras fuera de vocabulario.
- **Representaciones:**
  - reproducibilidad de Word2Vec;
  - truncamiento medido en tokens y no en caracteres.

## 6. Resultados actuales (PROVISIONALES)

| Representación | P@5 | IC95 | dif. pareada vs TF-IDF |
|---|---|---|---|
| SBERT (truncado) | 0,363 | [0,225; 0,513] | +0,185 [+0,061; +0,335] |
| TF-IDF n-gramas de caracteres | 0,288 | [0,150; 0,425] | +0,110 [+0,010; +0,235] |
| Word2Vec / FastText propios | 0,200 | | el IC incluye 0 |
| **TF-IDF (TP1)** | 0,177 | [0,065; 0,314] | — |
| SBW (promedio) | 0,163 | | el IC incluye 0 |
| Azar | 0,041 | | |

Lecturas que **sí** se sostienen hoy:

- SBERT > TF-IDF de forma robusta.
- Ningún promedio de word vectors supera a TF-IDF.
- En consultas léxicas empatan (0,70 contra 0,70).
- En las consultas sin solapamiento, TF-IDF queda **debajo** del azar, por los distractores
  léxicos.
- En recuperar otro tomo de la misma saga, TF-IDF gana (MRR 0,881 contra 0,828 de SBERT).

Todas dependen de la validación de las consultas.

## 7. Problemas metodológicos a tener presentes (para la defensa)

- **Juicios de relevancia hechos por IA.** Se armaron leyendo las 200 sinopsis *antes* de
  ejecutar los modelos y juzgando cada consulta contra todo el corpus. No se buscaron
  candidatos por palabras, porque eso favorecería a TF-IDF. Aun así, son de un solo juez y
  hay que declararlo. **Al validar, no miren los resultados del notebook.**
- **Etiqueta "léxica".** Las consultas q01 y q10 parecían léxicas, pero sus relevantes no
  comparten las palabras: plural contra singular. Por eso el control con TF-IDF de n-gramas
  de caracteres.
- **Consulta amplia (q06).** Usa la etiqueta "Terror" como relevancia, con un piso de 0,27.
  Es la única que no se juzgó leyendo.
- **Modelos propios entrenados sobre los libros evaluados.** El corpus ampliado contiene a
  los 200. Es aceptable para un modelo no supervisado, pero hay que declararlo.
- **Sinopsis duplicada.** Norby 124496 y 124498 tienen el mismo texto. Infla el MRR por saga
  de todas las representaciones por igual.
- **Truncamiento.** El 86 % de las sinopsis se trunca, pero la variante por fragmentos no
  cambia la P@5 (0,363 contra 0,363). En esta tarea alcanza con el principio de la sinopsis.
- **Pocas consultas.** Con 16, los intervalos son anchos. Solo la diferencia de SBERT sobre
  TF-IDF tiene un IC que excluye el 0.
- **P3 usa un único split de 60 libros.** No se le agregó variabilidad.

## 8. Lo que no se pudo probar

- **Ejecución en Colab** [NO PROBADO]: no tengo acceso a Colab. Se probó en Docker con
  Python 3.12 y `requirements.txt` exacto. Riesgos en Colab: que `pip install gensim` pida
  reiniciar el entorno por una versión de numpy, que la descarga del SBW (1,1 GB, ~1-2 min)
  falle, y que la versión de sentence-transformers sea otra (ya se contempló el método
  renombrado).
- **Postgres** [NO PROBADO]: el camino `fuente="postgres"` no se ejecutó (no hay base). Ya
  no es parte del TP2.

## 9. Lo que requiere intervención del grupo

1. Validar `docs/propuesta_queries.md`: tildar relevantes, decidir dudosos y agregar
   faltantes, **sin mirar resultados**. Después pásenme las correcciones y genero
   `queries.json`.
2. ~~Subir los cambios a `borrar`~~: hecho.
3. Ejecutar el notebook en Colab y guardarlo con las salidas. La celda "Verificación de la
   ejecución" tiene que dar 15 de 15 (con las consultas provisorias falla solo la 14, a
   propósito).
4. Con los números definitivos: escribir la sección 7 del notebook y cerrar el informe. Lo
   puedo hacer yo y ustedes revisan. Completar en el informe el párrafo de uso de IA, con
   qué validaron ustedes.
5. Decidir si se mergea `borrar` a `main`.

## 10. Riesgos para la nota

- Entregar con `queries_propuesta.json` sin validar: es la parte que más pesa (30 %), y la
  consigna pide relevancia propia.
- En la defensa oral, poder explicar:
  - por qué la P@k tiene en cuenta los empates;
  - por qué TF-IDF puede quedar debajo del azar;
  - por qué los promedios de word vectors no superan a TF-IDF (dilución: SBW con similitud
    media 0,85 entre pares al azar);
  - por qué el MRR por saga favorece a TF-IDF.
- Decir "SBERT es mejor" sin los matices: empate en consultas léxicas, la saga y el control
  de n-gramas de caracteres.
- Que el notebook falle en Colab por algo del entorno. Hay que ejecutarlo con tiempo.

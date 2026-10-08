# Contexto del proyecto, para quien lo retome

> Documento de traspaso. Está escrito pensando en un agente de IA que tome el proyecto sin
> haber visto la conversación anterior, pero sirve igual para una persona.
>
> Para **instalar y correr**, ver [TRASPASO.md](TRASPASO.md). Para **entender el código**,
> ver [README.md](README.md). Este documento explica **por qué** las cosas son como son.

---

## 1. Qué es esto

Trabajo práctico de **Procesamiento del Lenguaje Natural**, Tecnicatura Universitaria en
Inteligencia Artificial (FCEIA, UNR). Grupo de cuatro: Valentín Cura, Franco Maragliano,
Sebastián Beltramo y Bautista Cortinas.

Son dos unidades encadenadas sobre el mismo corpus:

| Unidad | Qué pide | Estado |
|---|---|---|
| **1** | Scrapear 100–200 libros de Lectulandia con Playwright + BeautifulSoup, armar un CSV | **Terminada** |
| **2** | Traer el corpus desde PostgreSQL, responder 4 preguntas sobre TF-IDF, entrenar embeddings y compararlos contra uno pre-entrenado | **Terminada** |
| **TP2** | Embeddings y búsqueda semántica: SBERT, evaluación con `queries.json`, precision@k contra TF-IDF y piso de azar, notebook e informe | **Implementado; falta validar `queries.json` y cerrar el informe** (ver `docs/AUDITORIA_TP2.md`) |
| 3 en adelante | Un recomendador de libros | No empezada |

> **TP2, alcance acordado con la cátedra:** no hace falta Postgres ni Supabase (se trabaja
> desde el CSV), y la parte avanzada y las Partes E/F (pgvector, HNSW, filtro por metadata)
> quedaron fuera de la consigna.

El repositorio es <https://github.com/ValentinCura/NLP-C4-TUIA>.

**Ojo con las ramas:** `main` quedó en el commit `141a4ca`, que es el final de la Unidad 1.
Toda la Unidad 2 está en la rama **`borrar`**, que es donde se viene trabajando. Antes de
entregar hay que decidir si se mergea a `main`.

---

## 2. El principio que atraviesa todo el proyecto

Si hay una sola cosa que entender antes de tocar código, es esta.

> **El corpus guarda el texto crudo. El preprocesamiento pertenece al MODELO, no al corpus.**

La regla operativa para decidir dónde va cada transformación:

- ¿Podría cambiar la respuesta de algún modelo razonable?
  - **No** (colapsar espacios, `&nbsp;`, saltos de línea) → es **formato**, va en el corpus.
  - **Sí** (minúsculas, puntuación, stopwords, stemming) → es **lingüístico**, va en el modelo.

El motivo: TF-IDF trata al documento como un *multiconjunto* de términos, así que normalizar
agresivamente le aumenta la señal. Un modelo de oración lo trata como una *secuencia* y fue
entrenado sobre texto natural, así que el mismo preprocesamiento lo deja fuera de dominio.
Son necesidades opuestas. Guardar una versión normalizada en el corpus sería imponerle a un
modelo el hiperparámetro del otro.

Consecuencias concretas, por si se está tentado de "mejorar" algo:

- `Documento` **no tiene** un campo `.tokens`, y es deliberado. No existe *la* tokenización
  sino *la que quiere cada modelo*. Si mañana se agrega un LDA con otra lista de stopwords,
  ese atributo habría que partirlo en `.tokens_tfidf` y `.tokens_lda` — que un atributo
  necesite apellidarse con el nombre del modelo es la señal de que pertenece al modelo.
- `preprocesamiento.tokenizar()` recibe **`str`, no `Documento`**. Por eso
  `preprocesamiento.py` no importa `corpus.py` nunca, y se puede testear con literales sin
  base de datos. Es una regla verificable con un `grep`.
- El DDL **rechaza a propósito** un índice `GIN (to_tsvector('spanish', sinopsis))`, que es
  lo más tentador en una materia de PLN: `to_tsvector` baja a minúsculas, quita stopwords y
  hace stemming, o sea hornea el preprocesamiento de un modelo dentro del corpus.
- Hay un `COMMENT ON COLUMN libros.sinopsis` en la base con la prohibición escrita, para
  que la lea quien esté por hacer el `UPDATE`.

---

## 3. Arquitectura

```
src/scraper.py            869 lineas  UNIDAD 1: extraccion en 3 fases
src/db.py                 142         conexion a PostgreSQL desde el entorno
src/etl.py                298         CSV -> staging -> tabla tipada + verificacion
src/corpus.py             188         PostgreSQL -> list[Documento]   (texto CRUDO)
src/preprocesamiento.py   234         tokenizar()                     (TOKENIZADO)
src/experimentos.py       691         las 4 preguntas, medidas
src/embeddings.py         532         Word2Vec/FastText propios contra SBW
src/vectores.py                       TP2: validar (NaN/Inf/nulos/dim), normalizar, coseno
src/busqueda.py                       TP2: TF-IDF, promedios y SBERT con interfaz comun
src/evaluacion.py                     TP2: consultas, precision@k con empates, azar, bootstrap
tests/test_parsers.py     225         pruebas sin red ni base
tests/test_tp2.py                     TP2: 35 tests, varios adversariales
tools/                                TP2: generar y ejecutar el notebook
sql/01_esquema.sql        162         DDL con constraints e indices GIN
sql/02_consultas.sql      163         verificacion y exploracion desde DBeaver
```

El scraper corre en tres fases, y la separación entre las dos primeras es la decisión de
eficiencia central: **la deduplicación ocurre antes de visitar una sola ficha**. Recorrer un
listado trae ~24 libros de un request; visitar una ficha cuesta un request por libro. Las
categorías se solapan, así que deduplicar primero evitó 18 visitas repetidas sobre 200.

---

## 4. Los datos

### Dos corpus

| | `data/libros.csv` | `data/libros_ampliado.csv` |
|---|---|---|
| Libros | 200 | 1700 |
| Tabla en Postgres | `libros` | `libros_ampliado` |
| Rol | **El entregable.** No se toca | Solo para entrenar embeddings |
| **Texto crudo** — tokens | 30.271 | 255.120 |
| vocabulario | 7.246 | 26.704 |
| **Tokenizado** — tokens | 14.183 | 118.962 |
| vocabulario | 6.898 | 26.384 |
| palabras con frecuencia ≥5 | 543 | 5.026 |
| *hapax* (aparecen 1 vez) | 65% | 51% |

> Las dos mediciones son distintas y no hay que mezclarlas. "Texto crudo" cuenta palabras
> con un regex sobre la sinopsis tal como está guardada. "Tokenizado" pasa por
> `preprocesamiento.tokenizar()`, que baja a minúsculas, quita stopwords y descarta tokens
> de menos de 3 caracteres: es lo que realmente consumen TF-IDF y los embeddings, y recorta
> el texto a la mitad. Los experimentos reportan siempre la segunda.

El ampliado existe porque **el corpus entregable es inviable para entrenar embeddings**: con
`min_count=5` el modelo aprendería unas 500 palabras. Se scrapearon 1700 libros de las mismas cuatro
categorías a un archivo aparte, dejando el entregable intacto.

### Las 13 columnas

`id`, `titulo`, `titulo_serie`, `autores`, `n_autores`, `generos`, `serie`, `serie_num`,
`sinopsis`, `url_libro`, `portada`, `categoria_origen`, `fecha_extraccion`.

Decisiones que conviene no revertir sin entender:

- **`autores`, `generos` y `categoria_origen` son arrays JSON**, no strings con separador.
  El destino es `::jsonb` en Postgres sin parsear delimitadores. Lo pidió el profesor
  explícitamente ("hacer una lista, no separar por coma ni punto y coma").
- **`url_libro` y `portada` guardan solo la parte variable.** Los prefijos
  (`https://ww3.lectulandia.co/book/` y `https://assets.lectulandia.co/b/ab/`) son
  constantes verificadas sobre 189 portadas de 6 géneros. El recorte **no es a ciegas**: si
  una URL no empieza con el prefijo esperado, el scraper lanza un error explícito.
- **El CSV va en UTF-8 SIN BOM.** La primera decisión fue `utf-8-sig` para que abriera
  cómodo en Excel, pero como el destino real es un `COPY` a PostgreSQL, el BOM se colaría
  en el nombre de la primera columna. Gana el destino real.
- **`''` en el CSV se traduce a `NULL` en Postgres**, solo en las tres columnas donde la
  ausencia es informativa (`serie`, `serie_num`, `portada`). El CSV usa `''` porque el
  formato CSV *no tiene* NULL; SQL sí lo tiene. El ETL es el traductor entre los dos
  idiomas, y un `CHECK (serie <> '')` garantiza que exista una sola codificación de la
  ausencia.
- **`serie` quedó en columna propia**, no concatenada al título. El profesor sugirió
  concatenar; se midió y el **53% de los libros pertenece a una serie**, así que no es un
  campo mayoritariamente vacío. Se mantuvieron las columnas atómicas **y** se agregó
  `titulo_serie` como columna derivada que sí concatena, para la legibilidad que pedía.
- **Los géneros NO se canonicalizan.** El sitio tiene un typo propio
  (`Publicaciónes periódicas`) y se conserva, por trazabilidad al origen.

---

## 5. Los cuatro experimentos, y qué dieron

Las preguntas se respondieron **midiendo**, no razonando. **Tres de las cuatro dieron
distinto de lo esperado**, y se reportaron como salieron. Si alguien los vuelve a correr y
le da otra cosa, hay un bug.

### 1. ¿Cuántas sinopsis hacen falta para que TF-IDF sea estable?

Se fijan 20 documentos sonda y se varía el corpus de **fondo** que define el IDF, con 30
réplicas por tamaño. Se mide el Jaccard@10 entre réplicas.

| fondo | Jaccard@10 | solape entre réplicas |
|---|---|---|
| 10 | 0,233 | 1% |
| 100 | 0,289 | 6% |
| 600 | 0,445 | 36% |
| 1200 | 0,622 | 71% |

**No se aplana**: el último salto es el más grande. Con 1700 documentos no alcanza. Causa:
el 51% del vocabulario aparece una sola vez y TF-IDF premia justamente esos términos.

> El diseño tiene una sutileza: lo único que varía es el **fondo**. Las sondas son fijas. Si
> se resamplearan también los documentos medidos, se mezclarían dos fuentes de variación.
> La primera versión del experimento tenía las sondas participando del `fit` del
> vectorizador, así que con fondo chico **definían ellas mismas el IDF** e inflaban la
> coincidencia: la curva bajaba y subía sin sentido. Se corrigió a `fit()` sobre el fondo y
> `transform()` sobre las sondas.
>
> La columna de solape es una advertencia: al muestrear sin reposición de un conjunto
> finito, cuando el fondo se acerca al total las réplicas comparten casi los mismos
> documentos y el Jaccard sube por eso, no por convergencia.

### 2. ¿Qué sesgo introduce que las sinopsis sean texto promocional?

**Contradice la hipótesis obvia.** Solo el 0,8% de los términos característicos son
vocabulario publicitario, y quitarlo del clasificador no cambia el micro-F1: **−0,000 ± 0,009**
sobre 20 splits. (El "+0,005" anterior venía de un solo split y de una ablación con un bug:
no quitaba las palabras con tilde.)

La razón: TF-IDF ya penaliza esas palabras porque aparecen en *todas* las sinopsis y su IDF
es bajo. El filtro ya estaba puesto por construcción.

El sesgo real está en otro lado: una sinopsis **no describe el libro, selecciona lo
vendible**. Un clasificador entrenado con esto aprende de qué trata *la campaña*.

### 3. ¿Cómo cambia la evaluación en multi-etiqueta?

Ningún libro tiene un solo género (mínimo 2, promedio 2,84): es multi-etiqueta **por
construcción**.

| métrica | clasificador | trivial |
|---|---|---|
| subset accuracy | 0,400 | 0,000 |
| f1 micro | 0,750 | **0,470** |
| f1 macro | 0,541 | **0,092** |

El "trivial" no lee el texto: predice siempre "Novela", que está en el 85% de los libros.
Reportar solo micro-F1 haría pasar por aceptable a un modelo ciego. Además, 13 de los 27
géneros tienen menos de 5 libros: no se pueden aprender ni estratificar.

### 4. ¿Qué pasa con los libros en gallego o catalán?

**Hay una sinopsis en gallego** (*Morning Star*, id 124205), y `lingua` la etiqueta como
castellano con confianza 0,998, la mínima del corpus. La encuentra un segundo instrumento
basado en marcadores léxicos del gallego. La versión anterior concluía "las 200 son
castellano": era falso.

Se verificó además el instrumento con tres traducciones del mismo párrafo:

| entrada | detectado | confianza |
|---|---|---|
| castellano | SPANISH | 0,97 |
| catalán | CATALAN | 1,00 |
| **gallego** | **PORTUGUESE** | **0,91** |

**`lingua` no tiene modelo de gallego.** Lo reporta como portugués con alta confianza: un
falso negativo silencioso, el peor caso porque no se distingue de un acierto.

---

## 6. Embeddings

Word2Vec y FastText sobre el corpus ampliado (118.962 tokens ya tokenizados), contra el
SBW (1.400 millones de palabras, unas 10.000 veces más).

Parámetros propios: `vector_size=100` (no 300: el corpus no sostiene tantas dimensiones),
`window=5`, `min_count=3`, `sg=1` (skip-gram anda mejor en corpus chicos), `epochs=30`,
`workers=1` y `seed=42` (reproducible byte a byte; con `workers=4` no lo era).

### Los tres fallan distinto, y se midió

Fracción de vecinos que comparte las primeras 4 letras con la palabra consultada:

| modelo | por forma | qué hace |
|---|---|---|
| word2vec propio | 1% | Responde a coocurrencia, pero devuelve **nombres de personajes** (`waxillium`, `sarren`): memoriza libros concretos |
| fasttext propio | 38% | Responde a la forma escrita: `magia`→`mafia`, `muerte`→`suerte` |
| SBW | 18% | Responde al significado; ese 18% son variantes legítimas (`mago`/`magos`) |

### El resultado más interesante del proyecto

Vector de documento = promedio de vectores de palabra. Tarea: recuperar libros de la misma
saga, usando la columna `serie` como *ground truth* gratis.

| representación | MRR | acierto@1 |
|---|---|---|
| **TF-IDF (sin embeddings)** | **0,881** | **83,5%** |
| SBERT distiluse (TP2) | 0,828 | 78,5% |
| word2vec propio | 0,780 | 68,4% |
| fasttext propio | 0,712 | 59,5% |
| SBW pre-entrenado | 0,578 | 46,8% |

**Gana TF-IDF y el SBW sale último.** Los libros de una saga comparten **nombres propios**,
palabras rarísimas a las que TF-IDF da peso máximo. Los embeddings *generalizan*, y acá
generalizar destruye la señal.

> **Esto importa para la Unidad 3.** No significa que los embeddings sirvan menos para un
> recomendador. Recuperar "el mismo universo narrativo" premia lo específico; recomendar
> "algo parecido pero distinto" premia lo que generaliza. Son tareas opuestas y con esta no
> se puede concluir sobre aquella. **No usar esta tabla para descartar embeddings en el
> recomendador.**

Ponderar por IDF **no mejora de forma uniforme**: sube Word2Vec (+0,016), no cambia
FastText (+0,004), empeora el SBW (−0,019). Con `workers=4` daba lo contrario para Word2Vec y
FastText: era ruido de un entrenamiento no reproducible.

---

## 7. Bugs encontrados, para no repetirlos

Los que más tiempo costaron. Varios son trampas que no dan error: producen datos plausibles
pero equivocados.

| Bug | Síntoma | Causa y arreglo |
|---|---|---|
| **Conexión de 130 segundos** | Todo funcionaba, pero lentísimo, con CPU en cero | `localhost` en Windows resuelve primero a `::1` (IPv6) y el puerto está atado a IPv4. Cada conexión esperaba el *timeout* de TCP. Con `127.0.0.1`: 0,02s. **Factor 6500** |
| **Colgado de 10+ minutos** | Con Docker apagado, `db.py` no mostraba nada | Faltaba `connect_timeout`. psycopg esperaba 130s por intento × 10 reintentos. Ahora 5,1s por intento |
| **Tarjetas del libro equivocado** | Ninguno: el CSV salía lleno de datos plausibles | La ficha individual tiene 12 `<article class="card">` de libros *relacionados*, misma etiqueta que el listado. Por eso hay **dos parsers separados** y un test que lo documenta |
| **El `id` del libro equivocado** | Ninguno | Los `<article id="post-N">` son los relacionados. El id real sale de la clase del `<body>` (`postid-N`) |
| **`"víctima perfecta ."`** | Espacio antes del punto | `get_text(separator=" ")` no distingue `<b>` (inline, no separa) de `<br>` (bloque, sí). Se reemplazan solo los `<br>` antes de extraer |
| **`"Más Allá 21 - Más Allá 21"`** | Columna derivada redundante | El título ya contenía la saga. Lo detectó **mirar una muestra de datos**, no un test |
| **`categoria_origen` desactualizado** | Ninguno | Es el único campo que no sale de la ficha sino de la Fase A. Al reanudar quedaba viejo *en silencio*. Se sincroniza |
| **Diff espurio de 200 líneas** | El CSV figuraba modificado sin cambiar datos | pandas escribe CRLF en Windows. Fijado `lineterminator="\n"` |
| **Exit code 1 con el corpus ampliado** | Scrapeo perfecto que terminaba en error | `validar()` aplicaba siempre "cantidad entre 50 y 200", que es requisito del entregable, no de todo corpus. Ahora es parámetro |
| **`CHECK (serie_num > 0)`** | Habría roto la carga | 4 libros tienen `serie_num = 0`: omnibus y precuelas. Va `>= 0` |
| **Cifra hardcodeada** | Decía "63% del vocabulario" | Cierto para el corpus de 200, falso para el ampliado (51%). Ahora se calcula del corpus en uso |

---

## 8. Errores de método que cometió el agente anterior

Vale la pena conocerlos, porque son recurrentes.

**Afirmar sin verificar.** El README decía *"el dominio del sitio cambió, hubo que localizar
el dominio vivo"*. Falso: el enunciado del TP traía el link correcto incrustado en la página
1. El agente leyó el texto del PDF, **supuso** que "Lectulandia" apuntaba a
`lectulandia.com`, le dio 403 y construyó una narrativa sobre esa suposición. Lo detectó el
usuario. Hubo que corregirlo en cuatro lugares.

**Escribir conclusiones antes de medir.** En los embeddings, el texto afirmaba que ponderar
por IDF mejora. La tabla mostraba que mejora en uno, no cambia en otro y empeora en el
tercero. Ahora esa conclusión **se calcula de los números de la corrida**.

**Cifras a mano que dejan de ser ciertas.** Ver el último bug de la tabla anterior.

La contramedida que funcionó: **cuando un texto afirme un número, que el número salga del
dato y no del teclado.**

---

## 9. Entorno

- **Python 3.13.14** en `venv/`. Hay wheels `cp313/win_amd64` para gensim, spacy y
  scikit-learn: no hay que compilar nada.
- **PostgreSQL 17** en Docker (`docker-compose.yml`), puerto **5433** en `127.0.0.1`.
- **DBeaver** para inspeccionar: host `127.0.0.1` (no `localhost`), puerto 5433,
  base/usuario/contraseña `tuia`.
- El **SBW** (1,07 GB) vive **fuera del repositorio**, en la carpeta que lo contiene.
- `venv/` (797 MB) y `data/modelos/` (777 MB) **no están en git**: el segundo tiene un
  archivo de 762 MB y GitHub rechaza todo lo que supere 100 MB. Ambos se regeneran.
- **En la PC de Bautista, Windows Smart App Control bloquea DLLs** de scikit-learn
  (`sklearn.metrics`), spaCy y, por arrastre, sentence-transformers. No es un bug del
  proyecto y se decidió no desactivarlo. `preprocesamiento.py` lee la lista de stopwords sin
  cargar spaCy para esquivarlo. Para ejecutar todo se usó un contenedor Docker
  `python:3.12-slim`; la entrega se ejecuta en Colab.

---

## 10. Cómo trabajar en este proyecto

Convenciones que ya están establecidas y conviene mantener:

- **Código y comentarios en castellano**, muy comentados. Los comentarios explican **por
  qué**, no qué.
- **El código fuente de `src/` es ASCII puro**, sin tildes, por la consola de Windows. Los
  `.md` sí llevan tildes. Cada `main()` hace
  `sys.stdout.reconfigure(encoding="utf-8")`.
- **Cada módulo tiene un `main()` con `argparse`** y se puede correr solo.
- **Las verificaciones usan el formato `[OK ]` / `[FALLA]`** y devuelven exit code.
- **Los tests no necesitan red ni base**: corren contra HTML guardado en
  `tests/fixtures/`, que **sí está versionado** para que cualquiera pueda validar.
- **Un test por cada trampa encontrada**, para que quede documentada además de corregida.
- **Preferir que falle ruidosamente** antes que guardar un dato dudoso en silencio.

---

## 11. Qué sigue

0. **TP2: validar `queries_propuesta.json`** (con `docs/propuesta_queries.md`), generar
   `queries.json`, re-ejecutar el notebook en Colab, escribir sus conclusiones y cerrar el
   informe (`docs/informe_borrador.md` -> `informe.pdf`). Detalle en `docs/AUDITORIA_TP2.md`.
1. **Decidir qué pasa con `main`**, que está varios commits atrás y no tiene la Unidad 2 ni
   el TP2.
2. ~~Sacar el `.env` de la rama~~ — hecho: se sacó del índice (`git rm --cached`) y ahora
   lo cubre el `.gitignore`. Contenía solo los valores por defecto del Docker local.
3. **Confirmar con el profesor** que los arrays JSON son lo que pidió cuando dijo "hacer una
   lista". El argumento del `::jsonb` está en el README.
4. **Unidad 3: el recomendador.** Lo que ya está listo para eso: el corpus en Postgres, las
   dos representaciones (TF-IDF y embeddings), la columna `serie` como *ground truth* y la
   advertencia de la sección 6 sobre no sacar conclusiones apresuradas de aquella tabla.

Hay cosas que el profesor remarcó y conviene tener presentes: **escalabilidad** (el diseño
tiene que poder crecer de 200 a 2M registros, y el README tiene una sección sobre eso) y
**documentación** (que el README explique cómo proceder para correr el proyecto).

---

## 12. TP2: qué se hizo el 2026-10-07 y qué decisiones no hay que deshacer

Se auditó todo lo existente contra la consigna del TP2, se corrigió lo que estaba mal y se
implementó lo que faltaba. Informe completo en `docs/AUDITORIA_TP2.md`; pasos para
continuar en `TRASPASO.md`.

**Decisiones de método de la evaluación** (todas con tests en `tests/test_tp2.py`):

- **precision@k con empates exactos.** TF-IDF da 0 a todo lo que no comparte palabras con
  la consulta, y `argsort` ordenaría esos empates por posición en el CSV. Se usa la
  precisión *esperada* bajo desempate aleatorio (hipergeométrica, sin semilla). Una consulta
  sin palabras en común con el corpus da exactamente el piso. No reemplazar por un
  `argsort` simple.
- **Piso de azar exacto: |R|/N.** La simulación solo se usa para mostrar la dispersión.
- **Se reportan techo y R-precision**, porque con 2 relevantes P@5 no pasa de 0,4.
- **Promedio macro con bootstrap sobre consultas, y comparaciones pareadas.** Con 16
  consultas, una diferencia de medias sin intervalo no dice nada.
- **Los dudosos cuentan como no relevantes.** Se informa aparte la variante que sí los
  cuenta.
- **Cada modelo recibe su propio preprocesamiento:** tokenizado para TF-IDF y los
  promedios, texto crudo para SBERT.
- **TF-IDF de n-gramas de caracteres es un CONTROL, no la línea de base.** Sirve para
  separar fallas de morfología (plurales) de fallas de semántica.
- **El truncamiento de SBERT se cuenta con su tokenizador.** Es el 86 % de las sinopsis;
  contando palabras se subestima. La variante por fragmentos mostró que, en esta tarea, el
  truncamiento no cambia P@5.

**Los juicios de relevancia** de `queries_propuesta.json` los propuso un asistente de IA,
que leyó las 200 sinopsis antes de ejecutar ningún modelo. No se buscaron candidatos por
palabras, porque eso favorecería a TF-IDF. Las tres consultas sin solapamiento se
reformularon mirando **solo** el solapamiento léxico, nunca resultados. **Falta que el
grupo las valide.** Después de validarlas no se tocan, salga lo que salga.

**Errores encontrados en esta sesión, para no repetirlos** (mismo patrón que la sección 8:
afirmaciones sin verificar):

| Error | Cómo se vio |
|---|---|
| "Las 200 sinopsis son castellano" | Leyendo el corpus: *Morning Star* está en gallego; `lingua` la llama castellano con 0,998 |
| Ablación promocional sin las palabras con tilde | Contando qué términos quitaba `stop_words` |
| Una conclusión que cambiaba entre corridas (IDF en Word2Vec) | Al fijar `workers=1`, se invirtió |
| "Estable" marcado en el piso de ruido y en el último punto por vacuidad | Comparando la tabla con su propia conclusión |
| "Si se cuentan los dudosos, el orden no cambia" (escrito en el borrador del informe) | Contrastando cada cifra del informe con la salida: sí cambia |

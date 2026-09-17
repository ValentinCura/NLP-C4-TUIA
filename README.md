# Procesamiento del Lenguaje Natural — TUIA

Corpus de sinopsis de libros construido a partir de la informacion publica de
[Lectulandia](https://ww3.lectulandia.co), y su analisis.

- **Unidad 1** — Extraccion de metadatos y sinopsis con Playwright y BeautifulSoup.
- **Unidad 2** — Carga a PostgreSQL, analisis TF-IDF y embeddings de palabras.

## Integrantes del grupo

- Valentín Cura
- Franco Maragliano
- Sebastián Beltramo
- Bautista Cortinas

---

## El dataset

> ### **[`data/libros.csv`](data/libros.csv)** ← el entregable
>
> **200 libros · 13 columnas · 233 KB · UTF-8 sin BOM**

Se puede abrir directamente desde GitHub haciendo clic en el enlace de arriba: GitHub
renderiza los CSV como tabla, con buscador incluido.

Para inspeccionarlo desde Python:

```bash
python -c "import pandas as pd; print(pd.read_csv('data/libros.csv').head())"
```

Así se ve un registro real del archivo (con la sinopsis recortada para que entre):

| campo | valor |
|---|---|
| `id` | `125043` |
| `titulo` | `Trilogía de los dioses` |
| `titulo_serie` | `Trilogía de los dioses` |
| `autores` | `["Ángel Torres Quesada"]` |
| `n_autores` | `1` |
| `generos` | `["Ciencia ficción","Novela"]` |
| `serie` | *(vacío: es autoconclusivo)* |
| `serie_num` | *(vacío)* |
| `sinopsis` | `Por fin el lector puede conocer el desenlace de la ya mítica serie "Los dioses"…` |
| `url_libro` | `trilogia-de-los-dioses/` |
| `portada` | `Angel%20Torres%20Quesada/Trilogia%20de%20los%20dioses%20(3)/big.jpg` |
| `categoria_origen` | `["ciencia-ficcion"]` |
| `fecha_extraccion` | `2026-09-09` |

Y la misma fila tal cual está escrita en el archivo. Nótese cómo el array JSON y las
comillas dobles que trae la propia sinopsis quedan correctamente escapados según RFC 4180
(el campo se envuelve entre comillas y las internas se duplican):

```csv
id,titulo,titulo_serie,autores,n_autores,generos,serie,serie_num,sinopsis,...
125043,Trilogía de los dioses,Trilogía de los dioses,"[""Ángel Torres Quesada""]",1,"[""Ciencia ficción"",""Novela""]",,,"Por fin el lector puede conocer el desenlace de la ya mítica serie ""Los dioses"" que…",...
```

El detalle de cada columna está en [Modelo de datos](#modelo-de-datos).

## Categorías seleccionadas

La cátedra habilitó usar más de una categoría. Se eligieron cuatro, mezclando pares
semánticamente **cercanos** (ciencia ficción / fantástico comparten vocabulario especulativo)
y **lejanos** (histórico), para que el recomendador de la Unidad 2 tenga que discriminar de
verdad y no separe los grupos por casualidad.

| Categoría | URL | Libros aportados |
|---|---|---|
| Ciencia ficción | `/genero/ciencia-ficcion/` | 50 |
| Fantástico | `/genero/fantastico/` | 50 |
| Terror | `/genero/terror/` | 50 |
| Histórico | `/genero/historico/` | 50 |

## Cantidad de libros extraídos

**200 libros únicos**, sin duplicados por `url_libro`.

- 16 libros pertenecen a más de una categoría semilla (por eso las categorías suman 218
  apariciones sobre 200 libros únicos).
- El 100% tiene sinopsis; la mediana es de 873 caracteres (mínimo 243, máximo 2228).
- El 53% pertenece a alguna serie.
- 14 libros tienen más de un autor.
- 27 géneros distintos declarados por el sitio.

## Qué hace este proyecto, en breve

Si nunca viste el proyecto, esto es lo que hay que saber antes de cualquier detalle.

**Unidad 1 — conseguir el texto.** *Scrapear* es leer páginas web de forma automática para
quedarse con los datos que uno necesita. Acá se usan dos herramientas que hacen cosas
distintas: **Playwright** maneja un navegador Chrome sin ventana (abre páginas, espera que
carguen, pasa a la siguiente) y **BeautifulSoup** lee el HTML que el navegador trajo y
encuentra los datos adentro.

La pieza que hace ese último trabajo se llama **parser**: una función que recibe el HTML
de una página y devuelve los datos ya ordenados. Este proyecto tiene dos, uno para la
página que lista los libros de una categoría y otro para la ficha de un libro. Están en
[`src/scraper.py`](src/scraper.py) y se explican en detalle más abajo.

**Unidad 2 — analizar el texto.** El corpus se carga a una base **PostgreSQL** (que corre
en Docker) y desde ahí lo consumen los análisis: TF-IDF para encontrar qué distingue a cada
sinopsis, y *embeddings* (Word2Vec, FastText) para representar palabras como vectores.

La decisión que atraviesa todo el proyecto: **el corpus guarda el texto crudo, sin
normalizar**. Pasar a minúsculas o quitar palabras vacías es algo que necesita *cada
modelo*, y cada uno necesita algo distinto, así que se hace en el momento de usarlo y no
antes. Está explicado en [El preprocesamiento pertenece al modelo](#el-preprocesamiento-pertenece-al-modelo).

---

## Instalación

Requiere **Python 3.10 o superior** y **Docker Desktop**. Desde la raíz del proyecto:

```bash
python -m venv venv
```

```bash
venv/Scripts/activate
```

```bash
pip install -r requirements.txt
```

Playwright necesita descargar su propio Chromium (~150 MB, una sola vez):

```bash
python -m playwright install chromium
```

> En Linux o macOS el activador es `source venv/bin/activate`.

### Base de datos

La configuración de conexión sale de un archivo `.env`, que no se versiona. Crearlo a
partir del ejemplo:

```bash
copy .env.example .env
```

Con **Docker Desktop abierto**, levantar PostgreSQL:

```bash
docker compose up -d
```

Verificar que quedó `healthy` antes de seguir:

```bash
docker compose ps
```

Cargar el corpus a la base:

```bash
python src/etl.py
```

Tiene que terminar con `CARGA OK`. Para el corpus ampliado (opcional, solo hace falta para
entrenar embeddings):

```bash
python src/etl.py --csv data/libros_ampliado.csv --tabla libros_ampliado
```

### Conectarse con DBeaver

| Campo | Valor |
|---|---|
| Host | `127.0.0.1` |
| Puerto | `5433` |
| Base | `tuia` |
| Usuario | `tuia` |
| Contraseña | `tuia` |

> **Usar `127.0.0.1`, no `localhost`.** El puerto se publica atado a IPv4 para no exponer la
> base a la red, pero `localhost` en Windows resuelve primero a `::1` (IPv6), donde no
> escucha nadie: cada conexión espera el *timeout* de TCP antes de caer a IPv4. Medido en
> este proyecto: **130 segundos contra 0,02**. Todo funciona igual, solo que
> inexplicablemente lento.

El puerto es **5433** y no el 5432 habitual, para no chocar con un PostgreSQL nativo que
alguien pueda tener instalado.

### El modelo pre-entrenado (opcional)

Para comparar contra el Spanish Billion Word Corpus hay que descargar
`SBW-vectors-300-min5.bin.gz` (**1,07 GB**) de [crscardellino.ar/SBWCE](https://crscardellino.ar/SBWCE/)
y dejarlo **en la carpeta que contiene al repositorio**, no adentro:

```
Procesamiento del Lenguaje Natural/
├── SBW-vectors-300-min5.bin.gz     <- acá
└── TUIA/                           <- el repositorio
```

Sin ese archivo todo lo demás sigue funcionando: los módulos avisan y usan solo los
modelos propios.

---

## Ejecución

### Unidad 1 — extracción

Corrida completa (Fases A, B y C). Tarda unos 8–9 minutos, casi todo esperando las pausas
entre requests:

```bash
python src/scraper.py
```

Genera `data/libros.csv` e imprime la estructura del dataset y los controles de calidad.
Devuelve código de salida distinto de cero si algún control falla.

| Comando | Para qué sirve |
|---|---|
| `python src/scraper.py --max-libros 5` | Prueba rápida: visita solo 5 fichas |
| `python src/scraper.py --solo-descubrir` | Corre solo la Fase A, sin visitar fichas |
| `python src/scraper.py --solo-consolidar` | Regenera el CSV sin usar internet |
| `python src/scraper.py --cupo 25` | Cambia cuántos libros aporta cada categoría |
| `python src/scraper.py --categorias terror poesia` | Usa otras categorías |
| `python src/scraper.py --salida data/otro.csv` | Arma un corpus aparte sin tocar el entregable |
| `python src/scraper.py --ver-navegador` | Muestra la ventana de Chromium |

El navegador corre **headless** (sin ventana) por defecto.

**Reanudación.** La extracción guarda cada libro apenas lo obtiene, en un archivo
`data/<nombre>_parcial.jsonl`. Si la corrida se interrumpe, volver a ejecutar el mismo
comando retoma donde había quedado. Para forzar una extracción desde cero, borrar ese
archivo.

### Unidad 2 — base de datos y análisis

| Comando | Qué hace |
|---|---|
| `python src/db.py` | Prueba la conexión e informa contra qué base se conectó |
| `python src/etl.py` | Carga el CSV a PostgreSQL y verifica la carga |
| `python src/etl.py --recrear` | Borra la tabla y la vuelve a crear desde cero |
| `python src/etl.py --solo-verificar` | Corre los controles sin cargar nada |
| `python src/corpus.py` | Resumen del corpus leído desde la base |
| `python src/preprocesamiento.py --demo` | Muestra las dos versiones del texto lado a lado |

### Los experimentos

Cada uno responde una de las cuatro preguntas de la consigna **con mediciones**, no con
prosa. Aceptan `--tabla libros_ampliado` para correr sobre el corpus grande.

```bash
python src/experimentos.py todos
```

| Comando | Pregunta que responde |
|---|---|
| `python src/experimentos.py idioma` | ¿Qué pasa con los libros en gallego o catalán? |
| `python src/experimentos.py estabilidad` | ¿Cuántas sinopsis hacen falta para que TF-IDF sea estable? |
| `python src/experimentos.py promocional` | ¿Qué sesgo introduce que el texto sea publicitario? |
| `python src/experimentos.py multietiqueta` | ¿Cómo cambia la evaluación respecto de multi-clase? |

### Los embeddings

```bash
python src/embeddings.py entrenar
```

| Comando | Qué hace |
|---|---|
| `python src/embeddings.py entrenar` | Entrena Word2Vec y FastText sobre el corpus ampliado |
| `python src/embeddings.py vecinos` | Compara vecinos más cercanos en los tres modelos |
| `python src/embeddings.py documentos` | Compara formas de armar el vector de documento |

Los modelos entrenados van a `data/modelos/` y **no se versionan**: se regeneran en
segundos con `entrenar`.

### Tests

Los parsers se prueban contra HTML guardado en disco, sin tocar la red ni la base (corre en
menos de un segundo):

```bash
python tests/test_parsers.py
```

---

## Resultados de la Unidad 2

Las cuatro preguntas se respondieron midiendo sobre el corpus. **Tres de las cuatro dieron
distinto de lo esperado**, y se reportan como salieron.

### 1. ¿Cuántas sinopsis hacen falta para que TF-IDF sea estable?

Se fijan 20 documentos sonda y se varía el **corpus de fondo** que define el IDF, con 30
réplicas por tamaño. Se mide cuánto coinciden los 10 términos característicos entre
réplicas distintas del mismo tamaño.

Es importante que lo único que varíe sea el fondo: si se resamplearan también los
documentos medidos, se mezclarían dos fuentes de variación.

| fondo | Jaccard@10 | solape entre réplicas |
|---|---|---|
| 10 | 0,233 | 1% |
| 100 | 0,289 | 6% |
| 300 | 0,371 | 18% |
| 600 | 0,445 | 36% |
| 1200 | 0,622 | 71% |

**La curva no se aplana**: el último salto es el más grande. Con 1700 documentos todavía no
alcanza. La causa es que el **51% del vocabulario aparece una sola vez**, y TF-IDF premia
justamente esos términos, con lo cual los característicos de un documento son en buena
medida accidentes de muestreo.

La columna de solape es una advertencia metodológica: al muestrear sin reposición de un
conjunto finito, cuando el fondo se acerca al total las réplicas comparten casi los mismos
documentos y el Jaccard sube por esa razón y no porque haya convergencia.

### 2. ¿Qué sesgo introduce el texto promocional?

**El resultado contradice la hipótesis obvia.** Solo el **1,5%** de los términos
característicos son vocabulario publicitario, y quitarlo del clasificador cambia el
micro-F1 en **+0,005**, o sea nada.

La razón: TF-IDF ya penaliza esas palabras, porque aparecen en *todas* las sinopsis y su
IDF es bajo. El filtro ya estaba puesto por construcción.

El sesgo real está en otro lado y no se arregla con una lista de palabras: una sinopsis no
describe el libro, **selecciona lo vendible**. Omite el final, exagera el conflicto y
destaca lo que se parece a otros éxitos. Un clasificador entrenado con esto aprende de qué
trata *la campaña*, y no generalizaría a reseñas ni al texto del libro.

### 3. ¿Cómo cambia la evaluación en multi-etiqueta?

Ningún libro del corpus tiene un solo género (mínimo 2, promedio 2,84), así que el problema
es multi-etiqueta **por construcción**, no por elección.

| métrica | clasificador | trivial |
|---|---|---|
| subset accuracy | 0,400 | 0,000 |
| hamming loss | 0,113 | 0,192 |
| **f1 micro** | 0,750 | **0,470** |
| **f1 macro** | 0,541 | **0,092** |

La columna "trivial" es un clasificador que **no lee el texto**: predice siempre "Novela",
que aparece en el 85% de los libros. Saca micro-F1 0,470 y macro-F1 0,092.

Ahí está la respuesta: reportar solo micro-F1 haría pasar por aceptable a un modelo ciego.
En multi-clase hay una métrica obvia; en multi-etiqueta hay varias que cuentan historias
opuestas, y hay que reportar micro y macro **juntas** y siempre contra una línea base
trivial.

Además, 13 de los 27 géneros tienen menos de 5 libros: no se pueden aprender ni
estratificar en un split.

### 4. ¿Qué pasa con los libros en gallego o catalán?

**Las 200 sinopsis son castellano**, con confianza mínima 0,998. La premisa no se cumple en
nuestra muestra.

Pero decir "no encontramos ninguno" no significa nada si el instrumento no puede
encontrarlos, así que se verificó con tres traducciones del mismo párrafo:

| lengua de entrada | detectado | confianza |
|---|---|---|
| castellano | SPANISH | 0,97 |
| catalán | CATALAN | 1,00 |
| **gallego** | **PORTUGUESE** | **0,91** |

**`lingua` no tiene modelo de gallego.** Lo reporta como portugués con alta confianza: un
falso negativo silencioso, que es el peor caso porque no se distingue de un acierto. El
catalán sí se detecta bien.

Conclusión operativa: detectar el idioma antes de tokenizar sí, pero sin confiar en la
etiqueta cuando las lenguas candidatas son cercanas y alguna no está en el repertorio del
detector.

---

## Embeddings: propio contra pre-entrenado

Se entrenaron Word2Vec y FastText sobre el corpus ampliado (255.120 tokens) y se compararon
contra el SBW (1.400 millones de palabras, unas 5000 veces más).

**Parámetros del modelo propio y por qué:**

| Parámetro | Valor | Razón |
|---|---|---|
| `vector_size` | 100 | El SBW usa 300 porque tiene datos para estimarlas. Con 255k tokens, las dimensiones de más modelan ruido |
| `window` | 5 | Punto medio entre relaciones sintácticas (2–3) y temáticas (10) |
| `min_count` | 3 | El default de 5 descartaría demasiado en un corpus chico |
| `sg` | 1 (skip-gram) | Anda mejor que CBOW en corpus chicos y con palabras poco frecuentes |
| `epochs` | 30 | El default de 5 asume un corpus grande. Con pocos datos convienen más pasadas |

> Con `workers=4` el entrenamiento **no** es del todo reproducible aunque se fije la
> semilla, porque el orden en que los hilos actualizan los pesos varía. Para
> reproducibilidad exacta habría que usar `workers=1`.

### Vecinos más cercanos

```
'magia'
  word2vec propio      ivy 0.64, mensual 0.57, waxillium 0.56, zach 0.53
  fasttext propio      mafia 0.70, maga 0.69, mago 0.67, magos 0.60
  SBW pre-entrenado    mágico 0.70, mago 0.70, hechizos 0.69, mágicas 0.69

'vampiro'
  word2vec propio      sarren 0.65, rodrigo 0.60, beber 0.59, creador 0.59
  fasttext propio      vampira 0.85, vampirismo 0.84, vampiros 0.82, zafiro 0.62
  SBW pre-entrenado    vampiros 0.80, vampira 0.74, Drácula 0.74, licántropo 0.73
```

Los tres fallan **distinto**, y eso se puede medir. Contando qué fracción de los vecinos
comparte las primeras 4 letras con la palabra consultada:

| modelo | vecinos por forma | qué significa |
|---|---|---|
| word2vec propio | **1%** | Responde a coocurrencia. Pero sus vecinos son nombres de personajes (`waxillium`, `sarren`): está memorizando libros concretos, no aprendiendo significado |
| fasttext propio | **38%** | Responde a la forma escrita. Los n-gramas ayudan con palabras raras (`vampirismo`) pero producen falsos amigos: `magia`→`mafia`, `muerte`→`suerte` |
| SBW pre-entrenado | **18%** | Responde al significado. Ese 18% son variantes legítimas (`mago`/`magos`) |

### Vector de documento

Se construye **promediando** los vectores de las palabras del documento. La tarea de
evaluación es recuperar libros de la misma saga, usando la columna `serie` como *ground
truth* que no hubo que anotar a mano.

| representación | MRR | acierto@1 |
|---|---|---|
| **TF-IDF (sin embeddings)** | **0,881** | **83,5%** |
| word2vec propio, promedio | 0,787 | 68,4% |
| fasttext propio, promedio | 0,696 | 58,2% |
| SBW pre-entrenado, promedio | 0,578 | 46,8% |

**Gana TF-IDF y el SBW sale último**, que es lo contrario de lo esperado. La explicación
está en qué pide la tarea: los libros de una saga comparten **nombres propios**, que son
palabras rarísimas, y TF-IDF les da el peso máximo justamente por eso. Los embeddings
*generalizan*, y acá generalizar destruye la señal: el SBW convierte cada nombre propio en
"un nombre propio más".

La lección no es que los embeddings sean peores, sino que **la representación correcta
depende de la tarea**. Para recuperar "el mismo universo narrativo" conviene lo específico;
para recomendar "algo parecido pero distinto", que es lo que querría un recomendador de
verdad, conviene lo que generaliza.

**Qué se pierde al promediar**, además del orden (`"this is cool"` e `"is this cool"` dan
similitud 1,0):

- **negación y composición** — ninguna operación conmutativa puede expresar que una palabra
  modifique a otra, así que `"no es terror"` y `"es terror"` quedan casi en el mismo punto;
- **dilución** — los documentos largos tienden al centroide del corpus y se parecen entre sí
  por ser largos, no por el tema;
- **hubness** — las similitudes se comprimen hacia arriba y aparecen documentos que son
  vecinos de casi todos.

Se usa igual porque no requiere entrenamiento, es O(n), no tiene hiperparámetros y es la
línea base honesta contra la cual medir algo más sofisticado.

Ponderar por IDF **no mejora de forma uniforme**: sube el MRR de FastText (+0,025), no
cambia el de Word2Vec (−0,004) y empeora el del SBW (−0,018). Ayuda cuando el modelo de
palabras es pobre y necesita que le indiquen dónde mirar.

---

## El preprocesamiento pertenece al modelo

Es la decisión de diseño que atraviesa las dos unidades. El corpus guarda **un solo texto**:
el crudo.

La regla para decidir dónde va cada transformación:

> ¿Podría cambiar la respuesta de algún modelo razonable?
>
> **No** (colapsar espacios, `&nbsp;`, saltos de línea) → es **formato**, va en el corpus.
> **Sí** (minúsculas, puntuación, stopwords, stemming) → es **lingüístico**, va en el modelo.

Cada modelo necesita algo distinto. **TF-IDF** trata al documento como un multiconjunto de
términos: el orden y las palabras funcionales son ruido, y normalizar agresivamente aumenta
la señal. Un **modelo de oración** lo trata como una secuencia: fue entrenado sobre texto
natural, así que darle texto en minúsculas y sin stopwords lo deja fuera de dominio.

Las dos versiones existen como **vistas derivadas** del mismo corpus, nunca almacenadas
juntas:

```
corpus.traer_documentos() -> list[Documento]        una sola version: la cruda
     |
     +-- [d.texto for d in docs] ------------------> CRUDA      -> modelo de oracion
     +-- preprocesamiento.tokenizar_corpus(textos) -> TOKENIZADA -> TF-IDF
```

`Documento` **no** tiene un campo `.tokens`, y es deliberado: no existe *la* tokenización
sino *la que quiere cada modelo*. El día que se agregue un LDA con otra lista de stopwords,
ese atributo habría que partirlo en `.tokens_tfidf` y `.tokens_lda` — que un atributo
necesite apellidarse con el nombre del modelo es la señal de que pertenece al modelo.

`tokenizar()` recibe **`str`, no `Documento`**, con lo cual `preprocesamiento.py` no importa
`corpus.py` nunca y se puede testear con literales, sin base de datos.

Para verlo en funcionamiento:

```bash
python src/preprocesamiento.py --demo
```

Cada decisión de tokenización es una **pérdida deliberada**: bajar a minúsculas pierde las
entidades nombradas; quitar stopwords pierde las negaciones (`no` está en la lista); el
patrón de tokens descarta la puntuación y con ella los límites de oración. **Las tildes se
conservan a propósito**: en castellano distinguen palabras (`papa`/`papá`, `esta`/`está`),
a diferencia del inglés.

---

## Cómo funciona: las etapas del scraper

Todo vive en [`src/scraper.py`](src/scraper.py), dividido en secciones con el mismo nombre
que se usa acá. El programa corre en **tres fases**, y la separación entre las dos primeras
es la decisión de diseño más importante del trabajo.

### Fase A — Descubrimiento de las URL a visitar

*Recorre los listados de cada categoría y arma la lista de libros a extraer.*

Abre `/genero/<slug>/page/N/` con Playwright, analiza el HTML con BeautifulSoup y saca las
URL de las fichas. Pagina hasta juntar 50 libros nuevos por categoría.

**Acá ocurre la deduplicación**, antes de gastar un solo request en una ficha. Las
categorías se solapan: en la corrida real, 16 libros aparecían en más de una, lo que evitó
18 visitas repetidas. Recorrer un listado es barato (trae ~24 libros de una sola vez);
visitar una ficha es caro (un request por libro). Conviene gastar los requests baratos
primero. Resultado: `{ slug -> [categorías] }`.

### Fase B — Extracción de las fichas

*Visita cada ficha una sola vez y extrae los metadatos y la sinopsis.*

Por cada libro pendiente: entra a la ficha, la parsea y **guarda el registro al instante**
en `data/libros_parcial.jsonl`, una línea por libro. Escribir es un append puro, así que si
el proceso muere en el libro 150, los 149 anteriores quedan intactos y la próxima corrida
los saltea.

Un libro que falla no corta la corrida: se informa y se sigue con el siguiente. Ante errores
de red se reintenta tres veces con espera creciente. Entre visita y visita hay una pausa
aleatoria de 1 a 2 segundos.

### Fase C — Consolidación y control de calidad

*Convierte el archivo de trabajo en el CSV entregable y lo audita.*

No toca internet ni contenido: solo deja el archivo estructuralmente correcto (columnas en
orden, tipos coherentes, sin duplicados, ausentes uniformes) y corre los controles de
calidad. Después imprime la estructura del dataset y devuelve código de salida distinto de
cero si algo falla.

### Funciones de apoyo

| Sección del código | Qué hace |
|---|---|
| **Configuración** | Dominio, categorías, cupos, pausas y rutas. Todo lo que se toca para cambiar el comportamiento está acá arriba |
| **Reconstrucción de URLs** | Arma y desarma las URL completas a partir de los prefijos constantes |
| **Utilidades de limpieza** | Colapsa espacios y serializa los campos multivaluados como array JSON |
| **Parser del listado** | `parse_listado(html)` → las URL de las fichas de una página de categoría |
| **Parser de la ficha** | `parse_ficha(html, url)` → el registro completo de un libro |
| **Capa de red** | Abre el navegador, bloquea imágenes y CSS, y maneja los reintentos |

Los dos parsers son **funciones puras**: reciben HTML y devuelven datos, sin red adentro. Por
eso se pueden probar contra archivos guardados, en menos de un segundo y sin castigar al
sitio.

> **Por qué son dos parsers y no uno:** la ficha individual también contiene
> `<article class="card">` — son los 12 libros relacionados del lateral, exactamente la
> misma etiqueta que usa el listado. Un parser único habría producido un CSV lleno de datos
> plausibles pero del libro equivocado.

## Modelo de datos

`data/libros.csv` — UTF-8 **sin BOM**, delimitador coma, quoting según RFC 4180, header en
la primera fila.

| # | Variable | Tipo | Descripción | Uso previsto |
|---|---|---|---|---|
| 1 | `id` | entero | Identificador propio de Lectulandia (`postid-N` del `<body>`) | Clave primaria. Deduplicación y joins |
| 2 | `titulo` | texto | Título del libro, tal como lo publica el sitio | Texto para el recomendador; visualización |
| 3 | `titulo_serie` | texto | **Derivada**: título con la saga al lado, separados por guión medio | Solo lectura humana del CSV |
| 4 | `autores` | array JSON | Autor o autores. Siempre array, incluso con uno solo | Filtrado por autor; señal para el recomendador |
| 5 | `n_autores` | entero | **Derivada**: cantidad de autores | Detectar antologías y obras colectivas |
| 6 | `generos` | array JSON | Géneros declarados por el sitio. Más ricos que `categoria_origen` | Etiquetas para clasificación y evaluación |
| 7 | `serie` | texto | Serie a la que pertenece. Vacío si es autoconclusivo | Agrupar sagas; validar el recomendador |
| 8 | `serie_num` | entero o vacío | Número de tomo dentro de la serie | Ordenar una saga |
| 9 | `sinopsis` | texto | Texto completo de la sinopsis, **sin normalizar** | Insumo principal del recomendador (TF-IDF / embeddings) |
| 10 | `url_libro` | texto | Slug de la ficha, **sin** el prefijo constante | Clave de negocio; trazabilidad al origen |
| 11 | `portada` | texto | Ruta de la portada, **sin** el prefijo constante | Mostrar la tapa en una interfaz |
| 12 | `categoria_origen` | array JSON | Categorías semilla desde las que se llegó al libro | Etiqueta de agrupamiento; control de balance |
| 13 | `fecha_extraccion` | fecha ISO | Día en que se obtuvo el registro (`YYYY-MM-DD`) | Trazabilidad y control de frescura |

### Reconstruir las URL completas

Las columnas `url_libro` y `portada` guardan solo la parte que varía. Los prefijos son
constantes y viven en `src/scraper.py`:

```
url completa de la ficha   = "https://ww3.lectulandia.co/book/"    + url_libro
url completa de la portada = "https://assets.lectulandia.co/b/ab/" + portada
```

Repetir esos prefijos en las 200 filas sería almacenar el mismo dato 200 veces. Tenerlos
aislados en una constante permite además corregirlos en un solo lugar si el sitio cambiara
de dominio, en lugar de reescribir el dataset entero.

### Campos multivaluados

`autores`, `generos` y `categoria_origen` se guardan como **array JSON válido** dentro de la
celda:

```
["Intriga","Novela"]     varios valores
["Intriga"]              uno solo, igual va como array
[]                       ninguno
```

No se usa un separador propio (coma, punto y coma, barra) porque el CSV es solo el formato
de transporte: los datos se cargan después a PostgreSQL mediante un ETL, donde la celda se
castea directo con `::jsonb` sin tener que parsear delimitadores ni preocuparse por que el
separador aparezca dentro de un valor.

```sql
SELECT generos::jsonb FROM libros;
```

Que la celda contenga comillas dobles no es un problema: el escritor de CSV la quotea según
RFC 4180 (envuelve el campo y duplica las comillas internas). El archivo nunca se arma
concatenando strings a mano.

### Valores ausentes

Los ausentes se representan **siempre igual**: celda vacía. Nunca `NULL`, `N/A`, `None` ni
una mezcla. En el ETL, `serie_num` se convierte con `NULLIF(serie_num,'')::smallint`.

### Sobre la escalabilidad a millones de registros

El diseño está dimensionado para ~200 libros y la arquitectura escala, pero las
implementaciones concretas no. Vale dejarlo explícito:

- **Los campos multivaluados en una celda son un compromiso del formato CSV**, que no tiene
  tipo lista. A escala real la respuesta no es cambiar de separador sino **cambiar de
  formato**: Parquet con `list<string>` nativo, o una tabla normalizada `libro_genero` con
  índice invertido. El array JSON es lo que mejor sobrevive el viaje hasta PostgreSQL, donde
  `jsonb` sí es un tipo indexable (GIN).
- **La deduplicación vive en memoria** (`dict` de slugs). A 2M registros son cientos de MB y
  conviene moverla a un índice `UNIQUE` en la base.
- **El cuello de botella es la latencia de red**, no el parseo. Escalar significa pasar de
  un proceso secuencial a una cola de trabajo con varios workers — y, a esa escala, negociar
  un acceso masivo en vez de scrapear.

Lo que **sí** escala tal como está: la separación entre descubrimiento y extracción (es la
arquitectura de un crawler real: frontier + fetchers), el guardado incremental con
reanudación por clave, y los parsers, que son funciones puras `HTML -> dict` sin red adentro
y por lo tanto paralelizables sin cambiarles una línea.

## Estructura del proyecto

```
README.md                       este archivo
requirements.txt                dependencias de Python
docker-compose.yml              PostgreSQL 17 en Docker
.env.example                    plantilla de conexión (copiar a .env)

src/
  scraper.py                    UNIDAD 1: extracción (Fases A, B y C)
  db.py                         conexión a PostgreSQL desde el entorno
  etl.py                        CSV -> staging -> tabla tipada + verificación
  corpus.py                     PostgreSQL -> list[Documento]   (texto CRUDO)
  preprocesamiento.py           tokenizar()                     (TOKENIZADO)
  experimentos.py               las cuatro preguntas, medidas
  embeddings.py                 Word2Vec/FastText propios contra SBW

sql/
  01_esquema.sql                DDL: tipos, constraints, índices GIN
  02_consultas.sql              verificación y exploración desde DBeaver

data/
  libros.csv                    ENTREGABLE: el dataset de 200 libros
  libros_ampliado.csv           corpus de 1700 libros, solo para embeddings
  *_parcial.jsonl               buffers de trabajo (ignorados por git)
  modelos/                      embeddings entrenados (ignorados por git)

docs/
  diseno_extraccion.md          análisis previo de la Unidad 1 (Parte 1)

tests/
  test_parsers.py               pruebas de los parsers, sin red ni base
  fixtures/                     HTML de muestra, versionado
```

**Fuera del repositorio**, en la carpeta que lo contiene:

```
SBW-vectors-300-min5.bin.gz     modelo pre-entrenado (1,07 GB)
```

## Principales dificultades encontradas

### Unidad 2

**Cada conexión a la base tardaba 130 segundos.** El síntoma era desconcertante porque todo
*funcionaba*, solo que absurdamente lento, y el consumo de CPU era cero: estaba esperando
red. La causa: el puerto se publica atado a `127.0.0.1` (IPv4) para no exponer la base, pero
`localhost` en Windows resuelve primero a `::1` (IPv6), donde no escucha nadie, y cada
conexión aguardaba el *timeout* de TCP antes de caer a IPv4. Con `127.0.0.1` explícito son
0,02 segundos. **Un factor de 6500.**

**`COPY FROM '/ruta/archivo'` no encuentra el CSV.** Es el error más natural de cometer:
esa forma de `COPY` lee un archivo *del servidor*, que dentro de Docker es el filesystem del
contenedor, donde el CSV no existe. Hay que usar `COPY ... FROM STDIN` y transmitir desde el
cliente, que es lo que hace `cursor.copy()` de psycopg3.

**Un `CHECK (serie_num > 0)` habría roto la carga.** Es la constraint que uno escribe por
reflejo, y hay 4 libros con `serie_num = 0`: son omnibus y precuelas (*Dune: La saga
completa*, *El Archivo de las Tormentas 1-5*). Va `>= 0`. Se detectó antes de escribir el
DDL, midiendo los datos reales en vez de asumirlos.

**La validación del scraper bloqueaba el corpus ampliado.** `validar()` aplicaba siempre el
control *"cantidad entre 50 y 200"*, que es un requisito del entregable de la Unidad 1 y no
una propiedad de cualquier corpus: el scrapeo de 1700 libros terminaba con código de error
pese a estar perfecto. El rango pasó a ser un parámetro.

**El experimento de estabilidad estaba mal diseñado al principio.** Los documentos sonda
participaban del `fit` del vectorizador, así que con un fondo de 10 documentos las 20
sondas eran dos tercios del corpus y **definían ellas mismas el IDF**. Como son las mismas
en todas las réplicas, eso inflaba artificialmente la coincidencia justo en los tamaños
donde el experimento tenía que mostrar inestabilidad: la curva bajaba y subía sin sentido.
Corregido a `fit()` sobre el fondo y `transform()` sobre las sondas, quedó monótona. En el
camino apareció otro error: la detección de meseta aceptaba saltos **negativos** como
estabilización, y un descenso es ruido, no convergencia.

**Cifras escritas a mano que dejaron de ser ciertas.** El experimento imprimía *"el 63% del
vocabulario"*, correcto para el corpus de 200 y falso para el ampliado (51%). Ahora se
calcula sobre el corpus en uso. Al corregirlo salió que la cifra real del entregable era
65%, no 63%: la medición original contaba tipos sin pasar por el tokenizador.

**pandas escribía el CSV con finales de línea distintos según el sistema operativo.** Cada
integrante que regenerara `libros.csv` producía un diff de 200 líneas sin haber cambiado un
solo dato. Se fijó `lineterminator="\n"` para que la salida sea idéntica byte a byte en
cualquier máquina.

### Unidad 1

**La ficha individual contiene tarjetas de otros libros.** Cada ficha trae 12
`<article class="card">` con libros relacionados en el lateral — exactamente la misma
etiqueta que usa el listado de categorías. Reutilizar un solo parser para ambas páginas
habría producido un CSV lleno de datos plausibles pero del libro equivocado. Por eso hay dos
parsers separados y un test que verifica justamente eso. El mismo problema aplica al `id`:
los `<article id="post-N">` son los relacionados, así que el id real se toma de la clase del
`<body>`.

**Etiquetas inline y de bloque mezcladas en la sinopsis.** Un `get_text(separator=" ")`
global producía `"víctima perfecta ."` con un espacio antes del punto, porque no distingue
un `<b>` (que no debe separar) de un `<br>` (que sí). Se resolvió reemplazando solo los
`<br>` antes de extraer el texto.

**Títulos que ya contienen el nombre de la saga.** La revista *Más Allá* tiene el libro
titulado "Más Allá 21" dentro de la serie "Más Allá", con lo cual la columna derivada
`titulo_serie` producía `"Más Allá 21 - Más Allá 21"`. Se agregó una comprobación para no
repetir la saga cuando el título ya la incluye.

**El campo `serie` no existe en el HTML** cuando el libro es autoconclusivo: el `<div>`
directamente no está, en lugar de estar vacío. Todo el acceso a los campos es defensivo e
independiente, de modo que un libro con estructura distinta devuelva ese campo vacío en vez
de hacer fallar la corrida entera.

**Errores intermitentes del servidor.** Durante la corrida real apareció un HTTP 503. Los
reintentos con espera creciente lo resolvieron sin perder el libro ni cortar la extracción.

**El `categoria_origen` podía quedar desactualizado al reanudar.** Es el único campo que no
sale de la ficha sino de la Fase A, así que depende de qué categorías y qué cupo se usaron.
Un libro guardado con `["terror"]` puede pertenecer también a `["terror","fantastico"]` si
se amplía el cupo. Se agregó una sincronización al reanudar, porque un dato silenciosamente
desactualizado es peor que un error visible.

**El CSV no lleva BOM.** La primera intención fue usar `utf-8-sig` para que el archivo
abriera cómodo en Excel, pero como el destino real es un `COPY` a PostgreSQL, el BOM se
colaría en el nombre de la primera columna. Gana el destino real.

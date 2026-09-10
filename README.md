# Procesamiento del Lenguaje Natural — TUIA

## Unidad 1 — Extracción de metadatos y sinopsis con Playwright y BeautifulSoup

Construcción de un corpus de libros a partir de la información pública de
[Lectulandia](https://ww3.lectulandia.co). El dataset resultante se reutiliza en la Unidad 2
para actividades de procesamiento de texto y para construir un recomendador de libros.

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

## Instalación

Requiere **Python 3.10 o superior**. Desde la raíz del proyecto:

```bash
python -m venv venv
```

```bash
venv/Scripts/activate
```

```bash
pip install -r requirements.txt
```

Playwright necesita además descargar su propio Chromium (~150 MB, una sola vez):

```bash
python -m playwright install chromium
```

> En Linux o macOS el activador es `source venv/bin/activate`.

## Ejecución

Corrida completa (Fase A + B + C). Tarda unos 8–9 minutos, casi todo esperando las pausas
entre requests:

```bash
python src/scraper.py
```

Genera `data/libros.csv` e imprime la estructura del dataset y los controles de calidad.
Devuelve código de salida distinto de cero si algún control falla.

### Opciones útiles

| Comando | Para qué sirve |
|---|---|
| `python src/scraper.py --max-libros 5` | Prueba rápida: visita solo 5 fichas |
| `python src/scraper.py --solo-descubrir` | Corre solo la Fase A, sin visitar fichas |
| `python src/scraper.py --solo-consolidar` | Regenera el CSV sin usar internet |
| `python src/scraper.py --cupo 25` | Cambia cuántos libros aporta cada categoría |
| `python src/scraper.py --categorias terror poesia` | Usa otras categorías |
| `python src/scraper.py --ver-navegador` | Muestra la ventana de Chromium |

El navegador corre **headless** (sin ventana) por defecto.

### Reanudación

La extracción guarda cada libro apenas lo obtiene, en `data/libros_parcial.jsonl`. Si la
corrida se interrumpe, volver a ejecutar el comando retoma donde había quedado sin
revisitar lo ya extraído.

Para forzar una extracción desde cero, borrar ese archivo:

```bash
rm data/libros_parcial.jsonl
```

### Tests

Los parsers se prueban contra HTML guardado en disco, sin tocar la red (corre en menos de
un segundo):

```bash
python tests/test_parsers.py
```

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
README.md                     este archivo
requirements.txt              dependencias de Python
src/
  scraper.py                  extracción completa (Fases A, B y C)
data/
  libros.csv                  ENTREGABLE: el dataset
  libros_parcial.jsonl        buffer de trabajo (ignorado por git)
docs/
  diseno_extraccion.md        análisis previo (Parte 1)
tests/
  test_parsers.py             pruebas de los parsers, sin red
  fixtures/                   HTML de muestra (ignorado por git)
```

## Principales dificultades encontradas

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

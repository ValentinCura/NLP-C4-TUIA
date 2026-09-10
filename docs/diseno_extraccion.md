# Diseño de la extracción — Parte 1

Unidad 1, Procesamiento del Lenguaje Natural (TUIA — FCEIA/UNR).

---

## 1. Categorías seleccionadas

El sitio es **Lectulandia**, en la URL que indica el enunciado del trabajo práctico:
`https://ww3.lectulandia.co`.

> Esa URL está aislada en la constante `BASE_URL` de `src/scraper.py`, de modo que si el
> sitio cambiara de dominio se corrija en un solo lugar.

La cátedra habilitó usar más de una categoría. Se eligieron **cuatro**:

| Categoría | URL | Páginas disponibles | Cupo |
|---|---|---|---|
| Ciencia ficción | `/genero/ciencia-ficcion/` | 182 | 50 |
| Fantástico | `/genero/fantastico/` | 225 | 50 |
| Terror | `/genero/terror/` | 102 | 50 |
| Histórico | `/genero/historico/` | 285 | 50 |

**Total objetivo: 200 libros.**

### Criterio de selección de las categorías

La elección no es arbitraria: el dataset se reutiliza en la Unidad 2 para construir un
recomendador basado en similitud de texto, y el conjunto de categorías determina qué tan
exigente es esa tarea.

Se buscó deliberadamente una mezcla de distancias semánticas:

- **Pares cercanos** — ciencia ficción y fantástico comparten vocabulario especulativo
  (mundos, criaturas, poderes, futuro). Un recomendador que los separe bien está
  capturando señal real, no ruido.
- **Par lejano** — histórico usa vocabulario de épocas, guerras y lugares reales, muy
  distinto del resto.
- **Intermedio** — terror se solapa emocionalmente con fantástico pero tiene léxico propio.

Se descartaron alternativas por motivos concretos:

- **Intriga** (380 páginas, la más grande) está muy solapada con "Novela" y con thriller
  genérico: las sinopsis resultan poco diferenciables entre sí.
- **Poesía** (41 páginas): riesgo alto de fichas con sinopsis corta o ausente, y la
  consigna exige que la mayoría de los registros tenga sinopsis.

### Criterio de selección de las páginas

Se recorren las páginas del listado **en orden**, desde la página 1, hasta completar el
cupo de 50 libros nuevos por categoría. No se saltean ni se eligen páginas al azar: el
orden del sitio es estable, de modo que dos corridas producen el mismo dataset.

El listado trae entre 20 y 24 libros por página según la categoría, así que el cupo se
cubre en 2–3 páginas. El código **no asume un número fijo por página**: pagina hasta juntar
el cupo o hasta agotar las páginas disponibles.

---

## 2. Datos que se extraen

| Campo | Descripción |
|---|---|
| `id` | Identificador propio de Lectulandia |
| `titulo` | Título del libro |
| `titulo_serie` | Derivado: título con la saga al lado, separados por guión medio |
| `autores` | Autor o autores (array JSON) |
| `n_autores` | Derivado: cantidad de autores |
| `generos` | Género o géneros que declara el sitio (array JSON) |
| `serie` | Serie a la que pertenece, si corresponde |
| `serie_num` | Número de tomo dentro de la serie |
| `sinopsis` | Texto completo de la sinopsis |
| `url_libro` | Dirección de la ficha (solo la parte variable) |
| `portada` | Dirección de la portada (solo la parte variable) — campo opcional |
| `categoria_origen` | Categorías semilla desde las que se llegó al libro (array JSON) |
| `fecha_extraccion` | Fecha en que se obtuvo el registro |

El diccionario completo, con tipos y uso previsto de cada variable, está en el
[README](../README.md#modelo-de-datos).

### Decisiones de modelado

**`id` como identificador.** Todos los libros tienen un id propio del sitio, publicado por
WordPress en la clase del `<body>` (`postid-125510`). Se usa como clave primaria natural.

> Trampa: los `<article id="post-N">` que aparecen en una ficha son los **libros
> relacionados** del lateral. Tomar ese id devolvería el identificador de otro libro.

**Campos multivaluados como array JSON.** `autores`, `generos` y `categoria_origen` se
guardan como `["Intriga","Novela"]` dentro de la celda. No se usa un separador propio
(coma, punto y coma, barra) porque el destino de los datos es una carga a PostgreSQL por
ETL, donde la celda se castea directo con `::jsonb` sin parsear delimitadores. Que la celda
contenga comillas dobles no es problema: pandas la quotea según RFC 4180.

**Prefijos constantes fuera del CSV.** Todas las fichas empiezan con
`https://ww3.lectulandia.co/book/` y todas las portadas con
`https://assets.lectulandia.co/b/ab/` — verificado sobre 189 portadas de 6 géneros
distintos. Guardar solo la parte variable evita repetir el mismo dato 200 veces. El recorte
**no es a ciegas**: si una URL no empieza con el prefijo esperado, el scraper lanza un error
explícito en vez de guardar una URL cortada por la mitad.

**`serie` en columna propia, no concatenada al título.** Se midió: el **53%** de los libros
pertenece a una serie, así que no es un campo mayoritariamente vacío. Mantenerlo aparte
permite filtrar por saga sin parsear strings y conserva una señal fuerte para el
recomendador (los libros de una misma saga deberían ser las primeras recomendaciones).
Para la legibilidad que pedía la cátedra se agregó `titulo_serie` como columna **derivada**,
que sí concatena — sin destruir las columnas atómicas.

**No se normaliza el texto.** No se pasa a minúsculas, no se quitan acentos ni stopwords,
no se unifican variantes de nombres de género. Eso corresponde al ETL y a la Unidad 2.
Hacerlo en el scraper destruiría información de forma irreversible. La única limpieza
aplicada es de **formato**: colapsar espacios, tabs y saltos de línea.

---

## 3. Localización de los datos

Los selectores se obtuvieron inspeccionando el HTML del sitio con las herramientas de
desarrollo del navegador, y están verificados por los tests de `tests/test_parsers.py`.

### Ficha individual — `/book/<slug>/`

| Dato | Tipo de página | Etiqueta HTML | Selector |
|---|---|---|---|
| Id | Ficha individual | `<body class="... postid-N">` | clase que empieza con `postid-` |
| Título | Ficha individual | `<h1>` dentro de `<div id="title">` | `#title h1` |
| Autores | Ficha individual | `<a class="dinSource">` dentro de `<div id="autor">` | `#autor a.dinSource` |
| Géneros | Ficha individual | `<a class="dinSource">` dentro de `<div id="genero">` | `#genero a.dinSource` |
| Serie | Ficha individual | `<a class="dinSource">` dentro de `<div id="serie">` | `#serie a.dinSource` |
| Nº de tomo | Ficha individual | `<span class="tagTitle">` (`"Libro 1 de: "`) | `#serie span.tagTitle` + regex `\d+` |
| Sinopsis | Ficha individual | `<div id="sinopsis">` | `#sinopsis` |
| Portada | Ficha individual | `<img>` dentro de `<div id="cover">` | `#cover img[src]` |

### Listado de categoría — `/genero/<slug>/page/<n>/`

| Dato | Tipo de página | Etiqueta HTML | Selector |
|---|---|---|---|
| Tarjeta de libro | Listado | `<article class="card">` | `article.card` |
| URL de la ficha | Listado | `<a class="title" href="...">` | `article.card a.title[href]` |

### Dos detalles que importan

**El `<div id="serie">` no existe** cuando el libro no pertenece a una serie: no está vacío,
directamente no está en el HTML. Por eso el acceso a cada campo es defensivo e
independiente.

**La sinopsis mezcla etiquetas inline y de bloque.** Adentro hay un `<b>` con la frase gancho
y varios `<br>` separando párrafos, y hay que tratarlos distinto:

- `<br>` separa párrafos → **sí** debe generar un espacio
- `<b>` resalta texto → **no** debe generar un espacio

Un `get_text(separator=" ")` global no los distingue y, como el HTML real es
`<b>...víctima perfecta</b>.<br>Sydney`, produce `"perfecta ."` con un espacio antes del
punto. La solución es reemplazar solo los `<br>` por un salto y recién después extraer el
texto sin separador.

---

## 4. Estrategia de extracción

El scraper corre en **tres fases separadas**. La separación entre las dos primeras es la
decisión de eficiencia central del diseño.

```
FASE A — Descubrimiento          ~12 requests
  Para cada categoría semilla:
    Abrir /genero/<slug>/page/N/ con Playwright
    Analizar el HTML con BeautifulSoup
    Extraer las URL de las fichas (article.card a.title)
    Paginar hasta cubrir el cupo
  Acumular en { slug_del_libro -> [categorías] }
  >>> La deduplicación ocurre ACÁ, antes de visitar una sola ficha

FASE B — Extracción              ~200 requests
  Para cada slug único:
    Si ya está en libros_parcial.jsonl -> saltear (reanudable)
    Visitar la ficha con Playwright
    Extraer metadatos y sinopsis con BeautifulSoup
    Append inmediato al .jsonl + flush
    Pausa aleatoria entre 1 y 2 segundos

FASE C — Consolidación           0 requests
  jsonl -> pandas -> tipos, orden de columnas, dedup -> data/libros.csv
  Reporte de estructura (df.info()) y controles de calidad
```

### Por qué deduplicar antes y no después

Las categorías **se solapan**: un mismo libro puede figurar en ciencia ficción y en
fantástico a la vez. En la corrida real, 16 de los 200 libros aparecieron en más de una
categoría, lo que evitó **18 visitas** a fichas ya obtenidas. Si la deduplicación ocurriera
después de visitar las fichas, esos requests serían trabajo tirado a la basura.

Recorrer un listado es barato (trae ~24 libros de una sola vez); visitar una ficha es caro
(un request por libro). Conviene gastar los requests baratos primero.

### Otras decisiones de eficiencia

1. **Un solo navegador y una sola pestaña** para toda la corrida. Abrir Chromium cuesta
   cerca de un segundo; hacerlo 200 veces serían más de 3 minutos regalados.
2. **Bloqueo de recursos** con `page.route()`: se abortan `image`, `stylesheet`, `font` y
   `media`. De la portada guardamos la URL, no el archivo, y una página de listado trae 24
   imágenes de 50–200 KB. Reduce el tráfico más de un 80%.
3. **`wait_until="domcontentloaded"`**, no `networkidle`. El sitio es WordPress y sirve el
   contenido ya renderizado en el HTML inicial: no hay que esperar JavaScript.
   `networkidle` además esperaría a los iframes de publicidad.
4. **Guardado incremental en JSONL.** Escribir es un append puro: agregar un libro es
   escribir una línea y hacer `flush()`, sin releer ni reescribir el archivo. Si el proceso
   muere en el libro 150, los 149 anteriores quedan intactos y la próxima corrida los
   saltea.
5. **Reintentos con backoff.** Ante un error se reintenta 3 veces con espera creciente
   (2s, 4s). Durante la corrida real apareció un HTTP 503 y se resolvió solo, sin perder el
   libro.
6. **Pausa aleatoria** de 1 a 2 segundos entre visitas: cortesía con el servidor y evita el
   patrón de tráfico regular que dispararía un rate-limit.

### Limitación explícita

**No se descarga ningún EPUB, PDF ni ningún otro contenido.** El trabajo se limita a los
metadatos y las sinopsis públicas. Las URL de `download.php` presentes en el HTML se
ignoran deliberadamente.

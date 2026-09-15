"""
Extraccion de metadatos y sinopsis de libros desde Lectulandia.

Unidad 1 - Procesamiento del Lenguaje Natural (TUIA, FCEIA-UNR).

El corpus resultante (data/libros.csv) se reutiliza en la Unidad 2 para
procesamiento de texto y para construir un recomendador de libros.

Solo se extraen metadatos y sinopsis publicas. No se descarga ningun EPUB/PDF.
"""

import argparse
import json
import random
import re
import sys
import time
from datetime import date
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

# URL del sitio, tal como la indica el enunciado del trabajo practico.
# Aislada aca para poder cambiarla en un solo lugar si el sitio migrara de
# dominio: si la extraccion deja de funcionar, es lo primero que hay que mirar.
BASE_URL = "https://ww3.lectulandia.co"

# Prefijos constantes que NO se guardan en el CSV.
#
# Todas las fichas empiezan igual y todas las portadas tambien (verificado sobre
# 189 portadas de 6 generos distintos). Repetirlos en cada una de las 200 filas
# es almacenar el mismo dato 200 veces sin aportar informacion.
#
# Se guarda solo la parte variable y la URL completa se reconstruye con
# url_ficha() / url_portada(). El CSV queda mas liviano y mas legible, y si el
# dominio vuelve a cambiar se arregla tocando una constante en vez de reescribir
# las 200 filas del dataset.
PREFIJO_FICHA = "https://ww3.lectulandia.co/book/"
PREFIJO_PORTADA = "https://assets.lectulandia.co/b/ab/"

# Los campos multivaluados (autores, generos, categoria_origen) se guardan como
# un ARRAY JSON valido dentro de la celda, no con un separador propio.
#
# El motivo es el destino de los datos: el CSV es solo transporte y despues se
# carga a PostgreSQL por ETL, donde la celda se castea directo con ::jsonb sin
# tener que partir strings. Un separador (coma, punto y coma, barra) obligaria
# al ETL a parsear a mano y a preocuparse por que el separador no aparezca
# dentro de un valor. El array JSON elimina esa clase de problema entera.
#
# Que la celda contenga comillas dobles no es un problema: pandas la quotea
# segun RFC 4180 (envuelve el campo y duplica las comillas internas). Nunca se
# arma el CSV concatenando strings a mano.
#
# Codificacion UTF-8 SIN BOM: utf-8-sig abriria mas comodo en Excel, pero el BOM
# se colaria en el nombre de la primera columna al hacer COPY en Postgres.
ENCODING_CSV = "utf-8"

# Categorias semilla desde las que se descubren los libros.
#
# La mezcla es deliberada: ciencia-ficcion y fantastico comparten vocabulario
# especulativo (pares CERCANOS) mientras que historico es muy distinto (par
# LEJANO). Asi el recomendador de la Unidad 2 tiene que discriminar de verdad y
# no separa los grupos por casualidad.
CATEGORIAS = ["ciencia-ficcion", "fantastico", "terror", "historico"]

# Libros nuevos que aporta cada categoria (4 x 50 = 200).
CUPO_POR_CATEGORIA = 50

# Tope de seguridad: si el paginado se comportara raro, la corrida termina
# igual en vez de quedarse dando vueltas para siempre.
MAX_PAGINAS = 40

# Rango de la pausa entre paginas visitadas, en segundos.
PAUSA_MIN, PAUSA_MAX = 1.0, 2.0

# Archivos de salida. El parcial es un buffer de trabajo que se regenera solo
# (esta en .gitignore); el CSV es el entregable.
DIR_DATOS = Path(__file__).resolve().parent.parent / "data"
RUTA_PARCIAL = DIR_DATOS / "libros_parcial.jsonl"
RUTA_CSV = DIR_DATOS / "libros.csv"

# Columnas del dataset, en el orden en que van al CSV. El id va primero porque
# es el identificador unico del registro.
COLUMNAS = [
    "id",
    "titulo",
    "titulo_serie",
    "autores",
    "n_autores",
    "generos",
    "serie",
    "serie_num",
    "sinopsis",
    "url_libro",
    "portada",
    "categoria_origen",
    "fecha_extraccion",
]


# ---------------------------------------------------------------------------
# Reconstruccion de URLs
# ---------------------------------------------------------------------------

def url_ficha(slug):
    """Reconstruye la URL completa de una ficha a partir del valor guardado."""
    return PREFIJO_FICHA + slug if slug else ""


def url_portada(sufijo):
    """Reconstruye la URL completa de una portada a partir del valor guardado."""
    return PREFIJO_PORTADA + sufijo if sufijo else ""


def _sin_prefijo(url, prefijo, campo):
    """Quita un prefijo constante, avisando fuerte si la URL no lo tiene.

    Recortar a ciegas seria peligroso: si algun libro usara otro servidor de
    imagenes, un slice fijo cortaria la URL por la mitad y guardaria basura sin
    que nadie se entere. Preferimos enterarnos.
    """
    if not url:
        return ""
    if not url.startswith(prefijo):
        raise ValueError(
            f"{campo}: la URL no empieza con el prefijo esperado.\n"
            f"  esperado: {prefijo}\n  recibido: {url}\n"
            f"Revisar si el sitio cambio de dominio o de servidor de imagenes."
        )
    return url[len(prefijo):]


# ---------------------------------------------------------------------------
# Utilidades de limpieza
# ---------------------------------------------------------------------------

def limpiar(texto):
    """Normaliza el espaciado de un texto extraido del HTML.

    El HTML viene con saltos de linea e indentacion del template, que en un CSV
    resultan molestos. Se colapsa todo espacio en blanco (espacios, tabs,
    saltos) a un unico espacio simple.

    Ojo: esto es limpieza de *formato*, no normalizacion linguistica. No se pasa
    a minusculas ni se quitan acentos ni stopwords: eso es trabajo de la Unidad 2
    y hacerlo aca destruiria informacion de forma irreversible.
    """
    if not texto:
        return ""
    # \xa0 es el espacio no-separable (&nbsp;), muy comun en HTML.
    texto = texto.replace("\xa0", " ")
    return re.sub(r"\s+", " ", texto).strip()


def lista_json(valores):
    """Serializa una lista de valores como array JSON valido para la celda.

        ["Intriga","Novela"]      varios valores
        ["Intriga"]               uno solo, igual va como array
        []                        ninguno

    ensure_ascii=False conserva tildes y enies como caracteres UTF-8 reales en
    vez de escaparlos a \\u00f3. separators sin espacios deja la celda compacta.

    json.dumps se encarga de escapar las comillas dobles que pudiera haber
    dentro de un nombre; despues el escritor de CSV vuelve a escapar esas mismas
    comillas segun RFC 4180. Los dos niveles se destrabolan solos al leer.

    NO se normaliza ni se deduplica: el valor va tal cual viene del scraping.
    Unificar variantes y mapear a generos canonicos es trabajo del ETL.
    """
    limpios = [limpiar(v) for v in valores]
    return json.dumps([v for v in limpios if v], ensure_ascii=False,
                      separators=(",", ":"))


def sopa(html):
    """Construye el arbol de BeautifulSoup.

    Se usa el parser 'lxml' por velocidad. Importante: se le pasa siempre un str
    ya decodificado (el que devuelve Playwright), nunca bytes, porque las
    paginas del sitio no declaran <meta charset> y BeautifulSoup podria adivinar
    mal la codificacion y romper los acentos.
    """
    return BeautifulSoup(html, "lxml")


# ---------------------------------------------------------------------------
# Parser del listado de una categoria: /genero/<slug>/page/<n>/
# ---------------------------------------------------------------------------

def parse_listado(html):
    """Devuelve las URL absolutas de las fichas que aparecen en un listado.

    Cada libro del listado esta en un <article class="card">, y el enlace a la
    ficha es el <a class="title"> que hay adentro.

    Se conserva el orden de aparicion y se eliminan repetidos dentro de la misma
    pagina (dict.fromkeys preserva el orden, a diferencia de set()).
    """
    urls = []
    for card in sopa(html).select("article.card"):
        enlace = card.select_one("a.title[href]")
        if enlace:
            urls.append(urljoin(BASE_URL, enlace["href"]))
    return list(dict.fromkeys(urls))


# ---------------------------------------------------------------------------
# Parser de la ficha individual: /book/<slug>/
# ---------------------------------------------------------------------------

def _extraer_id(s):
    """Obtiene el identificador propio del sitio para este libro.

    WordPress lo publica en la clase del <body>: "... postid-125510 ...".

    NO se usa el id="post-N" de los <article class="card">, porque en una ficha
    esos articles son los libros RELACIONADOS del lateral: tomar el primero
    devolveria el id de otro libro.
    """
    body = s.find("body")
    if not body:
        return ""
    for clase in body.get("class", []):
        if clase.startswith("postid-"):
            return clase[len("postid-"):]
    return ""


def parse_ficha(html, url_completa):
    """Extrae todos los campos de la ficha de un libro.

    Devuelve siempre un dict con las mismas claves. Los campos ausentes se
    representan con cadena vacia "" (nunca None ni NaN), tal como pide la
    consigna: los ausentes tienen que ser consistentes.

    El acceso a cada campo es defensivo e independiente: un libro con estructura
    rara (por ejemplo, sin div de sinopsis) devuelve ese campo vacio en lugar de
    hacer fallar toda la extraccion.
    """
    s = sopa(html)

    # Cuidado: la ficha individual TAMBIEN contiene <article class="card"> con
    # libros relacionados en el lateral. Por eso todos los selectores de abajo
    # apuntan a ids concretos del bloque principal (#title, #autor, ...) y nunca
    # a .card. Mezclar ambos parsers traeria datos del libro equivocado.

    titulo_tag = s.select_one("#title h1")
    titulo = limpiar(titulo_tag.get_text()) if titulo_tag else ""

    # Un libro puede tener varios autores, cada uno en su propio <a>.
    autores = [a.get_text() for a in s.select("#autor a.dinSource")]

    # Los generos del sitio: son mas ricos que categoria_origen, porque incluyen
    # etiquetas que no usamos como semilla (ej. "Novela", "Psicologico").
    generos = [a.get_text() for a in s.select("#genero a.dinSource")]

    # El div #serie directamente NO EXISTE si el libro no pertenece a una serie.
    serie_tag = s.select_one("#serie a.dinSource")
    serie = limpiar(serie_tag.get_text()) if serie_tag else ""

    # El numero de tomo esta en el texto de la etiqueta: "Libro 1 de: ".
    serie_num = ""
    etiqueta_serie = s.select_one("#serie span.tagTitle")
    if etiqueta_serie:
        m = re.search(r"\d+", etiqueta_serie.get_text())
        if m:
            serie_num = m.group()

    # Sinopsis: se toma el div entero, con todo lo que contenga.
    #
    # Adentro conviven dos tipos de etiqueta que hay que tratar distinto:
    #   - <br>  separa parrafos       -> SI tiene que generar un espacio
    #   - <b>   resalta la frase gancho -> NO tiene que generar un espacio
    #
    # Un get_text(separator=" ") global no distingue entre ambas y, como el HTML
    # real es "<b>...victima perfecta</b>.<br>Sydney", produciria "perfecta ."
    # con un espacio antes del punto. Por eso primero se reemplazan solo los
    # <br> por un salto y recien despues se extrae el texto sin separador.
    sinopsis = ""
    sinopsis_tag = s.select_one("#sinopsis")
    if sinopsis_tag:
        for br in sinopsis_tag.find_all("br"):
            br.replace_with("\n")
        sinopsis = sinopsis_tag.get_text()

    # De la portada se guarda la URL, nunca el archivo de imagen.
    portada_tag = s.select_one("#cover img[src]")
    portada = limpiar(portada_tag["src"]) if portada_tag else ""

    return {
        "id": _extraer_id(s),
        "titulo": titulo,
        # Campo derivado, pedido por la catedra: titulo y serie juntos, con un
        # guion medio en el medio, para identificar la saga de un vistazo al
        # mirar el CSV. Es redundante a proposito: se calcula a partir de
        # titulo/serie/serie_num, que siguen existiendo como columnas propias.
        "titulo_serie": _titulo_con_serie(titulo, serie, serie_num),
        "autores": lista_json(autores),
        "n_autores": len([a for a in autores if limpiar(a)]),
        "generos": lista_json(generos),
        "serie": serie,
        "serie_num": serie_num,
        "sinopsis": limpiar(sinopsis),
        "url_libro": _sin_prefijo(url_completa, PREFIJO_FICHA, "url_libro"),
        "portada": _sin_prefijo(portada, PREFIJO_PORTADA, "portada"),
    }


def _titulo_con_serie(titulo, serie, serie_num):
    """'El silencio del bosque' + 'Steinbeck y Reed' 1 -> con la saga al lado.

    Si el libro no pertenece a ninguna serie, devuelve el titulo tal cual.
    """
    if not serie:
        return titulo

    # Hay titulos que YA incluyen el nombre de la saga: la revista "Mas Alla"
    # tiene el libro titulado "Mas Alla 21" dentro de la serie "Mas Alla".
    # Concatenar a ciegas daria "Mas Alla 21 - Mas Alla 21".
    if serie.casefold() in titulo.casefold():
        return titulo

    saga = f"{serie} {serie_num}" if serie_num else serie
    return f"{titulo} - {saga}"


# ---------------------------------------------------------------------------
# Capa de red (Playwright)
# ---------------------------------------------------------------------------

# Tipos de recurso que se abortan antes de descargarse.
#
# Solo necesitamos el HTML: de la portada guardamos la URL, no la imagen. Las
# portadas pesan 50-200 KB cada una y una pagina de listado trae 24. Bloquearlas
# reduce el trafico mas de un 80% y acelera muchisimo cada visita.
RECURSOS_BLOQUEADOS = {"image", "stylesheet", "font", "media"}

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def nueva_pagina(navegador):
    """Crea la unica pestania que se reutiliza durante toda la corrida.

    Abrir un navegador cuesta cerca de un segundo: hacerlo una vez por libro
    serian minutos regalados. Se abre una sola vez y se navega con ella.
    """
    contexto = navegador.new_context(user_agent=USER_AGENT)
    pagina = contexto.new_page()

    def filtrar(ruta):
        if ruta.request.resource_type in RECURSOS_BLOQUEADOS:
            ruta.abort()
        else:
            ruta.continue_()

    pagina.route("**/*", filtrar)
    return pagina


def get_html(pagina, url, intentos=3):
    """Navega a una URL y devuelve su HTML, reintentando ante fallos puntuales.

    wait_until="domcontentloaded" en vez de "networkidle": el sitio es WordPress
    y sirve el contenido ya renderizado en el HTML inicial, asi que no hay que
    esperar a que corra JavaScript. "networkidle" ademas esperaria a los iframes
    de publicidad, que son lentos y no aportan nada.

    Ante error se reintenta con espera creciente (backoff). Si se agotan los
    intentos devuelve None: el que llama decide si saltear ese libro y seguir.
    """
    for intento in range(1, intentos + 1):
        try:
            respuesta = pagina.goto(url, wait_until="domcontentloaded",
                                    timeout=30_000)
            if respuesta and respuesta.status >= 400:
                raise RuntimeError(f"HTTP {respuesta.status}")
            return pagina.content()
        except Exception as e:
            if intento == intentos:
                print(f"    ERROR definitivo en {url}: {e}")
                return None
            espera = 2 ** intento
            print(f"    intento {intento}/{intentos} fallo ({e}); "
                  f"reintento en {espera}s")
            time.sleep(espera)
    return None


def pausa():
    """Espera aleatoria entre visitas.

    Es cortesia con el servidor y ademas evita el patron de trafico regular que
    disparia un rate-limit. El rango sale de PAUSA_MIN/PAUSA_MAX.
    """
    time.sleep(random.uniform(PAUSA_MIN, PAUSA_MAX))


# ---------------------------------------------------------------------------
# FASE A - Descubrimiento de las URL a visitar
# ---------------------------------------------------------------------------

def url_categoria(categoria, pagina_num):
    """URL de una pagina del listado de una categoria.

    La pagina 1 no lleva sufijo /page/1/: el sitio la sirve en /genero/<slug>/.
    """
    if pagina_num == 1:
        return f"{BASE_URL}/genero/{categoria}/"
    return f"{BASE_URL}/genero/{categoria}/page/{pagina_num}/"


def descubrir_urls(pagina, categorias, cupo):
    """Recorre los listados y devuelve {slug_del_libro: [categorias_origen]}.

    Esta es la decision de eficiencia central del scraper. Recorrer los listados
    es barato (una pagina trae ~24 libros de una) y visitar cada ficha es caro
    (un request por libro). Al deduplicar ACA, antes de tocar una sola ficha, no
    se gasta ni un request en libros que ya teniamos.

    Importa porque las categorias se solapan: un mismo libro puede figurar en
    ciencia-ficcion y en fantastico a la vez. Si se deduplicara despues de
    visitar las fichas, ese solapamiento serian requests tirados a la basura.

    Se usa dict (no set) para conservar el orden de descubrimiento, de modo que
    dos corridas sobre el mismo sitio produzcan el mismo dataset.
    """
    encontrados = {}

    for categoria in categorias:
        print(f"\n[{categoria}] objetivo: {cupo} libros")
        nuevos_en_categoria = 0
        pagina_num = 1

        while nuevos_en_categoria < cupo and pagina_num <= MAX_PAGINAS:
            url = url_categoria(categoria, pagina_num)
            html = get_html(pagina, url)

            if html is None:
                print(f"  pagina {pagina_num}: sin respuesta, se corta aca")
                break

            urls = parse_listado(html)
            if not urls:
                print(f"  pagina {pagina_num}: sin libros, fin de la categoria")
                break

            repetidos = 0
            for url_libro in urls:
                if nuevos_en_categoria >= cupo:
                    break
                slug = _sin_prefijo(url_libro, PREFIJO_FICHA, "url_libro")

                if slug in encontrados:
                    # Ya lo trajo otra categoria: no se vuelve a contar para el
                    # cupo, pero si se registra que tambien pertenece a esta.
                    if categoria not in encontrados[slug]:
                        encontrados[slug].append(categoria)
                        repetidos += 1
                else:
                    encontrados[slug] = [categoria]
                    nuevos_en_categoria += 1

            print(f"  pagina {pagina_num}: {len(urls)} libros -> "
                  f"{nuevos_en_categoria}/{cupo} nuevos"
                  + (f", {repetidos} ya vistos en otra categoria"
                     if repetidos else ""))

            pagina_num += 1
            if nuevos_en_categoria < cupo:
                pausa()

        if nuevos_en_categoria < cupo:
            print(f"  AVISO: solo se juntaron {nuevos_en_categoria} de {cupo}")

    return encontrados


# ---------------------------------------------------------------------------
# FASE B - Extraccion de las fichas
# ---------------------------------------------------------------------------

def cargar_parcial(ruta):
    """Lee lo ya extraido en corridas anteriores: {slug: registro}.

    Si una linea quedo cortada por la mitad (el proceso murio justo mientras
    escribia), se descarta esa linea y se sigue: el libro simplemente se vuelve
    a visitar.
    """
    ya_extraidos = {}
    if not ruta.exists():
        return ya_extraidos

    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea:
            continue
        try:
            registro = json.loads(linea)
            ya_extraidos[registro["url_libro"]] = registro
        except (json.JSONDecodeError, KeyError):
            print("  aviso: se descarta una linea corrupta del parcial")
    return ya_extraidos


def _sincronizar_categorias(ya_extraidos, encontrados, ruta_parcial):
    """Actualiza categoria_origen de los libros guardados en corridas previas.

    categoria_origen es el unico campo que no sale de la ficha, sino de la Fase
    A: depende de que categorias y que cupo se hayan usado. Un libro guardado
    ayer con ["terror"] puede pertenecer hoy tambien a ["terror","fantastico"]
    si se amplio el cupo.

    Sin esto, reanudar una corrida dejaria ese campo desactualizado en silencio,
    que es la peor forma de tener un dato mal. Como el archivo son unos cientos
    de lineas, reescribirlo entero es instantaneo.
    """
    desactualizados = 0
    for slug, registro in ya_extraidos.items():
        if slug not in encontrados:
            continue
        actual = lista_json(encontrados[slug])
        if registro.get("categoria_origen") != actual:
            registro["categoria_origen"] = actual
            desactualizados += 1

    if desactualizados:
        print(f"  se actualizo categoria_origen en {desactualizados} registros "
              f"ya guardados")
        with ruta_parcial.open("w", encoding="utf-8") as archivo:
            for registro in ya_extraidos.values():
                archivo.write(json.dumps(registro, ensure_ascii=False) + "\n")


def extraer_libros(pagina, encontrados, ruta_parcial, max_libros=None):
    """Visita cada ficha pendiente y va guardando los registros uno por uno.

    Se guarda en JSONL (un objeto JSON por linea) y no directamente en CSV
    porque escribir es un append puro: agregar un libro es escribir una linea,
    sin releer ni reescribir el archivo entero. Si el proceso muere en el libro
    150, los 149 anteriores quedan intactos y la proxima corrida los saltea.

    Un libro que falla no corta la corrida: se informa y se sigue con el
    siguiente. Al final se reporta cuantos fallaron.
    """
    ya_extraidos = cargar_parcial(ruta_parcial)
    _sincronizar_categorias(ya_extraidos, encontrados, ruta_parcial)
    pendientes = [s for s in encontrados if s not in ya_extraidos]
    if max_libros:
        pendientes = pendientes[:max_libros]

    print(f"\n{'=' * 62}\nFASE B - EXTRACCION DE FICHAS\n{'=' * 62}")
    print(f"Ya extraidos en corridas anteriores: {len(ya_extraidos)}")
    print(f"Pendientes en esta corrida:          {len(pendientes)}")

    hoy = date.today().isoformat()
    fallidos = []

    # "a" (append) y no "w": nunca se pisa lo que ya se habia guardado.
    with ruta_parcial.open("a", encoding="utf-8") as archivo:
        for i, slug in enumerate(pendientes, 1):
            url = url_ficha(slug)
            html = get_html(pagina, url)

            if html is None:
                fallidos.append((slug, "sin respuesta"))
                continue

            try:
                registro = parse_ficha(html, url)
            except Exception as e:
                # Un libro con estructura rara no puede tirar abajo la corrida.
                print(f"  [{i}/{len(pendientes)}] ERROR al parsear {slug}: {e}")
                fallidos.append((slug, str(e)))
                continue

            # Estos dos campos no salen de la ficha: uno viene de la Fase A y el
            # otro es el momento de la extraccion.
            registro["categoria_origen"] = lista_json(encontrados[slug])
            registro["fecha_extraccion"] = hoy

            archivo.write(json.dumps(registro, ensure_ascii=False) + "\n")
            # flush inmediato: sin esto el registro se quedaria en el buffer de
            # Python y se perderia si el proceso muere de golpe.
            archivo.flush()

            print(f"  [{i}/{len(pendientes)}] {registro['titulo'][:52]:<54}"
                  f" {len(registro['sinopsis']):>5} chars")
            pausa()

    if fallidos:
        print(f"\n{len(fallidos)} libros fallaron:")
        for slug, motivo in fallidos[:10]:
            print(f"    {slug[:44]:<46} {motivo[:40]}")

    return cargar_parcial(ruta_parcial)


# ---------------------------------------------------------------------------
# FASE C - Consolidacion y control de calidad
# ---------------------------------------------------------------------------

def consolidar(ruta_parcial, ruta_csv):
    """Convierte el JSONL de trabajo en el CSV entregable.

    Aca no se toca contenido: no se normalizan generos ni se unifican variantes
    de nombres. Eso corre despues en el ETL. Lo unico que se hace es dejar el
    archivo estructuralmente correcto: columnas en orden, tipos coherentes, sin
    duplicados y con los ausentes representados siempre igual.
    """
    import pandas as pd

    registros = list(cargar_parcial(ruta_parcial).values())
    if not registros:
        raise SystemExit("No hay registros para consolidar. Corre la Fase B primero.")

    df = pd.DataFrame(registros)

    # Orden de columnas fijo y explicito: el ETL espera siempre el mismo header.
    faltantes = [c for c in COLUMNAS if c not in df.columns]
    if faltantes:
        raise SystemExit(f"Faltan columnas en los registros: {faltantes}")
    df = df[COLUMNAS]

    # Deduplicacion por la clave del negocio. Deberia ser un no-op porque la
    # Fase A ya dedupica, pero es la garantia que pide la consigna y cuesta
    # una linea.
    antes = len(df)
    df = df.drop_duplicates(subset="url_libro", keep="first")
    if antes != len(df):
        print(f"  se eliminaron {antes - len(df)} duplicados por url_libro")

    # Tipos numericos. Int64 (con I mayuscula) es el entero *nullable* de
    # pandas: permite que serie_num quede vacio en los libros que no son de
    # serie, sin convertir toda la columna a float (que escribiria "1.0").
    for col in ["id", "n_autores", "serie_num"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    df = df.sort_values("id").reset_index(drop=True)

    # na_rep="" -> los ausentes salen como celda vacia, igual que el resto.
    #
    # lineterminator="\n" es importante aunque parezca cosmetico: por defecto
    # pandas usa el salto de linea del sistema operativo, asi que el mismo
    # dataset regenerado en Windows y en Linux da archivos distintos byte a
    # byte. Eso ensucia el repo con diffs de 200 lineas donde no cambio ni un
    # dato. Fijarlo hace que la salida sea identica en cualquier maquina.
    df.to_csv(ruta_csv, index=False, na_rep="", encoding=ENCODING_CSV,
              lineterminator="\n")
    print(f"\nEscrito: {ruta_csv}  ({len(df)} filas)")
    return df


def resumen_dataset(df):
    """Muestra la estructura del dataset: columnas, no nulos y tipo de dato."""
    print(f"\n{'=' * 62}\nESTRUCTURA DEL DATASET (df.info())\n{'=' * 62}")
    df.info()

    # info() cuenta "no nulos", pero nuestros ausentes son cadena vacia, no NaN:
    # para las columnas de texto siempre diria 100%. Este segundo cuadro mide lo
    # que de verdad interesa, que es cuantas celdas tienen contenido.
    print(f"\n{'columna':<20}{'no vacios':>12}{'%':>8}   ejemplo")
    print("-" * 88)
    for col in df.columns:
        serie = df[col].astype("string").fillna("")
        con_dato = (serie.str.strip() != "").sum()
        ejemplo = next((v for v in serie if v.strip()), "")
        print(f"{col:<20}{con_dato:>8}/{len(df):<4}{100 * con_dato / len(df):>7.0f}%"
              f"   {ejemplo[:44]}")


def validar(df):
    """Controles minimos exigidos por la consigna, mas algunos propios.

    Devuelve la lista de problemas encontrados: vacia significa que el dataset
    esta listo para entregar.
    """
    import pandas as pd

    print(f"\n{'=' * 62}\nCONTROLES DE CALIDAD\n{'=' * 62}")
    problemas = []

    def control(ok, descripcion, detalle=""):
        print(f"  [{'OK ' if ok else 'FALLA'}] {descripcion}{detalle}")
        if not ok:
            problemas.append(descripcion)

    # --- Exigidos por la consigna ---
    dups = int(df["url_libro"].duplicated().sum())
    control(dups == 0, "sin registros duplicados por url_libro", f" ({dups})")

    sin_titulo = int((df["titulo"].str.strip() == "").sum())
    control(sin_titulo == 0, "todos los registros tienen titulo", f" ({sin_titulo} sin)")

    urls_ok = df["url_libro"].map(lambda s: bool(s) and not s.startswith("http"))
    control(bool(urls_ok.all()), "todas las url_libro son slugs validos")

    con_sinopsis = (df["sinopsis"].str.strip() != "").mean()
    control(con_sinopsis >= 0.90, "la mayoria tiene sinopsis",
            f" ({100 * con_sinopsis:.0f}%)")

    texto = df.select_dtypes(include="object")
    sucios = [c for c in texto.columns
              if texto[c].str.contains(r"\n|\t|  |^\s|\s$", regex=True, na=False).any()]
    control(not sucios, "sin saltos de linea ni espacios sobrantes",
            f" (sucias: {sucios})" if sucios else "")

    nans = int(df.isna().sum().sum()) - int(df["serie_num"].isna().sum())
    control(nans == 0, "ausentes consistentes (solo serie_num admite nulo)")

    control(50 <= len(df) <= 200, "cantidad entre 50 y 200", f" ({len(df)})")

    # --- Propios, pensando en el recomendador de la Unidad 2 ---
    json_ok = True
    for col in ["autores", "generos", "categoria_origen"]:
        try:
            df[col].map(json.loads)
        except Exception:
            json_ok = False
    control(json_ok, "autores/generos/categoria_origen son JSON valido")

    control(bool(df["id"].notna().all() and df["id"].is_unique),
            "id presente y unico en todos los registros")

    largos = df["sinopsis"].str.len()
    control(largos.median() > 400, "sinopsis con senal suficiente para TF-IDF",
            f" (mediana {largos.median():.0f} chars)")

    conteo = (df["categoria_origen"].map(json.loads).explode()
              .value_counts().to_dict())
    minimo = min(conteo.values()) if conteo else 0
    control(minimo >= 0.6 * len(df) / max(len(conteo), 1),
            "categorias razonablemente balanceadas", f" {conteo}")

    print(f"\n{'DATASET OK' if not problemas else f'{len(problemas)} PROBLEMAS'}")
    return problemas


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extrae metadatos y sinopsis de libros desde Lectulandia."
    )
    parser.add_argument("--categorias", nargs="+", default=CATEGORIAS,
                        help="slugs de las categorias semilla")
    parser.add_argument("--cupo", type=int, default=CUPO_POR_CATEGORIA,
                        help="libros nuevos a descubrir por categoria")
    parser.add_argument("--ver-navegador", action="store_true",
                        help="muestra la ventana de Chromium (por defecto va "
                             "headless, sin ventana)")
    parser.add_argument("--max-libros", type=int, default=None,
                        help="tope de fichas a visitar en esta corrida; sirve "
                             "para probar sin hacer la corrida completa")
    parser.add_argument("--solo-descubrir", action="store_true",
                        help="corre solo la Fase A y no visita ninguna ficha")
    parser.add_argument("--solo-consolidar", action="store_true",
                        help="salta la red y regenera el CSV desde el parcial "
                             "ya guardado")
    # Rutas parametrizables: permiten armar un corpus aparte (por ejemplo uno
    # mas grande, para entrenar embeddings) sin tocar el entregable de la
    # Unidad 1. Cada corpus necesita SU PROPIO parcial: si dos corridas
    # compartieran el jsonl, la segunda creeria que ya extrajo los libros de la
    # primera y el CSV saldria incompleto.
    parser.add_argument("--salida", type=Path, default=RUTA_CSV,
                        help="ruta del CSV final (default: data/libros.csv)")
    parser.add_argument("--parcial", type=Path, default=None,
                        help="ruta del jsonl incremental (default: el que "
                             "corresponde a --salida)")
    args = parser.parse_args()

    ruta_csv = args.salida
    ruta_parcial = args.parcial or ruta_csv.with_name(
        ruta_csv.stem + "_parcial.jsonl")

    # Regenerar el CSV no necesita internet: se hace desde el JSONL de trabajo.
    if args.solo_consolidar:
        df = consolidar(ruta_parcial, ruta_csv)
        resumen_dataset(df)
        return 1 if validar(df) else 0

    # Import local: asi los parsers se pueden importar y testear sin tener
    # Playwright instalado ni levantar un navegador.
    from playwright.sync_api import sync_playwright

    inicio = time.time()
    ruta_csv.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        navegador = pw.chromium.launch(headless=not args.ver_navegador)
        try:
            pagina = nueva_pagina(navegador)

            encontrados = descubrir_urls(pagina, args.categorias, args.cupo)
            resumen_descubrimiento(encontrados, args.categorias)

            if args.solo_descubrir:
                print(f"\nFase A completada en {time.time() - inicio:.0f}s")
                return

            extraer_libros(pagina, encontrados, ruta_parcial,
                           args.max_libros)
        finally:
            navegador.close()

    df = consolidar(ruta_parcial, ruta_csv)
    resumen_dataset(df)
    problemas = validar(df)

    print(f"\nCorrida completada en {time.time() - inicio:.0f}s")
    return 1 if problemas else 0


def resumen_descubrimiento(encontrados, categorias):
    """Reporte de la Fase A: cuanto se junto y cuanto se solapan las categorias.

    El solapamiento es el dato que justifica haber deduplicado antes de visitar
    las fichas, asi que se informa explicitamente.
    """
    print(f"\n{'=' * 62}\nRESUMEN DE DESCUBRIMIENTO\n{'=' * 62}")
    print(f"URLs unicas a visitar: {len(encontrados)}")

    print("\nLibros por categoria (un libro puede contar en varias):")
    for categoria in categorias:
        n = sum(1 for cats in encontrados.values() if categoria in cats)
        print(f"  {categoria:<18} {n:>4}")

    compartidos = {s: c for s, c in encontrados.items() if len(c) > 1}
    total_apariciones = sum(len(c) for c in encontrados.values())
    ahorro = total_apariciones - len(encontrados)

    print(f"\nLibros en mas de una categoria: {len(compartidos)} "
          f"({100 * len(compartidos) / max(len(encontrados), 1):.0f}%)")
    print(f"Fichas que NO hay que volver a visitar gracias al dedup: {ahorro}")

    for slug, cats in list(compartidos.items())[:5]:
        print(f"    {slug[:44]:<46} {cats}")


if __name__ == "__main__":
    sys.exit(main() or 0)

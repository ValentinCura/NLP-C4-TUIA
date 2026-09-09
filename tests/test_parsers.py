"""Prueba de los parsers contra HTML guardado en disco (sin acceso a red).

Ejecutar desde la raiz del proyecto:
    venv/Scripts/python.exe tests/test_parsers.py
"""

import io
import json
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from scraper import (  # noqa: E402
    COLUMNAS,
    _titulo_con_serie,
    ENCODING_CSV,
    PREFIJO_FICHA,
    PREFIJO_PORTADA,
    parse_ficha,
    lista_json,
    parse_listado,
    url_ficha,
    url_portada,
)

FIXTURES = RAIZ / "tests" / "fixtures"


def leer(nombre):
    # encoding explicito: el sitio manda UTF-8 pero no lo declara en el HTML.
    return (FIXTURES / nombre).read_text(encoding="utf-8")


def mostrar(titulo, datos):
    print(f"\n=== {titulo} ===")
    for clave, valor in datos.items():
        if clave == "sinopsis":
            valor = f"[{len(valor)} chars] {valor[:100]}..."
        print(f"  {clave:<16} {valor!r}")


fallos = []


def chequear(condicion, mensaje):
    print(f"  [{'OK ' if condicion else 'FALLA'}] {mensaje}")
    if not condicion:
        fallos.append(mensaje)


# --- Ficha CON serie -------------------------------------------------------
con_serie = parse_ficha(
    leer("ficha_con_serie.html"),
    PREFIJO_FICHA + "el-silencio-del-bosque-jess-lourey/",
)
mostrar("Ficha CON serie", con_serie)
print()
chequear(con_serie["id"] == "125310", "id del sitio extraido del <body>")
chequear(con_serie["titulo"] == "El silencio del bosque", "titulo correcto")
chequear(con_serie["autores"] == '["Jess Lourey"]', "autores como array JSON")
chequear(con_serie["n_autores"] == 1, "n_autores contado")
chequear(con_serie["serie"] == "Steinbeck y Reed", "serie en columna propia")
chequear(con_serie["serie_num"] == "1", "numero de tomo extraido")
chequear(
    con_serie["titulo_serie"] == "El silencio del bosque - Steinbeck y Reed 1",
    "titulo_serie derivado con guion medio",
)
chequear(con_serie["generos"] == '["Intriga","Novela"]', "generos como array JSON")
chequear(
    con_serie["url_libro"] == "el-silencio-del-bosque-jess-lourey/",
    "url_libro guarda solo la parte variable",
)
chequear(
    url_ficha(con_serie["url_libro"])
    == "https://ww3.lectulandia.co/book/el-silencio-del-bosque-jess-lourey/",
    "URL de ficha se reconstruye completa",
)
chequear(
    not con_serie["portada"].startswith("http")
    and url_portada(con_serie["portada"]).startswith(PREFIJO_PORTADA),
    "portada guarda solo la parte variable y se reconstruye",
)

# --- Ficha SIN serie -------------------------------------------------------
sin_serie = parse_ficha(
    leer("ficha_sin_serie.html"), PREFIJO_FICHA + "la-habitacion-de-las-voces/"
)
mostrar("Ficha SIN serie", sin_serie)
print()
chequear(sin_serie["titulo"] == "La habitación de las voces", "titulo con acento OK")
chequear(sin_serie["serie"] == "", "serie ausente -> cadena vacia")
chequear(sin_serie["serie_num"] == "", "serie_num ausente -> cadena vacia")
chequear(
    sin_serie["titulo_serie"] == sin_serie["titulo"],
    "sin serie, titulo_serie == titulo (no cuelga un guion suelto)",
)
chequear(
    con_serie.keys() == sin_serie.keys(),
    "ambas fichas devuelven exactamente las mismas claves",
)

# --- Ficha "El novio": sinopsis con <b> y varios <br> ----------------------
novio = parse_ficha(leer("ficha_el_novio.html"), PREFIJO_FICHA + "el-novio/")
mostrar("Ficha El novio", novio)
print()
chequear(novio["id"] == "125510", "id correcto")
chequear(
    novio["generos"] == '["Intriga","Novela","Psicológico"]',
    "tres generos como array JSON, con tilde sin escapar",
)
chequear(
    novio["sinopsis"].startswith("Ella busca al hombre perfecto."),
    "la sinopsis incluye la frase en negrita del <b> inicial",
)
chequear(
    "próxima víctima" in novio["sinopsis"],
    "la sinopsis llega hasta el ultimo parrafo",
)
chequear(
    "perfecta. Sydney Shaw" in novio["sinopsis"],
    "los <br> quedan separados por espacio (no 'perfecta.Sydney')",
)
chequear(
    not any(c in novio["sinopsis"] for c in "\n\t") and "  " not in novio["sinopsis"],
    "sinopsis sin saltos de linea ni espacios dobles",
)

# --- Listado ---------------------------------------------------------------
urls = parse_listado(leer("listado_scifi.html"))
print(f"\n=== Listado ciencia-ficcion: {len(urls)} URLs ===")
for u in urls[:3]:
    print(f"  {u}")
print("  ...")
print()
chequear(len(urls) >= 20, f"encuentra >=20 libros (encontro {len(urls)})")
chequear(len(urls) == len(set(urls)), "sin URLs repetidas")
chequear(all(u.startswith(PREFIJO_FICHA) for u in urls), "todas absolutas a /book/")

# La trampa importante: la ficha individual tambien tiene <article class="card">
# con libros relacionados. Si por error se le pasara una ficha a parse_listado,
# devolveria URLs que no corresponden al listado.
relacionados = parse_listado(leer("ficha_sin_serie.html"))
chequear(
    len(relacionados) > 0,
    f"CONFIRMADO: una ficha contiene {len(relacionados)} article.card de "
    "relacionados -> por eso los parsers estan separados",
)
chequear(
    novio["id"] not in [u for u in relacionados],
    "el id NO sale de esos article.card (saldria el libro equivocado)",
)

# --- titulo_serie: casos borde --------------------------------------------
print("\n=== titulo_serie ===")
chequear(
    _titulo_con_serie("El silencio del bosque", "Steinbeck y Reed", "1")
    == "El silencio del bosque - Steinbeck y Reed 1",
    "caso normal: se agrega la saga con guion medio",
)
chequear(
    _titulo_con_serie("El novio", "", "") == "El novio",
    "sin serie: devuelve el titulo tal cual",
)
chequear(
    _titulo_con_serie("Más Allá 21", "Más Allá", "21") == "Más Allá 21",
    "el titulo YA contiene la saga -> no se repite ('Más Allá 21 - Más Allá 21')",
)
chequear(
    _titulo_con_serie("Dune", "Dune", "") == "Dune",
    "titulo igual al nombre de la saga -> no se duplica",
)

# --- Contrato del CSV que consume el ETL -----------------------------------
# Los campos multivaluados tienen que sobrevivir el viaje entero:
#   lista -> array JSON -> celda CSV quoteada -> lectura -> lista original
print("\n=== Contrato JSON / CSV ===")

chequear(lista_json([]) == "[]", "sin valores -> array JSON vacio, no cadena vacia")
chequear(lista_json(["Ficción"]) == '["Ficción"]', "un solo valor igual va en array")
chequear(
    json.loads(lista_json(['Género "raro"'])) == ['Género "raro"'],
    "las comillas dobles internas se escapan segun JSON",
)
chequear(
    "\\u" not in lista_json(["Ficción", "Psicológico"]),
    "las tildes van como UTF-8 real, no escapadas a \\u00f3",
)
for celda in [novio["generos"], con_serie["autores"], lista_json([])]:
    chequear(isinstance(json.loads(celda), list), f"{celda[:34]} es JSON valido")

# Viaje completo por un CSV real, con el caso feo incluido.
raro = {**novio, "id": "1", "generos": lista_json(['Género "raro"'])}
df = pd.DataFrame([con_serie, sin_serie, novio, raro])
buf = io.StringIO()
df.to_csv(buf, index=False, na_rep="")
crudo = buf.getvalue()
vuelta = pd.read_csv(io.StringIO(crudo), dtype=str, keep_default_na=False)

chequear(
    json.loads(vuelta["generos"].iloc[-1]) == ['Género "raro"'],
    "round-trip CSV: un genero con comillas dobles vuelve intacto",
)
chequear(
    vuelta["sinopsis"].iloc[2] == novio["sinopsis"],
    "round-trip CSV: la sinopsis con comas y comillas vuelve intacta",
)
chequear(
    not crudo.encode(ENCODING_CSV).startswith(b"\xef\xbb\xbf"),
    "el CSV NO lleva BOM (rompería el COPY de Postgres)",
)
chequear(
    crudo.splitlines()[0].split(",")[0] == "id",
    "header en la primera fila, con id como primera columna",
)
chequear(
    [c for c in COLUMNAS if c in con_serie] == list(con_serie.keys()),
    "parse_ficha devuelve las claves en el orden declarado en COLUMNAS",
)

print(f"\n{'TODO OK' if not fallos else f'{len(fallos)} FALLAS: {fallos}'}")
sys.exit(1 if fallos else 0)

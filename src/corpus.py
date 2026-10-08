"""
Acceso al corpus, desde el CSV o desde PostgreSQL.

Unidad 2 - Procesamiento del Lenguaje Natural (TUIA, FCEIA-UNR).

Este modulo devuelve los documentos en su version CRUDA, que es la unica que el
corpus almacena. La version tokenizada no vive aca ni en la base: la produce
preprocesamiento.py a demanda. El motivo esta explicado abajo, en la nota sobre
por que Documento no tiene un campo `tokens`.

La fuente por defecto es el CSV. La catedra aviso que para el TP2 no hace falta
la conexion a Postgres ("por ahora desde el CSV"), y el notebook corre en Colab,
donde no hay base. Postgres sigue disponible con fuente="postgres": las dos
fuentes devuelven exactamente los mismos Documento, en el mismo orden.

    python src/corpus.py                     resumen del corpus (desde el CSV)
    python src/corpus.py --fuente postgres   lo mismo, desde la base
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

# Cada "tabla" del corpus tiene su CSV. Se mantiene el nombre de tabla como
# identificador para que el resto del codigo no cambie segun la fuente.
CSV_POR_TABLA = {
    "libros": RAIZ / "data" / "libros.csv",
    "libros_ampliado": RAIZ / "data" / "libros_ampliado.csv",
}

# Columnas que se traen. El orden importa: tiene que coincidir con los campos de
# Documento para que class_row los mapee.
CAMPOS = ("id, titulo, autores, generos, categoria_origen, serie, serie_num, "
          "url_libro, sinopsis")


@dataclass(frozen=True)
class Documento:
    """Un libro del corpus, con el texto tal como lo publico el sitio.

    frozen=True porque un documento es un dato, no un objeto mutable: si alguien
    quisiera "limpiarlo" in place, la asignacion falla y queda claro que el
    corpus no es el lugar para eso.

    Los tres campos multivaluados llegan como list[str] de Python sin que nadie
    llame a json.loads: psycopg3 adapta jsonb automaticamente. Eso cierra el arco
    que empezo el scraper cuando eligio arrays JSON en vez de un separador
    propio: el dato viaja lista -> JSON -> celda CSV -> jsonb -> lista, y en
    ningun punto del camino hay un parseo hecho a mano.
    """

    id: int
    titulo: str
    autores: list[str] = field(default_factory=list)
    generos: list[str] = field(default_factory=list)
    categoria_origen: list[str] = field(default_factory=list)
    serie: str | None = None
    serie_num: int | None = None
    url_libro: str = ""
    sinopsis: str = ""

    @property
    def texto(self):
        """El texto que representa al documento para los modelos.

        Elegir QUE campos componen el documento es una decision de composicion
        del corpus, no de preprocesamiento linguistico: no altera ninguna
        palabra, solo decide cuales entran. Por eso vive legitimamente aca, pero
        en UN SOLO lugar, de modo que cambiarlo a "solo la sinopsis" sea una
        linea.

        Se incluye el titulo porque suele condensar el tema del libro en pocas
        palabras de alto valor semantico.
        """
        return f"{self.titulo.rstrip('.')}. {self.sinopsis}"


# NOTA IMPORTANTE: Documento NO tiene un campo `tokens`, y es deliberado.
#
#   1. No existe "la" tokenizacion: existe "la que quiere TF-IDF". El dia que se
#      agregue un LDA con otra lista de stopwords, `tokens` se vuelve ambiguo y
#      hay que partirlo en tokens_tfidf / tokens_lda. Que un atributo necesite
#      apellidarse con el nombre del modelo es la senal de que pertenece al
#      modelo, no al documento.
#   2. Acoplaria las capas: este modulo habla con PostgreSQL, y tokenizar
#      necesita spaCy. Quien solo quiera explorar el corpus tendria que
#      instalarse todo el stack de PLN.
#   3. El modelo de oracion, que consume el texto crudo, pagaria memoria y CPU
#      por algo que nunca usa.
#
# Las dos versiones que pide la consigna existen, pero como vistas derivadas del
# mismo list[Documento], alineadas por indice (y por .id, que es la clave real):
#
#     docs = corpus.traer_documentos()
#     crudos     = [d.texto for d in docs]                       -> modelo de oracion
#     tokenizados = preprocesamiento.tokenizar_corpus(crudos)    -> TF-IDF


def traer_documentos(generos=None, categorias=None, limite=None, tabla="libros",
                     fuente="csv"):
    """Devuelve los documentos del corpus, ordenados por id.

    fuente="csv" (default) lee data/<tabla>.csv; fuente="postgres" lee la base.
    Los filtros y el orden son los mismos en las dos.
    """
    if fuente == "csv":
        return _desde_csv(tabla, generos, categorias, limite)
    if fuente != "postgres":
        raise ValueError(f"fuente desconocida: {fuente!r} (csv o postgres)")
    return _desde_postgres(generos, categorias, limite, tabla)


def _desde_csv(tabla, generos=None, categorias=None, limite=None):
    """Lee el CSV aplicando la MISMA traduccion que hace el ETL al cargar la base.

    - autores, generos y categoria_origen: el CSV los trae como array JSON, se
      parsean con json.loads (nunca partiendo por comas).
    - serie y serie_num: '' en el CSV significa ausencia y se traduce a None,
      igual que el NULLIF del ETL. Asi un Documento leido del CSV es identico al
      leido de Postgres.
    """
    import csv

    if tabla not in CSV_POR_TABLA:
        raise ValueError(f"tabla desconocida: {tabla!r}. "
                         f"Opciones: {sorted(CSV_POR_TABLA)}")
    ruta = CSV_POR_TABLA[tabla]
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}")

    with ruta.open(encoding="utf-8", newline="") as archivo:
        filas = list(csv.DictReader(archivo))

    docs = []
    for f in filas:
        d = Documento(
            id=int(f["id"]),
            titulo=f["titulo"],
            autores=json.loads(f["autores"]),
            generos=json.loads(f["generos"]),
            categoria_origen=json.loads(f["categoria_origen"]),
            serie=f["serie"] or None,
            serie_num=int(f["serie_num"]) if f["serie_num"] else None,
            url_libro=f["url_libro"],
            sinopsis=f["sinopsis"],
        )
        # Mismo criterio que el operador ?| de Postgres: alcanza con que el
        # documento tenga CUALQUIERA de los valores pedidos.
        if generos and not set(generos) & set(d.generos):
            continue
        if categorias and not set(categorias) & set(d.categoria_origen):
            continue
        docs.append(d)

    docs.sort(key=lambda d: d.id)
    return docs[:limite] if limite else docs


def _desde_postgres(generos=None, categorias=None, limite=None, tabla="libros"):
    """Devuelve los documentos desde PostgreSQL, ordenados por id.

    Se devuelve una LISTA y no un generador a proposito: el corpus se recorre
    varias veces (ajustar TF-IDF, despues entrenar embeddings, despues evaluar)
    y un generador se agota en la primera pasada, produciendo el bug clasico de
    "la segunda vez me da vacio". Son 200 documentos, alrededor de 1 MB: no hay
    nada que transmitir de a poco.

    El ORDER BY id no es prolijidad sino correccion: sin ORDER BY, SQL no
    garantiza el orden de las filas, asi que una recarga podria devolverlas
    distinto y romperia en silencio cualquier matriz TF-IDF o de embeddings ya
    calculada, donde la fila i tiene que seguir siendo el libro i.

    El filtrado va en SQL y no en Python: es lo que le da sentido a haber
    guardado jsonb con indices GIN.
    """
    # Imports diferidos: quien usa el CSV (Colab) no necesita psycopg instalado.
    from psycopg.rows import class_row
    import db

    condiciones, parametros = [], []

    if generos:
        # ?| es "el array contiene CUALQUIERA de estas claves".
        condiciones.append("generos ?| %s")
        parametros.append(list(generos))
    if categorias:
        condiciones.append("categoria_origen ?| %s")
        parametros.append(list(categorias))

    where = f"WHERE {' AND '.join(condiciones)}" if condiciones else ""
    limit = "LIMIT %s" if limite else ""
    if limite:
        parametros.append(limite)

    consulta = f"SELECT {CAMPOS} FROM {tabla} {where} ORDER BY id {limit}"

    with db.conectar() as conexion:
        with conexion.cursor(row_factory=class_row(Documento)) as cur:
            cur.execute(consulta, parametros or None)
            return cur.fetchall()


def traer_documento(id_libro, tabla="libros", fuente="csv"):
    """Devuelve un documento por su id, o None si no existe."""
    if fuente == "csv":
        return next((d for d in _desde_csv(tabla) if d.id == id_libro), None)

    from psycopg.rows import class_row
    import db
    with db.conectar() as conexion:
        with conexion.cursor(row_factory=class_row(Documento)) as cur:
            cur.execute(f"SELECT {CAMPOS} FROM {tabla} WHERE id = %s",
                        (id_libro,))
            return cur.fetchone()


def a_dataframe(documentos):
    """Convierte los documentos a un DataFrame, para exploracion.

    El import de pandas va adentro de la funcion, siguiendo el mismo patron que
    consolidar() en scraper.py: quien solo quiere los documentos no necesita
    cargar pandas.
    """
    import pandas as pd
    return pd.DataFrame([vars(d) for d in documentos])


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Resumen del corpus cargado.")
    parser.add_argument("--tabla", default="libros")
    parser.add_argument("--generos", nargs="+", default=None)
    parser.add_argument("--categorias", nargs="+", default=None)
    parser.add_argument("--fuente", choices=["csv", "postgres"], default="csv")
    args = parser.parse_args()

    docs = traer_documentos(generos=args.generos, categorias=args.categorias,
                            tabla=args.tabla, fuente=args.fuente)
    if not docs:
        raise SystemExit("El corpus esta vacio. Corriste src/etl.py?")

    largos = sorted(len(d.texto) for d in docs)
    from collections import Counter
    generos = Counter(g for d in docs for g in d.generos)

    print(f"{'=' * 62}\nCORPUS: {len(docs)} documentos\n{'=' * 62}")
    print(f"  largo del texto : mediana {largos[len(largos) // 2]}, "
          f"min {largos[0]}, max {largos[-1]} caracteres")
    print(f"  con serie       : {sum(1 for d in docs if d.serie)}")
    print(f"  generos         : {len(generos)} distintos")
    print(f"  top 6           : {[g for g, _ in generos.most_common(6)]}")

    d = docs[0]
    print(f"\nPrimer documento:")
    print(f"  id      : {d.id}")
    print(f"  titulo  : {d.titulo}")
    # Estos dos llegaron como listas de Python sin un solo json.loads.
    print(f"  autores : {d.autores}   <- ya es list[str], no un string")
    print(f"  generos : {d.generos}")
    print(f"  serie   : {d.serie!r}   <- None, no cadena vacia")
    print(f"  texto   : {d.texto[:90]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())

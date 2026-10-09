"""
Carga del corpus a PostgreSQL.

Unidad 2 - Procesamiento del Lenguaje Natural (TUIA, FCEIA-UNR).

El flujo es el clasico de un ETL:

    CSV  --COPY-->  tabla staging (todo text)  --INSERT SELECT-->  tabla tipada

Se usa COPY y no INSERT fila por fila porque es el patron real de carga masiva,
y porque el CSV de la Unidad 1 se diseno explicitamente para esto: se guardo sin
BOM justamente para que un COPY no se tragara tres bytes invisibles dentro del
nombre de la primera columna.

La tabla intermedia existe para separar dos problemas: primero traer los datos
tal cual (donde lo unico que puede fallar es el formato del archivo), y despues
castearlos y validarlos en SQL declarativo, que es facil de leer y de justificar.

    python src/etl.py                      carga data/libros.csv
    python src/etl.py --recrear            borra la tabla y la crea de cero
    python src/etl.py --solo-verificar     no carga, solo controla
    python src/etl.py --csv data/otro.csv --tabla libros_ampliado
"""

import argparse
import csv
import sys
from pathlib import Path

import db

RAIZ = Path(__file__).resolve().parent.parent
RUTA_CSV = RAIZ / "data" / "libros.csv"
RUTA_ESQUEMA = RAIZ / "sql" / "01_esquema.sql"

# Mismo orden que COLUMNAS en scraper.py y que el header del CSV.
COLUMNAS = [
    "id", "titulo", "titulo_serie", "autores", "n_autores", "generos",
    "serie", "serie_num", "sinopsis", "url_libro", "portada",
    "categoria_origen", "fecha_extraccion",
]

# Columnas donde la cadena vacia significa "este libro no tiene ese atributo" y
# por lo tanto se traduce a NULL. En el resto, la columna es NOT NULL y una
# cadena vacia directamente no puede existir.
#
# El razonamiento de fondo: el CSV usa '' porque el formato CSV NO TIENE NULL
# (un campo vacio y un ausente son indistinguibles en RFC 4180, y cualquier
# centinela tipo "N/A" seria un valor mas que pandas malinterpretaria). SQL si
# tiene NULL, con semantica propia en agregados y joins. Cada formato expresa la
# ausencia en su propio idioma y el ETL es el traductor; guardar '' en Postgres
# seria importar la limitacion del formato de transporte al de destino.
NULLIFICAR = ["serie", "serie_num", "portada"]


# ---------------------------------------------------------------------------
# Esquema
# ---------------------------------------------------------------------------

def aplicar_esquema(conexion, tabla, recrear=False):
    """Crea la tabla destino y la staging."""
    sql = RUTA_ESQUEMA.read_text(encoding="utf-8")

    # El DDL esta escrito para la tabla "libros". Para cargar otro corpus en
    # otra tabla se reescriben los identificadores. Es un reemplazo de texto y
    # no una plantilla sofisticada a proposito: mantiene el .sql legible y
    # ejecutable a mano desde DBeaver, que es como se lo va a mirar.
    if tabla != "libros":
        sql = sql.replace("libros", tabla).replace("libro_genero",
                                                   f"{tabla}_genero")

    with conexion.cursor() as cur:
        if recrear:
            print(f"  DROP TABLE {tabla}")
            cur.execute(f"DROP TABLE IF EXISTS {tabla} CASCADE")
        cur.execute(sql)

        # Staging: las 13 columnas como text, sin constraints. UNLOGGED porque
        # es descartable y asi no genera WAL. No se dropea al terminar: si el
        # INSERT falla por una constraint, las filas crudas quedan disponibles
        # para inspeccionar cual fue la culpable.
        cur.execute(f"DROP TABLE IF EXISTS {tabla}_bruto")
        columnas_text = ", ".join(f"{c} text" for c in COLUMNAS)
        cur.execute(f"CREATE UNLOGGED TABLE {tabla}_bruto ({columnas_text})")


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def verificar_header(ruta_csv):
    """Comprueba el header ANTES de conectarse a la base.

    COPY es posicional: si alguien reordenara las columnas del CSV, los titulos
    entrarian en la columna de autores sin un solo error. Este chequeo y el
    HEADER MATCH del COPY convierten ese bug silencioso en una excepcion.
    """
    with ruta_csv.open(encoding="utf-8", newline="") as archivo:
        header = next(csv.reader(archivo))

    if header != COLUMNAS:
        raise SystemExit(
            f"El header del CSV no coincide con las columnas esperadas.\n"
            f"  esperado: {COLUMNAS}\n  recibido: {header}"
        )
    return header


def copiar_a_staging(conexion, ruta_csv, tabla):
    """Vuelca el CSV crudo en la tabla intermedia con COPY."""
    with conexion.cursor() as cur:
        cur.execute(f"TRUNCATE {tabla}_bruto")

        # OJO: se copia FROM STDIN, no FROM '/ruta/al/archivo'.
        #
        # Este es el error mas probable de todo el trabajo: COPY ... FROM
        # 'archivo' lee un archivo DEL SERVIDOR, que dentro de Docker es el
        # filesystem del contenedor, donde el CSV no existe. cursor.copy() abre
        # un COPY FROM STDIN y transmite los bytes desde el cliente.
        #
        # HEADER MATCH (ademas de saltear la primera linea) verifica que los
        # nombres del header coincidan con las columnas de la tabla.
        consulta = (f"COPY {tabla}_bruto ({', '.join(COLUMNAS)}) "
                    f"FROM STDIN WITH (FORMAT csv, HEADER MATCH)")

        with ruta_csv.open("rb") as archivo, cur.copy(consulta) as copy:
            while bloque := archivo.read(64 * 1024):
                copy.write(bloque)

        cur.execute(f"SELECT count(*) FROM {tabla}_bruto")
        return cur.fetchone()[0]


def insertar_tipado(conexion, tabla):
    """Pasa de la staging a la tabla final, casteando y aplicando la politica
    de NULL.

    Se usa ON CONFLICT DO UPDATE y no un TRUNCATE previo: si manana alguien
    re-scrapea con mas cupo y vuelve a cargar, el upsert agrega y actualiza,
    mientras que un TRUNCATE borraria en silencio las filas de una carga
    anterior mas amplia. El resultado es idempotente igual: correr el ETL dos
    veces seguidas deja la base exactamente igual.
    """
    # Cada columna se castea desde text al tipo real. NULLIF(col,'') traduce la
    # cadena vacia a NULL en las tres columnas donde eso significa ausencia.
    proyeccion = []
    for columna in COLUMNAS:
        expr = f"NULLIF({columna}, '')" if columna in NULLIFICAR else columna
        if columna in ("id", "n_autores", "serie_num"):
            tipo = "integer" if columna == "id" else "smallint"
            expr = f"{expr}::{tipo}"
        elif columna in ("autores", "generos", "categoria_origen"):
            expr = f"{expr}::jsonb"
        elif columna == "fecha_extraccion":
            expr = f"{expr}::date"
        proyeccion.append(f"{expr} AS {columna}")

    actualizables = [c for c in COLUMNAS if c != "id"]
    set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in actualizables)

    with conexion.cursor() as cur:
        cur.execute(f"""
            INSERT INTO {tabla} ({', '.join(COLUMNAS)})
            SELECT {', '.join(proyeccion)}
            FROM {tabla}_bruto
            ON CONFLICT (id) DO UPDATE SET {set_clause}
        """)
        cur.execute(f"SELECT count(*) FROM {tabla}")
        return cur.fetchone()[0]


# ---------------------------------------------------------------------------
# Verificacion post-carga
# ---------------------------------------------------------------------------

def verificar(conexion, tabla, ruta_csv):
    """Controles de que la carga no deformo los datos.

    Mismo formato [OK ]/[FALLA] que validar() en scraper.py: el grupo ya conoce
    esa salida.
    """
    import pandas as pd

    print(f"\n{'=' * 62}\nVERIFICACION DE LA CARGA\n{'=' * 62}")
    df = pd.read_csv(ruta_csv, dtype=str, keep_default_na=False)
    problemas = []

    def control(ok, descripcion, detalle=""):
        print(f"  [{'OK ' if ok else 'FALLA'}] {descripcion}{detalle}")
        if not ok:
            problemas.append(descripcion)

    def escalar(sql):
        with conexion.cursor() as cur:
            cur.execute(sql)
            return cur.fetchone()[0]

    filas = escalar(f"SELECT count(*) FROM {tabla}")
    control(filas == len(df), "la base tiene las mismas filas que el CSV",
            f" ({filas} vs {len(df)})")

    control(escalar(f"SELECT count(DISTINCT id) FROM {tabla}") == filas,
            "id unico en todas las filas")
    control(escalar(f"SELECT count(DISTINCT url_libro) FROM {tabla}") == filas,
            "url_libro unica en todas las filas")

    # El control mas rentable de todos: una sola suma detecta cualquier
    # deformacion de UTF-8 en las sinopsis. Si los acentos se rompieran en el
    # viaje, las longitudes cambiarian.
    suma_base = escalar(f"SELECT sum(length(sinopsis)) FROM {tabla}")
    suma_csv = int(df.sinopsis.str.len().sum())
    control(suma_base == suma_csv,
            "checksum de longitudes de sinopsis (detecta UTF-8 roto)",
            f" ({suma_base} vs {suma_csv})")

    # Round-trip puntual sobre una fila con tildes, por si la suma coincidiera
    # por casualidad.
    con_tilde = df[df.sinopsis.str.contains("ó", regex=False)].iloc[0]
    with conexion.cursor() as cur:
        cur.execute(f"SELECT sinopsis FROM {tabla} WHERE id = %s",
                    (int(con_tilde.id),))
        control(cur.fetchone()[0] == con_tilde.sinopsis,
                "una sinopsis con tildes vuelve identica caracter a caracter")

    # La politica de NULL se aplico...
    sin_serie_csv = int((df.serie == "").sum())
    control(escalar(f"SELECT count(*) FROM {tabla} WHERE serie IS NULL")
            == sin_serie_csv,
            "las ausencias quedaron como NULL, no como cadena vacia",
            f" ({sin_serie_csv} libros sin serie)")

    # ...y no sobrevivio ninguna cadena vacia.
    vacias = escalar(f"""
        SELECT count(*) FROM {tabla}
        WHERE '' IN (titulo, titulo_serie, sinopsis, url_libro)
           OR serie = '' OR portada = ''
    """)
    control(vacias == 0, "ninguna cadena vacia sobrevivio a la carga")

    # Los jsonb son navegables desde SQL.
    generos = escalar(f"""
        SELECT count(DISTINCT valor) FROM {tabla},
        jsonb_array_elements_text(generos) AS valor
    """)
    import json
    generos_csv = len({g for gs in df.generos.map(json.loads) for g in gs})
    control(generos == generos_csv, "los jsonb son navegables desde SQL",
            f" ({generos} generos distintos)")

    print(f"\n{'CARGA OK' if not problemas else f'{len(problemas)} PROBLEMAS'}")
    return problemas


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Carga el corpus del CSV a PostgreSQL.")
    parser.add_argument("--csv", type=Path, default=RUTA_CSV,
                        help="CSV de origen (default: data/libros.csv)")
    parser.add_argument("--tabla", default="libros",
                        help="tabla destino (default: libros)")
    parser.add_argument("--recrear", action="store_true",
                        help="borra la tabla y la vuelve a crear desde cero")
    parser.add_argument("--solo-verificar", action="store_true",
                        help="no carga nada, solo corre los controles")
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"No existe {args.csv}")

    print(f"Origen : {args.csv}")
    print(f"Destino: tabla {args.tabla} en {db.conninfo(ocultar_password=True)}")

    verificar_header(args.csv)

    # Todo dentro de un unico `with`: psycopg3 hace COMMIT al salir sin
    # excepcion y ROLLBACK si hubo una. O entran todas las filas o no entra
    # ninguna; nunca queda media carga.
    with db.conectar() as conexion:
        if not args.solo_verificar:
            aplicar_esquema(conexion, args.tabla, args.recrear)
            crudas = copiar_a_staging(conexion, args.csv, args.tabla)
            print(f"\n  COPY  -> {crudas} filas en {args.tabla}_bruto")
            finales = insertar_tipado(conexion, args.tabla)
            print(f"  INSERT -> {finales} filas en {args.tabla}")

        problemas = verificar(conexion, args.tabla, args.csv)

    return 1 if problemas else 0


if __name__ == "__main__":
    sys.exit(main())

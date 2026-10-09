"""
Conexion a PostgreSQL.

Unidad 2 - Procesamiento del Lenguaje Natural (TUIA, FCEIA-UNR).

Todo el resto del proyecto pide conexiones aca y nunca construye una cadena de
conexion por su cuenta. La configuracion sale del entorno (archivo .env), jamas
de literales en el codigo: por eso quien no pueda usar Docker instala PostgreSQL
nativo, cambia su .env y no toca una linea de Python.
"""

import os
import sys
import time

import psycopg
from dotenv import load_dotenv

# Valores por defecto, iguales a los de .env.example. Estan aca para que el
# proyecto funcione aunque alguien todavia no haya copiado el .env, y porque
# tener el default visible junto al nombre documenta que se espera.
# PGHOST va 127.0.0.1 y no "localhost": el puerto se publica atado a IPv4, pero
# "localhost" en Windows resuelve primero a ::1 (IPv6), donde no escucha nadie.
# Cada conexion espera el timeout de TCP antes de caer a IPv4: medido, 130
# segundos contra 0,02. Todo funciona, solo que inexplicablemente lento.
DEFAULTS = {
    "PGHOST": "127.0.0.1",
    "PGPORT": "5433",
    "PGUSER": "tuia",
    "PGPASSWORD": "tuia",
    "PGDATABASE": "tuia",
}

# Cuantas veces reintentar la conexion antes de rendirse, y cuanto esperar entre
# intentos. Cubre el caso de correr el ETL inmediatamente despues de
# `docker compose up -d`, cuando initdb todavia esta creando la base.
INTENTOS = 10
ESPERA = 1.0

# Segundos que espera CADA intento antes de darse por vencido.
#
# Sin esto, psycopg espera lo que decida el sistema operativo. Medido en esta
# maquina con Docker Desktop apagado: 130 segundos por intento, que con los 10
# reintentos son mas de veinte minutos colgado antes de ver el mensaje de error.
# Con connect_timeout=5 cada intento tarda 5,1 segundos y el caso mas comun
# (olvidarse de abrir Docker) falla en alrededor de un minuto.
#
# Ojo con la diferencia entre los dos modos de falla: cuando PostgreSQL todavia
# esta arrancando, el puerto existe y RECHAZA la conexion, asi que el intento
# falla al instante y los reintentos cubren el arranque en unos 10 segundos. El
# timeout solo entra en juego cuando no hay nadie escuchando y los paquetes se
# descartan en silencio, que es lo que pasa con Docker apagado.
TIMEOUT_CONEXION = 5


def _config():
    """Lee la configuracion de conexion del entorno.

    override=True hace que el .env del proyecto le gane a cualquier variable
    PG* que el sistema operativo ya tuviera definida. Sin eso, un integrante con
    un PostgreSQL propio y un PGDATABASE exportado en su perfil se conectaria a
    la base equivocada sin entender por que.
    """
    load_dotenv(override=True)
    return {clave: os.environ.get(clave, valor)
            for clave, valor in DEFAULTS.items()}


def conninfo(ocultar_password=False):
    """Arma la cadena de conexion a partir del entorno."""
    c = _config()
    password = "***" if ocultar_password else c["PGPASSWORD"]
    return (f"host={c['PGHOST']} port={c['PGPORT']} user={c['PGUSER']} "
            f"password={password} dbname={c['PGDATABASE']} "
            f"connect_timeout={TIMEOUT_CONEXION}")


def conectar(intentos=INTENTOS):
    """Devuelve una conexion abierta, reintentando mientras la base arranca.

    psycopg3 no es autocommit: quien llama decide cuando confirmar, tipicamente
    usando la conexion como context manager (`with conectar() as conexion:`),
    que hace COMMIT al salir sin excepcion y ROLLBACK si hubo una.
    """
    ultimo_error = None

    for intento in range(1, intentos + 1):
        try:
            return psycopg.connect(conninfo())
        except psycopg.OperationalError as e:
            ultimo_error = e
            if intento < intentos:
                # Mensaje solo a partir del segundo intento: el primero suele
                # fallar de forma normal si la base recien arranco.
                if intento == 2:
                    print("  esperando a que PostgreSQL acepte conexiones...")
                time.sleep(ESPERA)

    _explicar_fallo(ultimo_error)
    raise SystemExit(1)


def _explicar_fallo(error):
    """Imprime que revisar, en vez de dejar un traceback de psycopg.

    El modo de falla mas probable no es un bug del codigo sino que Docker
    Desktop este apagado, que es exactamente el estado en que arranca una
    maquina Windows.
    """
    print(f"\nNo se pudo conectar a PostgreSQL.\n"
          f"  conexion : {conninfo(ocultar_password=True)}\n"
          f"  error    : {str(error).strip().splitlines()[0]}\n"
          f"\nQue revisar, en orden:\n"
          f"  1. Que Docker Desktop este abierto y con el icono en verde.\n"
          f"  2. `docker compose ps`  -> el servicio db tiene que decir "
          f"'healthy'.\n"
          f"     Si no aparece:  `docker compose up -d`\n"
          f"  3. Que el PGPORT del archivo .env coincida con el puerto "
          f"publicado\n"
          f"     en docker-compose.yml (por defecto 5433, no 5432).\n"
          f"  4. Si no existe el .env:  copy .env.example .env\n")


def main():
    """Prueba la conexion e informa contra que base se conecto."""
    sys.stdout.reconfigure(encoding="utf-8")

    print(f"Conectando a: {conninfo(ocultar_password=True)}")
    with conectar() as conexion:
        with conexion.cursor() as cur:
            cur.execute("SELECT version(), current_database(), "
                        "current_setting('server_encoding')")
            version, base, encoding = cur.fetchone()

    print(f"\n  OK  {version.split(',')[0]}")
    print(f"      base     : {base}")
    print(f"      encoding : {encoding}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

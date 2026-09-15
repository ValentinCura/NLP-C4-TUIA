"""
Tokenizacion y limpieza linguistica del texto.

Unidad 2 - Procesamiento del Lenguaje Natural (TUIA, FCEIA-UNR).

===========================================================================
 EL PREPROCESAMIENTO PERTENECE AL MODELO, NO AL CORPUS
===========================================================================

El corpus guarda UN SOLO texto: el crudo. Este modulo produce la version
tokenizada a demanda, para los modelos que la necesitan. La regla para decidir
donde va cada transformacion es:

    Podria esta transformacion cambiar la respuesta de algun modelo razonable?

    NO  -> es limpieza de FORMATO (colapsar espacios, &nbsp;, saltos de linea).
           Va en el corpus. Es idempotente y no destruye nada.
    SI  -> es normalizacion LINGUISTICA (minusculas, puntuacion, stopwords,
           stemming, lematizacion). Pertenece al modelo.

Esto no es un criterio nuevo: continua el de la Unidad 1. El docstring de
limpiar() en scraper.py ya dice que eso es limpieza de formato y que normalizar
ahi destruiria informacion de forma irreversible.

Por que cada modelo necesita algo distinto:

  - TF-IDF / bag-of-words trata al documento como un MULTICONJUNTO de terminos.
    El orden y las palabras funcionales son ruido que infla la dimensionalidad.
    Normalizar agresivamente AUMENTA la senal, porque colapsa Casa/casa/casas en
    una sola dimension.

  - Un modelo de oracion trata al documento como una SECUENCIA. Fue entrenado
    sobre texto natural, con su propio tokenizador y su propio vocabulario.
    Darle texto en minusculas y sin stopwords produce un desajuste respecto de
    lo que vio en entrenamiento: la entrada queda fuera de dominio. Ademas los
    tokenizadores de subpalabra ya manejan mayusculas y puntuacion por su
    cuenta, asi que sacarselas antes destruye informacion que el modelo sabe
    usar y no le ahorra trabajo.

De ahi la conclusion: el preprocesamiento es un HIPERPARAMETRO del modelo.
Guardarlo en el corpus seria imponerle a un modelo el hiperparametro del otro.

---------------------------------------------------------------------------
REGLA DE DEPENDENCIA: este modulo NO importa corpus.py, y corpus.py no importa
este. Las funciones reciben str, nunca Documento. Es verificable con un grep, y
es lo que permite testear la tokenizacion con literales, sin base de datos.
---------------------------------------------------------------------------

    python src/preprocesamiento.py --demo    las dos versiones lado a lado
"""

import argparse
import re
import sys
import unicodedata

from spacy.lang.es.stop_words import STOP_WORDS

# Las stopwords salen de spacy.lang.es y no de nltk porque estan disponibles con
# solo hacer pip install spacy, sin descargar ningun modelo. nltk obligaria a un
# nltk.download('stopwords') en tiempo de ejecucion, o sea a tener red: fragil
# para el companero que replica el trabajo en otra maquina.
STOPWORDS = set(STOP_WORDS)

# Se tokeniza con expresion regular y no con el tokenizador estadistico de
# spaCy. La razon es honesta: para bag-of-words, partir por caracteres de
# palabra da practicamente el mismo resultado, corre unas 100 veces mas rapido y
# no obliga a descargar es_core_news_sm (~15 MB) para replicar el trabajo. Si
# hiciera falta lematizar de verdad, ahi si convendria spaCy completo.
#
# Se incluyen las vocales acentuadas, la enie y la dieresis explicitamente:
# \w con re.UNICODE ya las cubre, pero tambien cubre digitos y guiones bajos.
PATRON_TOKEN = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+")


def tokenizar(texto, minusculas=True, quitar_stopwords=True, largo_min=3,
              quitar_acentos=False):
    """Convierte un texto en una lista de tokens.

    Los parametros tienen default explicito para que se pueda decir "este es el
    preprocesamiento del modelo X" en vez de "el preprocesamiento". Cada uno es
    una PERDIDA DE INFORMACION DELIBERADA:

    minusculas=True
        Pierde las entidades nombradas: "Madrid" y "madrid" se vuelven el mismo
        token, y tambien se pierde la distincion entre el inicio de oracion y un
        nombre propio. Para bag-of-words conviene igual, porque sin esto el
        modelo trata "El" y "el" como dimensiones distintas.

    quitar_stopwords=True
        Pierde las negaciones y las relaciones sintacticas. Es la decision mas
        peligrosa de todas: "no" esta en la lista, asi que "no es un thriller"
        y "es un thriller" quedan identicos. Para TF-IDF se acepta porque las
        stopwords aparecen en todos los documentos y su IDF ya seria casi cero;
        para cualquier tarea sensible a la polaridad, esto es inaceptable.

    largo_min=3
        Descarta tokens cortos. Barre ruido de OCR y abreviaturas, pero tambien
        se lleva palabras de contenido legitimas: "rey", "sol", "mar" sobreviven
        por poco, y "fe" o "ir" no.

    quitar_acentos=False
        POR DEFECTO NO SE QUITAN, y es una decision consciente. En castellano
        las tildes son distintivas: papa/papa, esta/esta, publico/publico/
        publico son palabras diferentes. Quitarlas fusiona lemas distintos y
        pierde senal. Es un error frecuente heredado de pipelines pensados para
        ingles, donde no hay nada que perder.

    Lo que NO se hace: puntuacion. El patron de tokens simplemente no la captura,
    con lo cual se pierden los limites de oracion. Para bag-of-words no importa
    (no hay oraciones, hay una bolsa); para cualquier modelo secuencial si.
    """
    if not texto:
        return []

    tokens = PATRON_TOKEN.findall(texto)

    if minusculas:
        tokens = [t.lower() for t in tokens]

    if quitar_acentos:
        tokens = [_sin_acentos(t) for t in tokens]

    if quitar_stopwords:
        # La comparacion se hace siempre en minusculas porque la lista de spaCy
        # esta en minusculas: sin esto, con minusculas=False no filtraria nada.
        tokens = [t for t in tokens if t.lower() not in STOPWORDS]

    if largo_min > 1:
        tokens = [t for t in tokens if len(t) >= largo_min]

    return tokens


def tokenizar_corpus(textos, **opciones):
    """Tokeniza una lista de textos, conservando el orden.

    Recibe list[str] y no list[Documento]: ver la regla de dependencia en la
    cabecera del modulo.
    """
    return [tokenizar(t, **opciones) for t in textos]


def _sin_acentos(palabra):
    """Reduce las vocales acentuadas a su forma base.

    NFD descompone cada caracter acentuado en letra + tilde, y despues se
    descartan los diacriticos. Se conserva la enie, que en castellano no es una
    "n con tilde" sino una letra propia: confundirlas volveria "ano" y "anio" la
    misma palabra, con resultados desafortunados.
    """
    descompuesta = unicodedata.normalize("NFD", palabra.replace("ñ", "\0"))
    sin_tildes = "".join(c for c in descompuesta
                         if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", sin_tildes).replace("\0", "ñ")


def para_sklearn(textos, **opciones):
    """Tokeniza y vuelve a unir con espacios.

    TfidfVectorizer espera strings, no listas de tokens; gensim espera listas.
    Esta funcion existe para no tener el " ".join disperso por todo el codigo.
    """
    return [" ".join(tokens) for tokens in tokenizar_corpus(textos, **opciones)]


# ---------------------------------------------------------------------------
# Demostracion
# ---------------------------------------------------------------------------

def demo(cantidad=3):
    """Muestra las dos versiones del texto, una al lado de la otra.

    Es el mejor argumento del informe: en vez de afirmar que tokenizar destruye
    informacion, se ve desaparecer el orden, las mayusculas y las negaciones.
    """
    # Import diferido: es el unico lugar del modulo que toca el corpus, y se
    # hace aca adentro para que la regla de dependencia siga valiendo a nivel
    # de modulo.
    import corpus

    docs = corpus.traer_documentos(limite=cantidad)
    if not docs:
        raise SystemExit("El corpus esta vacio. Corriste src/etl.py?")

    for d in docs:
        crudo = d.texto
        tokens = tokenizar(crudo)

        print(f"\n{'=' * 74}\n{d.titulo}\n{'=' * 74}")
        print(f"\n  CRUDA  (lo que guarda el corpus, {len(crudo)} caracteres)")
        print(f"  {'-' * 70}")
        print(f"  {crudo[:330]}...")
        print(f"\n  TOKENIZADA  (lo que consume TF-IDF, {len(tokens)} tokens)")
        print(f"  {'-' * 70}")
        print(f"  {tokens[:40]}")

    print(f"\n{'=' * 74}")
    print("Que se perdio en el camino:")
    print("  - el ORDEN de las palabras y los limites de oracion")
    print("  - las MAYUSCULAS, o sea las entidades nombradas")
    print("  - las NEGACIONES: 'no' es stopword, asi que")
    print("    'no es un thriller' y 'es un thriller' quedan identicos")
    print("  - la PUNTUACION")
    print("\nSe conservaron las TILDES a proposito: en castellano distinguen")
    print("palabras (papa/papa, esta/esta), a diferencia del ingles.")
    return 0


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Tokenizacion del corpus (preprocesamiento del modelo).")
    parser.add_argument("--demo", action="store_true",
                        help="muestra texto crudo y tokens lado a lado")
    parser.add_argument("--cantidad", type=int, default=3)
    args = parser.parse_args()

    if args.demo:
        return demo(args.cantidad)

    # Sin --demo, un ejemplo autocontenido que no necesita base de datos.
    ejemplo = ("El Dr. Madrid no publico su libro en 1999: esta esperando. "
               "Su padre, el papa de la nacion, si lo publico.")
    print(f"texto  : {ejemplo}")
    print(f"tokens : {tokenizar(ejemplo)}")
    print(f"\nsin quitar stopwords : {tokenizar(ejemplo, quitar_stopwords=False)}")
    print(f"\nProbar el corpus real:  python src/preprocesamiento.py --demo")
    return 0


if __name__ == "__main__":
    sys.exit(main())

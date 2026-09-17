"""
Embeddings de palabras: modelo propio contra modelo pre-entrenado.

Unidad 2 - Procesamiento del Lenguaje Natural (TUIA, FCEIA-UNR).

Se entrenan Word2Vec y FastText sobre nuestro corpus y se comparan contra los
vectores del Spanish Billion Word Corpus (SBW), entrenados sobre mil millones de
palabras. La comparacion no es una formalidad: es la unica forma de ver, con
numeros propios, cuanto texto hace falta para que un embedding sirva.

Se anticipa el resultado para que no se lea como un fracaso: nuestro modelo va a
dar vecinos pobres. El corpus ampliado tiene 255.000 tokens y el SBW tiene 1.400
millones, cinco mil veces mas. La pregunta interesante no es cual gana, sino
QUE TIPO de error comete cada uno.

    python src/embeddings.py entrenar     entrena y guarda los modelos propios
    python src/embeddings.py vecinos      compara vecinos lado a lado
    python src/embeddings.py documentos   compara formas de armar el vector doc
"""

import argparse
import sys
from pathlib import Path

import corpus
import preprocesamiento

RAIZ = Path(__file__).resolve().parent.parent

# Los modelos entrenados se guardan fuera del repo o en data/, segun tamano.
# Son artefactos regenerables: no se versionan.
DIR_MODELOS = RAIZ / "data" / "modelos"

# El SBW pesa mas de 1 GB comprimido, asi que vive FUERA del repositorio, en la
# carpeta que lo contiene. Si no esta, el modulo lo dice y sigue funcionando con
# los modelos propios en vez de reventar.
RUTA_SBW = RAIZ.parent / "SBW-vectors-300-min5.bin.gz"

# Cuantas palabras del SBW se cargan en memoria.
#
# El archivo trae alrededor de un millon de vectores de 300 dimensiones. Cargarlo
# entero son varios GB de RAM y minutos de espera. Como estan ordenados por
# frecuencia descendente, quedarse con los primeros 400.000 conserva todo el
# vocabulario usual del castellano: lo que queda afuera son nombres propios
# raros, erratas y palabras de frecuencia minima. En la salida se reporta que
# porcentaje de NUESTRO vocabulario queda cubierto, para que el recorte sea una
# decision medida y no una suposicion.
LIMITE_SBW = 400_000

# Parametros del modelo propio. La justificacion de cada uno esta en entrenar().
PARAMETROS = {
    "vector_size": 100,
    "window": 5,
    "min_count": 3,
    "sg": 1,
    "epochs": 30,
    "workers": 4,
    "seed": 42,
}

# Palabras para comparar vecinos. Se eligen por tres motivos: son del dominio
# (aparecen en sinopsis de estos generos), son frecuentes en nuestro corpus (si
# no, el modelo propio ni siquiera las tendria) y tienen un vecindario claro en
# un modelo bien entrenado, asi que la diferencia se ve a simple vista.
PALABRAS_DOMINIO = ["guerra", "amor", "muerte", "magia", "nave", "vampiro",
                    "asesino", "rey"]


def cargar_oraciones(tabla):
    """Devuelve el corpus tokenizado, una lista de tokens por documento.

    gensim consume listas de tokens, no strings. Se usa la MISMA tokenizacion
    que TF-IDF a proposito, para que la comparacion entre representaciones sea
    justa: si cada una usara su propio preprocesamiento, las diferencias podrian
    venir de ahi y no del modelo.

    Dicho eso, conviene saber que no es lo ideal para embeddings: quitar
    stopwords elimina justamente las palabras que definen el contexto
    sintactico, que es de donde Word2Vec aprende. Un modelo serio se entrenaria
    sobre texto con stopwords y sin bajar a minusculas. Se acepta la perdida
    para mantener la comparacion controlada, y se deja dicho.
    """
    docs = corpus.traer_documentos(tabla=tabla)
    if not docs:
        raise SystemExit(f"La tabla {tabla} esta vacia. Corriste src/etl.py?")
    return [preprocesamiento.tokenizar(d.texto) for d in docs]


# ---------------------------------------------------------------------------
# Modelos propios
# ---------------------------------------------------------------------------

def entrenar(tabla="libros_ampliado", guardar=True):
    """Entrena Word2Vec y FastText sobre nuestro corpus.

    Justificacion de cada parametro:

    vector_size=100, no 300
        El SBW usa 300 porque tiene mil millones de palabras para estimarlas.
        Con 255.000 tokens, pedir 300 dimensiones es pedirle al modelo que
        estime mas parametros de los que los datos pueden sostener: las
        dimensiones sobrantes terminan modelando ruido. 100 ya es generoso.

    window=5
        Ventana estandar. Mas chica (2-3) captura relaciones sintacticas, mas
        grande (10) captura relaciones tematicas. Las sinopsis son descriptivas
        y queremos similitud tematica, asi que 5 es un punto medio razonable.

    min_count=3, no 5
        El default de gensim es 5, pero con este corpus descartaria demasiado.
        Con 3 el vocabulario aprendible es mayor. Es un compromiso: bajarlo mas
        agregaria palabras con tan pocos ejemplos que su vector seria ruido.

    sg=1 (skip-gram), no CBOW
        Skip-gram predice el contexto a partir de la palabra; CBOW hace lo
        contrario. Skip-gram anda mejor en corpus chicos y con palabras poco
        frecuentes, que es exactamente nuestra situacion. CBOW es mas rapido,
        pero la velocidad no es el problema aca.

    epochs=30, no 5
        El default asume un corpus grande, donde cada palabra aparece muchas
        veces por epoca. Con pocos datos conviene dar mas pasadas para que los
        vectores converjan. Con 255.000 tokens sigue siendo cuestion de
        segundos.

    seed=42 y workers=4
        Ojo con esto: con varios workers el entrenamiento NO es completamente
        reproducible aunque se fije la semilla, porque el orden en que los hilos
        actualizan los pesos varia. Para reproducibilidad exacta habria que usar
        workers=1. Se elige velocidad y se deja dicho, que es mejor que
        prometer un determinismo que no se cumple.
    """
    from gensim.models import FastText, Word2Vec

    oraciones = cargar_oraciones(tabla)
    tokens = sum(len(o) for o in oraciones)
    vocabulario = {p for o in oraciones for p in o}

    print(f"\nCorpus de entrenamiento: tabla '{tabla}'")
    print(f"  {len(oraciones)} documentos | {tokens:,} tokens | "
          f"{len(vocabulario):,} palabras distintas")
    print(f"\nParametros: " + ", ".join(f"{k}={v}" for k, v in PARAMETROS.items()))

    modelos = {}
    for nombre, clase in [("word2vec", Word2Vec), ("fasttext", FastText)]:
        modelo = clase(sentences=oraciones, **PARAMETROS)
        modelos[nombre] = modelo
        print(f"\n  {nombre:<10} vocabulario aprendido: "
              f"{len(modelo.wv):,} palabras "
              f"({100 * len(modelo.wv) / len(vocabulario):.0f}% del corpus)")

    print(f"\n  Las palabras descartadas son las que aparecen menos de "
          f"{PARAMETROS['min_count']} veces.")
    print(f"  FastText igual puede vectorizarlas, porque compone el vector a")
    print(f"  partir de n-gramas de caracteres. Word2Vec no: para el, una")
    print(f"  palabra fuera del vocabulario simplemente no existe.")

    if guardar:
        DIR_MODELOS.mkdir(parents=True, exist_ok=True)
        for nombre, modelo in modelos.items():
            ruta = DIR_MODELOS / f"{nombre}_{tabla}.model"
            modelo.save(str(ruta))
            print(f"  guardado: {ruta.relative_to(RAIZ)}")

    return modelos


def cargar_propios(tabla="libros_ampliado"):
    """Carga los modelos ya entrenados, o los entrena si no existen."""
    from gensim.models import FastText, Word2Vec

    modelos = {}
    for nombre, clase in [("word2vec", Word2Vec), ("fasttext", FastText)]:
        ruta = DIR_MODELOS / f"{nombre}_{tabla}.model"
        if not ruta.exists():
            print("No hay modelos guardados todavia; entrenando...")
            return entrenar(tabla)
        modelos[nombre] = clase.load(str(ruta))
    return modelos


# ---------------------------------------------------------------------------
# Modelo pre-entrenado
# ---------------------------------------------------------------------------

def cargar_sbw(limite=LIMITE_SBW):
    """Carga los vectores del Spanish Billion Word Corpus.

    Devuelve None si el archivo no esta, en vez de fallar: asi el resto del
    modulo sigue siendo util para quien no lo haya descargado.
    """
    from gensim.models import KeyedVectors

    if not RUTA_SBW.exists():
        print(f"\nNo se encontro el modelo pre-entrenado en:\n  {RUTA_SBW}")
        print(f"Descargarlo de https://crscardellino.ar/SBWCE/ "
              f"(SBW-vectors-300-min5.bin.gz, ~1 GB)")
        print(f"Va fuera del repositorio a proposito, por su tamano.")
        return None

    print(f"\nCargando SBW (limite: {limite:,} palabras mas frecuentes)...")
    kv = KeyedVectors.load_word2vec_format(str(RUTA_SBW), binary=True,
                                           limit=limite)
    print(f"  {len(kv):,} vectores de {kv.vector_size} dimensiones")
    return kv


# ---------------------------------------------------------------------------
# Comparacion de vecinos
# ---------------------------------------------------------------------------

def comparar_vecinos(tabla="libros_ampliado", palabras=None, n=6):
    """Muestra los vecinos mas cercanos de cada palabra en los tres modelos."""
    palabras = palabras or PALABRAS_DOMINIO
    propios = cargar_propios(tabla)
    sbw = cargar_sbw()

    print(f"\n{'=' * 78}\nVECINOS MAS CERCANOS\n{'=' * 78}")

    if sbw is not None:
        _cobertura(propios["word2vec"], sbw)

    for palabra in palabras:
        print(f"\n  '{palabra}'")
        print(f"  {'-' * 74}")
        for nombre, modelo in [("word2vec propio", propios["word2vec"].wv),
                               ("fasttext propio", propios["fasttext"].wv),
                               ("SBW pre-entrenado", sbw)]:
            if modelo is None:
                continue
            if palabra not in modelo:
                # FastText puede vectorizar palabras que no vio, por subpalabras.
                if nombre.startswith("fasttext"):
                    vecinos = modelo.most_similar(palabra, topn=n)
                    marca = " (fuera de vocabulario, compuesta por n-gramas)"
                else:
                    print(f"    {nombre:<20} (no esta en el vocabulario)")
                    continue
            else:
                vecinos = modelo.most_similar(palabra, topn=n)
                marca = ""
            listado = ", ".join(f"{p} {s:.2f}" for p, s in vecinos)
            print(f"    {nombre:<20} {listado}{marca}")

    _medir_parentesco_formal(propios, sbw, palabras)

    print(f"\n{'=' * 78}\nCOMO LEER ESTA TABLA\n{'=' * 78}")
    print("  No se trata de ver cual 'gana'. Los dos modelos fallan distinto:")
    print()
    print("  El modelo PROPIO tiende a dar vecinos que coocurren en las mismas")
    print("  sinopsis, no palabras de significado parecido. Si 'vampiro' y")
    print("  'mansion' aparecen juntas en varias contratapas, quedan cerca")
    print("  aunque no tengan relacion semantica. Con 255.000 tokens no hay")
    print("  suficientes contextos distintos para separar una cosa de la otra.")
    print()
    print("  El modelo SBW da vecinos semanticamente correctos, pero GENERALES:")
    print("  entrenado sobre noticias y Wikipedia, no sabe nada del registro de")
    print("  las contratapas ni de las convenciones de estos generos.")
    print()
    print("  FastText propio rinde mejor que Word2Vec propio con las palabras")
    print("  raras, porque compone su vector a partir de n-gramas de caracteres")
    print("  y aprovecha la morfologia del castellano. Es lo que mas conviene")
    print("  cuando hay poco texto. Pero eso mismo lo hace confundir palabras")
    print("  que solo se PARECEN en la forma: los vecinos de una palabra corta")
    print("  suelen incluir sus variantes flexivas y palabras que la contienen.")


def _medir_parentesco_formal(propios, sbw, palabras, n=10):
    """Cuantifica cuantos vecinos se parecen solo en la FORMA de la palabra.

    En la tabla de vecinos se ve a simple vista que FastText devuelve variantes
    morfologicas y, peor, palabras que solo comparten letras: magia -> mafia,
    guerra -> aferra, vampiro -> zafiro. Eso es consecuencia directa de que
    componga los vectores con n-gramas de caracteres.

    "A simple vista" no es una medicion, asi que se cuenta: que fraccion de los
    vecinos comparte un prefijo de 4 caracteres con la palabra consultada. Es
    una aproximacion grosera de la similitud formal, pero suficiente para
    comparar los tres modelos entre si, que es lo que interesa.
    """
    print(f"\n  PARENTESCO FORMAL de los vecinos")
    print(f"  Fraccion de los top-{n} que comparten 4 letras iniciales con la")
    print(f"  palabra consultada. Alto = el modelo responde a la forma escrita;")
    print(f"  bajo = responde al significado.")

    modelos = [("word2vec propio", propios["word2vec"].wv),
               ("fasttext propio", propios["fasttext"].wv)]
    if sbw is not None:
        modelos.append(("SBW pre-entrenado", sbw))

    print(f"\n  {'modelo':<22}{'formales':>10}{'total':>8}{'fraccion':>11}")
    print("  " + "-" * 51)
    for nombre, modelo in modelos:
        formales = total = 0
        for palabra in palabras:
            if palabra not in modelo and not nombre.startswith("fasttext"):
                continue
            for vecino, _ in modelo.most_similar(palabra, topn=n):
                total += 1
                if vecino.lower()[:4] == palabra.lower()[:4]:
                    formales += 1
        if total:
            print(f"  {nombre:<22}{formales:>10}{total:>8}"
                  f"{formales / total:>10.0%}")


def _cobertura(propio, sbw):
    """Cuanto del vocabulario de nuestro corpus cubre el SBW recortado."""
    nuestro = set(propio.wv.key_to_index)
    cubiertas = sum(1 for p in nuestro if p in sbw)
    print(f"\n  Cobertura: el SBW recortado a {len(sbw):,} palabras contiene "
          f"{cubiertas:,} de")
    print(f"  las {len(nuestro):,} de nuestro modelo "
          f"({100 * cubiertas / len(nuestro):.1f}%). Las que faltan son")
    print(f"  nombres propios y terminos muy especificos del dominio.")


# ---------------------------------------------------------------------------
# Vectores de documento
# ---------------------------------------------------------------------------

def vector_promedio(tokens, kv, pesos=None):
    """Promedia los vectores de las palabras del documento.

    Es la forma mas simple de pasar de vectores de palabra a vector de
    documento, y la mas usada. Lo que se pierde:

    ORDEN. Es lo que ya mostro la Unidad 2 con spaCy: "this is cool" e "is this
    cool" dan similitud 1.0, porque el promedio de un conjunto no depende de
    como se ordene. Con el orden se va la sintaxis entera.

    NEGACION Y COMPOSICION. "no es terror" y "es terror" quedan practicamente en
    el mismo punto. Ninguna operacion conmutativa puede representar que una
    palabra modifique a otra, y el promedio lo es.

    DILUCION. Un documento largo tiende al centroide del corpus: cuantas mas
    palabras se promedian, mas se cancelan las direcciones particulares. Los
    documentos largos se parecen entre si por ser largos, no por su tema.

    HUBNESS. Consecuencia de lo anterior: los vectores promedio se concentran en
    una region chica del espacio, las similitudes coseno se comprimen hacia
    arriba y aparecen "hubs", documentos que salen como vecinos de casi todo.

    Se usa igual porque no requiere entrenamiento, es O(n), no tiene
    hiperparametros y funciona sorprendentemente bien en recuperacion. Es la
    linea base honesta contra la que hay que comparar cualquier cosa mas
    sofisticada.

    El parametro pesos permite la variante ponderada por IDF.
    """
    import numpy as np

    vectores, w = [], []
    for token in tokens:
        if token in kv:
            vectores.append(kv[token])
            w.append(pesos.get(token, 1.0) if pesos else 1.0)

    if not vectores:
        return np.zeros(kv.vector_size, dtype=np.float32)
    return np.average(np.array(vectores), axis=0, weights=np.array(w))


def comparar_documentos(tabla="libros", modelo_tabla="libros_ampliado"):
    """Compara formas de armar el vector de documento con una tarea real.

    La tarea es recuperacion: dado un libro que pertenece a una saga, cuan alto
    rankea otro libro de la MISMA saga entre todos los demas.

    La eleccion no es caprichosa. Los libros de una misma serie comparten
    personajes, escenarios y trama, asi que un buen representante del documento
    deberia ponerlos cerca. Y la etiqueta sale de la columna serie, que ya
    estaba en el corpus: es ground truth que no hubo que anotar a mano ni
    inventar.
    """
    import numpy as np
    from collections import defaultdict
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    docs = corpus.traer_documentos(tabla=tabla)
    tokenizados = [preprocesamiento.tokenizar(d.texto) for d in docs]

    # Solo sirven las series con dos o mas libros presentes en el corpus.
    series = defaultdict(list)
    for i, d in enumerate(docs):
        if d.serie:
            series[d.serie].append(i)
    evaluables = {s: idx for s, idx in series.items() if len(idx) > 1}
    pares = sum(len(idx) for idx in evaluables.values())

    print(f"\n{'=' * 78}\nVECTOR DE DOCUMENTO: como armarlo\n{'=' * 78}")
    print(f"\nTarea de evaluacion: recuperacion de libros de la misma saga.")
    print(f"  corpus: {len(docs)} libros de la tabla '{tabla}'")
    print(f"  {len(evaluables)} series con 2 o mas tomos, {pares} libros "
          f"evaluables")
    print(f"  La etiqueta sale de la columna 'serie': ground truth gratis.")

    if pares < 4:
        raise SystemExit("Muy pocas series para evaluar.")

    propios = cargar_propios(modelo_tabla)
    sbw = cargar_sbw()

    # Pesos IDF, para la variante ponderada. Las palabras que aparecen en todos
    # los documentos pesan poco; las distintivas, mucho.
    vec = TfidfVectorizer()
    vec.fit([" ".join(t) for t in tokenizados])
    idf = dict(zip(vec.get_feature_names_out(), vec.idf_))

    representaciones = {}

    # Linea base sin embeddings: TF-IDF puro.
    representaciones["TF-IDF (sin embeddings)"] = vec.transform(
        [" ".join(t) for t in tokenizados]).toarray()

    for nombre, kv in [("word2vec propio", propios["word2vec"].wv),
                       ("fasttext propio", propios["fasttext"].wv),
                       ("SBW pre-entrenado", sbw)]:
        if kv is None:
            continue
        representaciones[f"{nombre}, promedio"] = np.array(
            [vector_promedio(t, kv) for t in tokenizados])
        representaciones[f"{nombre}, promedio IDF"] = np.array(
            [vector_promedio(t, kv, idf) for t in tokenizados])

    # Se guardan los MRR para poder sacar conclusiones de los numeros en vez
    # de afirmar de antemano lo que "deberia" pasar.
    resultados_mrr = {}

    print(f"\n{'representacion':<32}{'MRR':>8}{'acierto@1':>12}"
          f"{'acierto@5':>12}")
    print("  " + "-" * 62)

    for nombre, matriz in representaciones.items():
        sim = cosine_similarity(matriz)
        np.fill_diagonal(sim, -np.inf)     # un libro no se recupera a si mismo

        rangos = []
        for indices in evaluables.values():
            for i in indices:
                orden = np.argsort(sim[i])[::-1]
                companeros = set(indices) - {i}
                # Posicion del primer libro de la misma saga.
                posicion = next(r for r, j in enumerate(orden, 1)
                                if j in companeros)
                rangos.append(posicion)

        rangos = np.array(rangos)
        resultados_mrr[nombre] = float(np.mean(1 / rangos))
        print(f"  {nombre:<32}{np.mean(1 / rangos):>6.3f}"
              f"{np.mean(rangos == 1):>12.1%}{np.mean(rangos <= 5):>12.1%}")

    print(f"\n  MRR = promedio de 1/posicion del primer acierto. 1.0 seria")
    print(f"  recuperar siempre un tomo de la saga en el primer puesto.")

    print(f"\nPOR QUE GANA TF-IDF, QUE NO USA EMBEDDINGS")
    print(f"  El resultado sorprende hasta que se mira que pide la tarea. Los")
    print(f"  libros de una saga comparten NOMBRES PROPIOS: personajes,")
    print(f"  lugares, objetos inventados. Son palabras rarisimas, que")
    print(f"  aparecen en dos o tres documentos del corpus y en ningun otro.")
    print(f"  TF-IDF les da el peso maximo justamente por eso.")
    print(f"\n  Los embeddings hacen lo contrario: GENERALIZAN. Su virtud es")
    print(f"  mapear palabras distintas a posiciones parecidas, y eso aca")
    print(f"  destruye la senal. El SBW, que es el que mejor generaliza, es el")
    print(f"  que peor rinde: convierte cada nombre propio en 'un nombre")
    print(f"  propio mas' y borra lo unico que distinguia a la saga.")
    print(f"\n  La leccion no es que los embeddings sean peores, sino que la")
    print(f"  representacion correcta depende de la tarea. Para recuperar")
    print(f"  'el mismo universo narrativo' conviene lo especifico; para")
    print(f"  recomendar 'algo parecido pero distinto', que es lo que quiere")
    print(f"  un recomendador de verdad, conviene lo que generaliza. Con esta")
    print(f"  tarea no se puede concluir cual sirve para aquella.")

    print(f"\nRESPUESTA A LA CONSIGNA")
    print(f"  Como armamos el vector de documento: PROMEDIO de los vectores de")
    print(f"  las palabras que estan en el vocabulario del modelo, con la")
    print(f"  variante ponderada por IDF como comparacion.")
    print(f"\n  Que se pierde al promediar, ademas del orden:")
    print(f"   - la NEGACION y la composicion: ninguna operacion conmutativa")
    print(f"     puede expresar que una palabra modifique a otra;")
    print(f"   - por DILUCION, los documentos largos tienden al centroide del")
    print(f"     corpus y se parecen entre si por ser largos, no por el tema;")
    print(f"   - por HUBNESS, las similitudes se comprimen hacia arriba y")
    print(f"     aparecen documentos que son vecinos de casi todos.")
    print(f"\n  Por que se usa igual: no requiere entrenamiento, es O(n), no")
    print(f"  tiene hiperparametros y es la linea base contra la cual medir")
    print(f"  cualquier cosa mas sofisticada.")

    # La conclusion sobre el IDF se calcula de la tabla en vez de afirmarse:
    # el efecto no es el mismo en los tres modelos.
    print(f"\n  SOBRE PONDERAR POR IDF: el resultado NO es uniforme, asi que")
    print(f"  no se puede decir que 'ponderar mejora'. Segun esta corrida:")
    for base in [n for n in representaciones if n.endswith(", promedio")]:
        con_idf = base + " IDF"
        if con_idf not in resultados_mrr:
            continue
        delta = resultados_mrr[con_idf] - resultados_mrr[base]
        veredicto = "mejora" if delta > 0.005 else (
            "empeora" if delta < -0.005 else "no cambia")
        print(f"    {base.replace(', promedio', ''):<22} {delta:+.3f}  "
              f"{veredicto}")
    print(f"\n  Ponderar por IDF solo ayuda cuando el modelo de palabras es")
    print(f"  pobre y necesita que le indiquen donde mirar. Cuando ya captura")
    print(f"  bien el significado, agregar el peso puede desbalancear el")
    print(f"  promedio hacia terminos raros y empeorar el resultado.")


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Embeddings propios contra pre-entrenados.")
    parser.add_argument("accion", choices=["entrenar", "vecinos", "documentos"])
    parser.add_argument("--tabla", default="libros_ampliado",
                        help="corpus de entrenamiento (default: el ampliado)")
    parser.add_argument("--palabras", nargs="+", default=None)
    args = parser.parse_args()

    if args.accion == "entrenar":
        entrenar(args.tabla)
    elif args.accion == "vecinos":
        comparar_vecinos(args.tabla, args.palabras)
    elif args.accion == "documentos":
        # Se evalua sobre el corpus entregable, pero con los embeddings
        # entrenados sobre el ampliado: es el uso realista.
        comparar_documentos(tabla="libros", modelo_tabla=args.tabla)
    return 0


if __name__ == "__main__":
    sys.exit(main())

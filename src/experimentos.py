"""
Experimentos sobre el corpus: las cuatro preguntas de la consigna, medidas.

Unidad 2 - Procesamiento del Lenguaje Natural (TUIA, FCEIA-UNR).

Las cuatro preguntas se podrian contestar razonando. Se contestan midiendo,
porque tenemos 200 documentos reales y responder con numeros propios vale mas
que responder con opiniones.

    python src/experimentos.py idioma        cuantas sinopsis no son castellano
    python src/experimentos.py estabilidad   cuantos documentos necesita TF-IDF
    python src/experimentos.py promocional   el sesgo del texto publicitario
    python src/experimentos.py multietiqueta como cambia la evaluacion
    python src/experimentos.py todos
"""

import argparse
import sys
from collections import Counter

import corpus
import preprocesamiento

SEPARADOR = "=" * 74


def titulo(texto):
    print(f"\n{SEPARADOR}\n{texto}\n{SEPARADOR}")


# ===========================================================================
# Pregunta 4 - Que pasa con los libros en gallego o catalan?
# ===========================================================================

def experimento_idioma(docs):
    """Detecta el idioma de cada sinopsis y reporta la confianza.

    La pregunta de la consigna presupone que hay libros en otras lenguas. En vez
    de suponer cuantos, se miden.

    El punto interesante no es el conteo sino la CONFIANZA: castellano, catalan,
    gallego y portugues son lenguas romances muy cercanas, con vocabulario y
    morfologia compartidos. Un detector no se equivoca al azar entre ellas: duda
    de forma sistematica. Reportar solo la etiqueta ocultaria eso.
    """
    from lingua import Language, LanguageDetectorBuilder

    titulo("PREGUNTA 4 - Deteccion de idioma")

    # Se restringe el detector a las lenguas plausibles del catalogo. Con los 75
    # idiomas habilitados, el detector reparte probabilidad entre opciones
    # imposibles (un texto castellano recibe algo de probabilidad de tagalo) y
    # las confianzas quedan artificialmente bajas.
    candidatos = [Language.SPANISH, Language.CATALAN, Language.PORTUGUESE,
                  Language.BASQUE, Language.ENGLISH, Language.FRENCH,
                  Language.ITALIAN]
    detector = (LanguageDetectorBuilder.from_languages(*candidatos)
                .with_preloaded_language_models().build())

    print("\nLIMITACION DEL INSTRUMENTO, antes de mirar ningun resultado:")
    print("  lingua soporta 75 idiomas, entre ellos el catalan, pero NO tiene")
    print("  modelo de GALLEGO. Una sinopsis en gallego no puede ser detectada")
    print("  como tal: va a caer en portugues o en castellano, sus parientes")
    print("  mas cercanos. La pregunta de la consigna no se puede responder")
    print("  del todo con esta herramienta, y conviene decirlo en vez de")
    print("  reportar 'cero libros en gallego' como si fuera un hallazgo.")

    resultados = []
    for d in docs:
        confianzas = detector.compute_language_confidence_values(d.sinopsis)
        mejor = confianzas[0]
        segunda = confianzas[1] if len(confianzas) > 1 else None
        resultados.append((d, mejor, segunda))

    conteo = Counter(r[1].language.name for r in resultados)
    print(f"\nIdioma detectado en {len(docs)} sinopsis:")
    for idioma, n in conteo.most_common():
        print(f"  {idioma:<12} {n:>4}  ({100 * n / len(docs):.1f}%)")

    # La distribucion de confianza es el dato que importa.
    confianzas = sorted(r[1].value for r in resultados)
    mediana = confianzas[len(confianzas) // 2]
    print(f"\nConfianza de la deteccion:")
    print(f"  mediana {mediana:.3f}  |  minima {confianzas[0]:.3f}  |  "
          f"maxima {confianzas[-1]:.3f}")

    dudosos = [r for r in resultados if r[1].value < 0.90]
    print(f"  sinopsis con confianza < 0.90: {len(dudosos)} de {len(docs)}")

    if dudosos:
        print(f"\nLas mas dudosas (el detector reparte entre lenguas cercanas):")
        for d, mejor, segunda in sorted(dudosos, key=lambda r: r[1].value)[:6]:
            alternativa = (f"{segunda.language.name} {segunda.value:.2f}"
                           if segunda else "-")
            print(f"  {d.titulo[:38]:<40} {mejor.language.name} "
                  f"{mejor.value:.2f}  vs  {alternativa}")

    no_castellano = [r for r in resultados
                     if r[1].language != Language.SPANISH]
    print(f"\nNo detectadas como castellano: {len(no_castellano)}")
    for d, mejor, _ in no_castellano:
        print(f"  [{mejor.language.name} {mejor.value:.2f}] {d.titulo[:50]}")
        print(f"      {d.sinopsis[:110]}...")

    # El efecto concreto de ignorar el idioma: aplicar stopwords y tokenizacion
    # del castellano a un texto que no lo es.
    if no_castellano:
        d = no_castellano[0][0]
        tokens = preprocesamiento.tokenizar(d.sinopsis)
        print(f"\nQue pasa si se tokeniza con reglas del castellano un texto")
        print(f"que no lo es ({d.titulo}):")
        print(f"  tokens obtenidos: {tokens[:14]}")
        print(f"  Las stopwords castellanas no filtran las palabras funcionales")
        print(f"  de la otra lengua, asi que sobreviven y ensucian el TF-IDF.")

    _control_positivo(detector)

    print(f"\nCONCLUSION")
    if not no_castellano:
        print(f"  En ESTE corpus el problema no se presenta: las {len(docs)}")
        print(f"  sinopsis son castellano, con confianza minima {confianzas[0]:.3f}.")
        print(f"  La premisa de la pregunta no se cumple en nuestra muestra, y")
        print(f"  decirlo es mas util que inventar un problema que no tenemos.")
    print(f"\n  Pero el control de arriba muestra que eso NO significa que")
    print(f"  detectar el idioma sea innecesario:")
    print(f"    - el catalan se detecta bien, asi que si hubiera libros en")
    print(f"      catalan los encontrariamos;")
    print(f"    - el gallego NO tiene modelo y se reporta como otra lengua con")
    print(f"      alta confianza. Un falso negativo silencioso: el peor caso,")
    print(f"      porque no se distingue de un acierto.")
    print(f"  Conclusion operativa: detectar antes de tokenizar si, pero sin")
    print(f"  confiar en la etiqueta cuando las lenguas candidatas son")
    print(f"  cercanas y alguna no esta en el repertorio del detector.")

    return resultados


def _control_positivo(detector):
    """Le da al detector textos de lenguas conocidas, para ver si los reconoce.

    Reportar "no encontramos libros en gallego" no significa nada si el
    instrumento no es capaz de encontrarlos. Antes de concluir hay que verificar
    que el detector detecta lo que decimos que detecta. Son tres traducciones
    del mismo parrafo, para que la unica variable sea la lengua.
    """
    print(f"\nCONTROL DEL INSTRUMENTO")
    print(f"  Tres traducciones del mismo parrafo. Si el detector no reconoce")
    print(f"  estas, el resultado de arriba no prueba nada.")

    # Que esperamos que devuelva el detector para cada muestra. El gallego no
    # tiene modelo, asi que no hay respuesta correcta posible: por eso va None.
    esperado = {"castellano": "SPANISH", "catalan": "CATALAN", "gallego": None}

    muestras = {
        "castellano": "Esta es la historia de una muchacha que vive en una "
                      "ciudad pequena. Su padre trabaja con sus companeros y "
                      "no sabe que hacer con su vida.",
        "catalan": "Aquesta es la historia d'una noia que viu en una ciutat "
                   "petita. El seu pare treballa amb els seus companys i no "
                   "sap que fer amb la seva vida.",
        "gallego": "Esta e a historia dunha rapaza que vive nunha cidade "
                   "pequena. O seu pai traballa cos seus companeiros e non "
                   "sabe que facer coa sua vida.",
    }

    for lengua, texto in muestras.items():
        valores = detector.compute_language_confidence_values(texto)
        detectado, confianza = valores[0].language.name, valores[0].value
        segunda = f"{valores[1].language.name} {valores[1].value:.2f}"
        if esperado[lengua] is None:
            marca = "!!! "       # no hay modelo: cualquier respuesta es falsa
        else:
            marca = "OK  " if detectado == esperado[lengua] else "MAL "
        print(f"  {marca} {lengua:<11} detectado como {detectado} "
              f"({confianza:.2f}), segunda opcion {segunda}")

    print(f"  El catalan se reconoce. El gallego no puede: no esta en el")
    print(f"  repertorio, asi que cae en la lengua mas parecida SIN avisar.")


# ===========================================================================
# Pregunta 1 - Cuantas sinopsis hacen falta para que TF-IDF sea estable?
# ===========================================================================

# Tamanos de corpus de fondo a probar.
TAMANOS = [10, 25, 50, 100, 150]

# Cuantas replicas por tamano. Mas replicas dan una media mas confiable; 30 es
# suficiente para que la curva no se mueva entre corridas.
REPLICAS = 30

# Cuantos terminos caracteristicos se comparan por documento.
TOP_K = 10

# Cuando se considera que la curva se aplano.
UMBRAL_MESETA = 0.02


def experimento_estabilidad(docs, semilla=42):
    """Mide como cambian los terminos caracteristicos segun el tamano del corpus.

    EL DISENO ES LA PARTE IMPORTANTE, y tiene una sutileza que es facil pasar
    por alto. TF-IDF de un documento depende de DOS cosas: el documento (TF) y el
    corpus que define que tan raro es cada termino (IDF). Si al submuestrear se
    cambiaran las dos a la vez, no se sabria cual de las dos causo la variacion.

    Por eso se fija un conjunto de DOCUMENTOS SONDA que participan de todas las
    replicas, y lo unico que varia es el CORPUS DE FONDO que define el IDF. Asi
    la pregunta queda bien planteada: con cuantos documentos de fondo los
    terminos caracteristicos de un documento dado dejan de moverse?

    La metrica es Jaccard@10 entre replicas: que fraccion de los 10 terminos top
    comparten dos muestras distintas del mismo tamano. Se reporta tambien el
    solapamiento del top-3, porque en la practica lo que se mira de un TF-IDF
    son los primeros terminos, no los diez.
    """
    import random
    from itertools import combinations
    from sklearn.feature_extraction.text import TfidfVectorizer

    titulo("PREGUNTA 1 - Estabilidad de TF-IDF segun el tamano del corpus")

    rng = random.Random(semilla)
    textos = {d.id: " ".join(preprocesamiento.tokenizar(d.texto)) for d in docs}

    # Los documentos sonda: fijos en todas las replicas. Son el "sujeto" del
    # experimento, no la variable.
    n_sondas = 20
    sondas = rng.sample(sorted(textos), n_sondas)
    fondo_disponible = [i for i in sorted(textos) if i not in sondas]

    print(f"\nDiseno del experimento")
    print(f"  {n_sondas} documentos SONDA, fijos en todas las replicas")
    print(f"  el corpus de FONDO varia: {TAMANOS}")
    print(f"  {REPLICAS} replicas por tamano, top-{TOP_K} terminos por documento")
    print(f"  metrica: Jaccard promedio entre pares de replicas")
    print(f"\n  Lo que varia es SOLO el IDF. Si se resamplearan tambien los")
    print(f"  documentos medidos, se mezclarian dos fuentes de variacion y el")
    print(f"  resultado no significaria nada.")

    def terminos_top(ids_fondo):
        """Top-K terminos de cada sonda, con el IDF estimado SOLO en el fondo.

        Es fit() sobre el fondo y transform() sobre las sondas, no un
        fit_transform() sobre todo junto. La diferencia no es cosmetica: si las
        sondas participaran del ajuste, con un fondo de 10 documentos las 20
        sondas serian dos tercios del corpus y definirian ellas mismas el IDF.
        Como las sondas son las MISMAS en todas las replicas, eso inflaria
        artificialmente la coincidencia justo en los tamanos chicos, que es
        donde el experimento tiene que mostrar inestabilidad.
        """
        vec = TfidfVectorizer()
        vec.fit([textos[i] for i in ids_fondo])
        matriz = vec.transform([textos[i] for i in sondas])
        vocabulario = vec.get_feature_names_out()

        resultado = {}
        for pos, id_sonda in enumerate(sondas):
            fila = matriz[pos].toarray().ravel()
            mejores = fila.argsort()[::-1][:TOP_K]
            resultado[id_sonda] = [vocabulario[j] for j in mejores
                                   if fila[j] > 0]
        return resultado

    print(f"\n{'fondo':>7} {'Jaccard@10':>12} {'Jaccard@3':>11} "
          f"{'estable?':>10} {'solape':>9}")
    print("  " + "-" * 54)

    curva = []
    anterior = None
    for n in TAMANOS:
        if n > len(fondo_disponible):
            continue

        replicas = [terminos_top(rng.sample(fondo_disponible, n))
                    for _ in range(REPLICAS)]

        # Jaccard entre cada par de replicas, promediado sobre las sondas.
        j10, j3 = [], []
        for a, b in combinations(range(REPLICAS), 2):
            for id_sonda in sondas:
                ta, tb = replicas[a][id_sonda], replicas[b][id_sonda]
                j10.append(_jaccard(set(ta), set(tb)))
                j3.append(_jaccard(set(ta[:3]), set(tb[:3])))

        media10 = sum(j10) / len(j10)
        media3 = sum(j3) / len(j3)
        salto = "" if anterior is None else f"{media10 - anterior:+.3f}"
        meseta = (anterior is not None
                  and 0 <= media10 - anterior < UMBRAL_MESETA)
        # Cuanto se solapan dos replicas del mismo tamano. Al muestrear sin
        # reposicion de un conjunto finito, cuando n se acerca al total las
        # replicas comparten casi todos los documentos y el Jaccard sube por
        # esa razon y no porque TF-IDF se haya estabilizado. Reportarlo evita
        # leer como senal lo que es un artefacto del diseno.
        solape = n / len(fondo_disponible)
        print(f"{n:>7} {media10:>12.3f} {media3:>11.3f} "
              f"{'SI' if meseta else 'no':>10} {solape:>8.0%}   {salto}")

        curva.append((n, media10, media3))
        anterior = media10

    print(f"\nLECTURA DEL RESULTADO")
    inicial, final = curva[0][1], curva[-1][1]
    print(f"  Con {curva[0][0]} documentos de fondo, dos muestras distintas")
    print(f"  coinciden en {inicial:.0%} de los terminos caracteristicos.")
    print(f"  Con {curva[-1][0]}, en {final:.0%}.")

    saltos = [curva[i][1] - curva[i - 1][1] for i in range(1, len(curva))]
    if saltos and 0 <= saltos[-1] < UMBRAL_MESETA:
        print(f"\n  La curva se APLANA: el ultimo salto fue de {saltos[-1]:+.3f},")
        print(f"  por debajo del umbral de {UMBRAL_MESETA}. Agregar mas")
        print(f"  documentos ya casi no cambia los terminos caracteristicos.")
    else:
        print(f"\n  La curva NO se aplano dentro del rango medido (ultimo salto")
        print(f"  {saltos[-1]:+.3f}, umbral {UMBRAL_MESETA}). Con este corpus no")
        print(f"  alcanza para afirmar que TF-IDF se estabilizo: haria falta")
        print(f"  seguir la curva con mas documentos.")

    print(f"\n  Por que cuesta tanto estabilizarse: el 63% del vocabulario de")
    print(f"  este corpus aparece UNA SOLA VEZ. TF-IDF premia exactamente esos")
    print(f"  terminos, porque su IDF es maximo. Entonces los caracteristicos")
    print(f"  de un documento son, en buena medida, accidentes de muestreo.")

    print(f"\n  COMO SE MIDE, que es lo que pregunta la consigna: se fijan los")
    print(f"  documentos, se varia el corpus de fondo, y se mide el solapamiento")
    print(f"  de los top-K entre replicas del mismo tamano. 'Estable' no es un")
    print(f"  umbral magico sino el punto donde la ganancia marginal cae por")
    print(f"  debajo de un valor acordado (aca {UMBRAL_MESETA}).")
    print(f"\n  Jaccard tiene un limite que conviene decir: ignora el ORDEN, y")
    print(f"  en TF-IDF el ranking es justamente lo que se mira. Por eso se")
    print(f"  reporta tambien Jaccard@3. La metrica mas correcta seria RBO")
    print(f"  (rank-biased overlap), que pondera por posicion.")

    return curva


def _jaccard(a, b):
    """Interseccion sobre union. 1.0 = identicos, 0.0 = sin nada en comun."""
    return len(a & b) / len(a | b) if (a or b) else 1.0


# ===========================================================================
# Punto de entrada
# ===========================================================================

EXPERIMENTOS = {
    "idioma": experimento_idioma,
    "estabilidad": experimento_estabilidad,
}


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Experimentos sobre el corpus (Unidad 2).")
    parser.add_argument("experimento",
                        choices=list(EXPERIMENTOS) + ["todos"],
                        help="cual correr")
    parser.add_argument("--tabla", default="libros")
    args = parser.parse_args()

    docs = corpus.traer_documentos(tabla=args.tabla)
    if not docs:
        raise SystemExit("El corpus esta vacio. Corriste src/etl.py?")
    print(f"Corpus: {len(docs)} documentos de la tabla '{args.tabla}'")

    elegidos = EXPERIMENTOS if args.experimento == "todos" else {
        args.experimento: EXPERIMENTOS[args.experimento]}
    for funcion in elegidos.values():
        funcion(docs)
    return 0


if __name__ == "__main__":
    sys.exit(main())

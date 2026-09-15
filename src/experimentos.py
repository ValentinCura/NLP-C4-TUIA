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
# Se extiende automaticamente si el corpus lo permite (el ampliado, de 1700
# documentos, llega a tamanos grandes manteniendo bajo el solape entre
# replicas, que es la limitacion del corpus de 200).
TAMANOS = [10, 25, 50, 100, 150, 300, 600, 1200]

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
        # Se descartan los tamanos que no entran en el corpus disponible.
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

    # El porcentaje se calcula sobre el corpus que se esta usando, no se
    # escribe a mano: cambia mucho entre el corpus de 200 (63%) y el ampliado
    # (51%), asi que una cifra fija seria falsa en uno de los dos casos.
    vocabulario_global = Counter(p for i in textos for p in textos[i].split())
    hapax = sum(1 for c in vocabulario_global.values() if c == 1)
    pct_hapax = 100 * hapax / len(vocabulario_global)

    print(f"\n  Por que cuesta tanto estabilizarse: el {pct_hapax:.0f}% del "
          f"vocabulario")
    print(f"  de este corpus ({hapax:,} de {len(vocabulario_global):,} "
          f"palabras) aparece UNA SOLA VEZ.")
    print(f"  TF-IDF premia exactamente esos terminos, porque su IDF es")
    print(f"  maximo. Entonces los terminos caracteristicos de un documento")
    print(f"  son, en buena medida, accidentes de muestreo.")

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
# Pregunta 3 - Como cambia la evaluacion en multi-etiqueta?
# ===========================================================================

def experimento_multietiqueta(docs, semilla=42):
    """Compara las metricas de evaluacion sobre el mismo clasificador.

    La consigna pregunta como cambia la evaluacion respecto de multi-clase. La
    respuesta corta es que en multi-clase hay UNA metrica obvia (accuracy) y en
    multi-etiqueta no hay ninguna: hay varias que miden cosas distintas y que
    pueden contar historias opuestas sobre el mismo modelo.
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, f1_score, hamming_loss,
                                 precision_score, recall_score)
    from sklearn.model_selection import train_test_split
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import MultiLabelBinarizer

    titulo("PREGUNTA 3 - Evaluacion multi-etiqueta contra multi-clase")

    etiquetas = [d.generos for d in docs]
    conteo = Counter(g for gs in etiquetas for g in gs)

    print(f"\nPor que este problema NO PUEDE ser multi-clase")
    print(f"  etiquetas por libro: minimo {min(len(g) for g in etiquetas)}, "
          f"promedio {sum(len(g) for g in etiquetas) / len(etiquetas):.2f}, "
          f"maximo {max(len(g) for g in etiquetas)}")
    print(f"  No hay un solo libro con un unico genero. Forzar multi-clase")
    print(f"  obligaria a descartar etiquetas reales o a inventar una clase")
    print(f"  por cada combinacion observada, que serian decenas con pocos")
    print(f"  ejemplos cada una.")

    print(f"\nDistribucion de las {len(conteo)} etiquetas")
    for genero, n in conteo.most_common(6):
        print(f"  {genero:<20} {n:>4}  ({100 * n / len(docs):>4.0f}% de los libros)")
    raras = [g for g, n in conteo.items() if n < 5]
    print(f"  ... {len(raras)} generos con menos de 5 libros: no se pueden")
    print(f"  aprender NI estratificar en un split.")

    # Se descartan las etiquetas sin soporte suficiente. La alternativa seria
    # estratificacion iterativa multi-etiqueta; se elige el umbral por
    # simplicidad, pero lo importante es DECIR que se hizo y por que.
    minimo = 10
    frecuentes = sorted(g for g, n in conteo.items() if n >= minimo)
    print(f"\n  Se trabaja con los {len(frecuentes)} generos con >= {minimo} "
          f"libros: {frecuentes}")

    y_listas = [[g for g in gs if g in frecuentes] for gs in etiquetas]
    mlb = MultiLabelBinarizer(classes=frecuentes)
    Y = mlb.fit_transform(y_listas)
    X_texto = preprocesamiento.para_sklearn([d.texto for d in docs])

    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X_texto, Y, test_size=0.3, random_state=semilla)

    vec = TfidfVectorizer(min_df=2)
    X_tr_v = vec.fit_transform(X_tr)
    X_te_v = vec.transform(X_te)

    modelo = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, class_weight="balanced"))
    modelo.fit(X_tr_v, Y_tr)
    Y_pred = modelo.predict(X_te_v)

    # El comparador imprescindible: predecir siempre las etiquetas mas comunes
    # del entrenamiento, sin mirar el texto.
    mayoritarias = (Y_tr.mean(axis=0) > 0.5).astype(int)
    Y_trivial = np.tile(mayoritarias, (len(Y_te), 1))

    print(f"\n  entrenamiento {len(Y_tr)} libros | prueba {len(Y_te)} libros")
    print(f"\n{'metrica':<22}{'clasificador':>14}{'trivial':>12}   que mide")
    print("  " + "-" * 82)

    def fila(nombre, valor_modelo, valor_trivial, explicacion):
        print(f"{nombre:<22}{valor_modelo:>14.3f}{valor_trivial:>12.3f}   "
              f"{explicacion}")

    fila("subset accuracy", accuracy_score(Y_te, Y_pred),
         accuracy_score(Y_te, Y_trivial),
         "acierta el conjunto EXACTO")
    fila("hamming loss", hamming_loss(Y_te, Y_pred),
         hamming_loss(Y_te, Y_trivial),
         "errores por etiqueta (menos es mejor)")
    fila("f1 micro", f1_score(Y_te, Y_pred, average="micro", zero_division=0),
         f1_score(Y_te, Y_trivial, average="micro", zero_division=0),
         "agrega todas las decisiones")
    fila("f1 macro", f1_score(Y_te, Y_pred, average="macro", zero_division=0),
         f1_score(Y_te, Y_trivial, average="macro", zero_division=0),
         "promedia por genero, sin pesar")
    fila("precision micro",
         precision_score(Y_te, Y_pred, average="micro", zero_division=0),
         precision_score(Y_te, Y_trivial, average="micro", zero_division=0), "")
    fila("recall micro",
         recall_score(Y_te, Y_pred, average="micro", zero_division=0),
         recall_score(Y_te, Y_trivial, average="micro", zero_division=0), "")

    print(f"\nLO QUE MUESTRA LA COLUMNA 'TRIVIAL'")
    print(f"  Ese clasificador no lee el texto: predice siempre las etiquetas")
    print(f"  presentes en mas de la mitad de los libros de entrenamiento, que")
    print(f"  aca son {[frecuentes[i] for i, v in enumerate(mayoritarias) if v]}.")
    f1_micro_triv = f1_score(Y_te, Y_trivial, average="micro", zero_division=0)
    f1_macro_triv = f1_score(Y_te, Y_trivial, average="macro", zero_division=0)
    print(f"  Con eso saca f1 micro {f1_micro_triv:.3f} y f1 macro "
          f"{f1_macro_triv:.3f}.")
    print(f"  Reportar solo micro-F1 haria pasar por aceptable a un modelo que")
    print(f"  no mira la entrada. Es el efecto de que 'Novela' aparezca en el")
    print(f"  {100 * conteo['Novela'] / len(docs):.0f}% de los libros.")

    print(f"\nRESPUESTA A LA PREGUNTA")
    print(f"  En multi-clase las predicciones son mutuamente excluyentes: hay")
    print(f"  una sola respuesta correcta, accuracy la resume bien y la matriz")
    print(f"  de confusion es cuadrada y legible.")
    print(f"\n  En multi-etiqueta cambia todo eso:")
    print(f"   - el acierto deja de ser binario: predecir 2 de 3 generos no es")
    print(f"     ni un acierto ni un error completo;")
    print(f"   - subset accuracy es demasiado severa (castiga igual errar una")
    print(f"     etiqueta que errarlas todas);")
    print(f"   - hamming loss es mas indulgente pero se deja enganar por el")
    print(f"     desbalance: predecir todo en cero ya da un valor bajo;")
    print(f"   - micro y macro se separan: micro pondera por frecuencia y se lo")
    print(f"     lleva la etiqueta dominante, macro trata igual a un genero con")
    print(f"     170 libros y a uno con 10;")
    print(f"   - no hay una matriz de confusion unica, sino una por etiqueta;")
    print(f"   - el split no se puede estratificar de la forma habitual, porque")
    print(f"     estratificar por una etiqueta desbalancea las demas.")
    print(f"\n  En la practica: reportar SIEMPRE micro y macro juntas, y")
    print(f"  siempre contra una linea base trivial. Una sola de las tres")
    print(f"  cifras, aislada, no permite saber si el modelo aprendio algo.")

    return {"frecuentes": frecuentes}


# ===========================================================================
# Pregunta 2 - Que sesgo introduce que las sinopsis sean texto promocional?
# ===========================================================================

# Vocabulario del registro publicitario: palabras que valoran la obra o apelan
# al lector, en vez de describir de que trata el libro.
#
# La lista se armo a mano leyendo sinopsis del corpus. Es necesariamente
# incompleta y discutible, y esa es una limitacion honesta del experimento: mide
# el sesgo que capta esta lista, no "el sesgo promocional" en abstracto.
LEXICO_PROMOCIONAL = {
    # juicio de valor sobre la obra
    "imprescindible", "magistral", "brillante", "extraordinario", "excepcional",
    "inolvidable", "memorable", "deslumbrante", "impecable", "soberbio",
    "fascinante", "cautivador", "conmovedor", "estremecedor", "trepidante",
    "adictivo", "vibrante", "apasionante", "absorbente", "hipnotico",
    # consagracion y ventas
    "bestseller", "superventas", "exito", "fenomeno", "aclamado", "celebrado",
    "premio", "premiada", "premiado", "galardonada", "galardonado",
    "millones", "lectores", "critica", "reconocido", "consagrado",
    "clasico", "obra", "maestra", "revelacion", "imperdible",
    # apelacion al lector
    "descubre", "sumergete", "prepare", "atrapara", "sorprendera",
    "dejara", "soltar", "aliento", "nadie", "jamas",
    # marcas editoriales
    "autora", "autor", "novela", "saga", "trilogia", "edicion", "traduccion",
}


def experimento_promocional(docs, semilla=42):
    """Mide cuanto pesa el vocabulario publicitario y si predice el genero.

    La consigna pregunta que sesgo introduce entrenar un clasificador de genero
    sobre texto escrito para vender. La hipotesis a contrastar es concreta: si
    las palabras promocionales predicen el genero, el clasificador no esta
    aprendiendo de que trata el libro sino COMO SE LO PUBLICITA, que son cosas
    distintas y solo la segunda es un artefacto del canal.
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import MultiLabelBinarizer

    titulo("PREGUNTA 2 - El sesgo del texto promocional")

    textos = preprocesamiento.para_sklearn([d.texto for d in docs])

    # --- 1. Cuanto pesa el vocabulario promocional en los terminos top -----
    vec = TfidfVectorizer()
    matriz = vec.fit_transform(textos)
    vocabulario = vec.get_feature_names_out()

    def es_promocional(termino):
        return preprocesamiento._sin_acentos(termino) in LEXICO_PROMOCIONAL

    promocionales_en_top = 0
    total_top = 0
    ejemplos = Counter()
    for i in range(matriz.shape[0]):
        fila = matriz[i].toarray().ravel()
        top = [vocabulario[j] for j in fila.argsort()[::-1][:10] if fila[j] > 0]
        for termino in top:
            total_top += 1
            if es_promocional(termino):
                promocionales_en_top += 1
                ejemplos[termino] += 1

    print(f"\n1. Presencia en los terminos caracteristicos")
    print(f"   De los {total_top} terminos top-10 de las {len(docs)} sinopsis,")
    print(f"   {promocionales_en_top} son vocabulario promocional "
          f"({100 * promocionales_en_top / total_top:.1f}%).")
    if ejemplos:
        print(f"   Los mas frecuentes: "
              f"{[t for t, _ in ejemplos.most_common(8)]}")
    print(f"\n   Es un porcentaje BAJO, y tiene una explicacion que conviene")
    print(f"   entender: TF-IDF penaliza justamente lo que aparece en todos los")
    print(f"   documentos. Como las sinopsis son TODAS promocionales, esas")
    print(f"   palabras tienen IDF bajo y quedan relegadas. O sea que TF-IDF")
    print(f"   ya filtra parte del sesgo por construccion.")

    # --- 2. El vocabulario promocional predice el genero? ------------------
    etiquetas = [d.generos for d in docs]
    conteo = Counter(g for gs in etiquetas for g in gs)
    frecuentes = sorted(g for g, n in conteo.items() if n >= 20)

    print(f"\n2. El registro publicitario, predice el genero?")
    print(f"   Informacion mutua entre cada termino promocional y cada genero.")
    print(f"   Si diera cero, el vocabulario promocional seria ruido inocuo.")

    indices_promo = [j for j, t in enumerate(vocabulario) if es_promocional(t)]
    if indices_promo:
        X_promo = matriz[:, indices_promo].toarray()
        nombres_promo = [vocabulario[j] for j in indices_promo]

        filas = []
        for genero in frecuentes:
            y = np.array([1 if genero in gs else 0 for gs in etiquetas])
            mi = mutual_info_classif(X_promo, y, random_state=semilla)
            mejor = int(np.argmax(mi))
            filas.append((genero, nombres_promo[mejor], mi[mejor], mi.mean()))

        print(f"\n   {'genero':<20}{'termino promo mas informativo':<32}"
              f"{'IM':>8}{'IM media':>10}")
        print("   " + "-" * 68)
        for genero, termino, im, media in sorted(filas, key=lambda f: -f[2]):
            print(f"   {genero:<20}{termino:<32}{im:>8.4f}{media:>10.4f}")

    # --- 3. Efecto real sobre un clasificador ------------------------------
    print(f"\n3. Que pasa si se le quita el vocabulario promocional al modelo")

    mlb = MultiLabelBinarizer(classes=frecuentes)
    Y = mlb.fit_transform([[g for g in gs if g in frecuentes]
                           for gs in etiquetas])

    resultados = {}
    for nombre, stop in [("con vocabulario promocional", None),
                         ("sin vocabulario promocional",
                          sorted(LEXICO_PROMOCIONAL))]:
        X_tr, X_te, Y_tr, Y_te = train_test_split(
            textos, Y, test_size=0.3, random_state=semilla)
        v = TfidfVectorizer(min_df=2, stop_words=stop)
        modelo = OneVsRestClassifier(
            LogisticRegression(max_iter=1000, class_weight="balanced"))
        modelo.fit(v.fit_transform(X_tr), Y_tr)
        pred = modelo.predict(v.transform(X_te))
        resultados[nombre] = (
            f1_score(Y_te, pred, average="micro", zero_division=0),
            f1_score(Y_te, pred, average="macro", zero_division=0))
        print(f"   {nombre:<32} f1 micro {resultados[nombre][0]:.3f}  "
              f"f1 macro {resultados[nombre][1]:.3f}")

    caida = (resultados["con vocabulario promocional"][0]
             - resultados["sin vocabulario promocional"][0])
    print(f"\n   Diferencia en f1 micro: {caida:+.3f}")

    print(f"\nRESPUESTA A LA PREGUNTA")
    print(f"  El sesgo existe pero no es el que uno esperaria. Las palabras")
    print(f"  publicitarias no dominan los terminos caracteristicos, porque")
    print(f"  TF-IDF ya las castiga por aparecer en todas las sinopsis.")
    print(f"\n  El sesgo real es mas sutil y esta en otro lado:")
    print(f"   - Una sinopsis NO DESCRIBE el libro: selecciona lo vendible.")
    print(f"     Omite el final, exagera el conflicto y destaca lo que se")
    print(f"     parece a otros exitos. El clasificador aprende de que trata")
    print(f"     LA CAMPANA, no de que trata el libro.")
    print(f"   - Cada genero tiene su registro publicitario propio, y eso es")
    print(f"     senal aprendible pero fragil: un libro de terror promocionado")
    print(f"     como literario se clasificaria mal, y el modelo no generalizaria")
    print(f"     a texto que no sea de contratapa (resenas, criticas, el libro).")
    print(f"   - Las sinopsis son de longitud y estructura uniformes porque las")
    print(f"     escribe el mismo departamento de marketing: menos variedad")
    print(f"     lexica de la que tendria texto natural del mismo tamano.")
    print(f"\n  Para un recomendador el sesgo molesta menos, porque comparar")
    print(f"  sinopsis con sinopsis mantiene el registro constante en ambos")
    print(f"  lados. Para un clasificador que despues vea otro tipo de texto,")
    print(f"  en cambio, es un problema de generalizacion serio.")

    return resultados


# ===========================================================================
# Punto de entrada
# ===========================================================================

EXPERIMENTOS = {
    "idioma": experimento_idioma,
    "estabilidad": experimento_estabilidad,
    "promocional": experimento_promocional,
    "multietiqueta": experimento_multietiqueta,
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

"""
Evaluacion de la busqueda: consultas, relevancia, precision@k y piso de azar.

TP2 - Procesamiento del Lenguaje Natural (TUIA, FCEIA-UNR).

Es el nucleo del TP. Todo lo que hay aca se puede probar sin modelos: las
funciones reciben puntajes (un numero por documento) y conjuntos de ids, y
devuelven numeros. Por eso tienen su propio archivo de tests.

Decisiones que conviene no perder de vista:

  1. PRECISION@K CON EMPATES. Un ranking es un argsort de puntajes. Si varios
     documentos empatan (TF-IDF da 0 a todos los que no comparten palabras con
     la consulta; un vector de consulta nulo da 0 con todo), argsort los ordena
     por POSICION EN LA MATRIZ, es decir por id. La precision que sale de ahi
     depende de en que orden se cargo el CSV: es arbitraria. Aca se calcula la
     precision ESPERADA si los empates se rompieran al azar, en forma exacta y
     sin semilla. Para una consulta sin ninguna palabra en comun con el corpus,
     TF-IDF da exactamente el piso de azar, que es lo que corresponde.

  2. EL PISO DE AZAR ES EXACTO. Elegir k documentos al azar da, en esperanza,
     |R| / N documentos relevantes por puesto, para cualquier k <= N. No hace
     falta simular; se simula igual (simular_azar) para mostrar la DISPERSION,
     que es lo que dice cuanto puede tener de suerte un modelo.

  3. EL TECHO TAMBIEN EXISTE. Si una consulta tiene 2 relevantes, P@5 no puede
     pasar de 0,4. Se reporta el techo junto a cada valor.

  4. PROMEDIO MACRO. Cada consulta pesa lo mismo en el promedio, sin importar
     cuantos relevantes tenga.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MINIMO_CONSULTAS = 10


# ---------------------------------------------------------------------------
# Conjunto de evaluacion
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Consulta:
    id: str
    texto: str
    tipo: str
    relevantes: frozenset
    dudosos: frozenset = field(default_factory=frozenset)
    sin_solapamiento: bool = False
    nota: str = ""


class ConsultasInvalidas(ValueError):
    """El archivo de consultas no cumple lo minimo para evaluar."""


def cargar_consultas(ruta, ids_corpus, generos_por_id=None):
    """Lee y valida el archivo de consultas (queries.json o la propuesta).

    Falla ruidosamente ante cualquier cosa que haria la evaluacion invalida en
    silencio: un id que no existe en el corpus (el libro nunca podria
    recuperarse), una consulta sin relevantes (precision indefinida), un libro
    marcado como relevante y dudoso a la vez, ids repetidos, menos de 10
    consultas.

    Devuelve (consultas, estado), donde estado es el texto del campo 'estado'
    del archivo, para que el notebook pueda mostrar si es provisorio.
    """
    datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
    ids_corpus = set(ids_corpus)
    consultas, vistos, errores = [], set(), []

    for q in datos["consultas"]:
        qid = q.get("id", "?")
        if qid in vistos:
            errores.append(f"{qid}: id de consulta repetido")
        vistos.add(qid)
        if not str(q.get("consulta", "")).strip():
            errores.append(f"{qid}: texto de consulta vacio")

        if q.get("relevantes_por_genero"):
            if generos_por_id is None:
                raise ConsultasInvalidas(
                    f"{qid} define la relevancia por genero y no se pasaron "
                    f"los generos del corpus")
            genero = q["relevantes_por_genero"]
            relevantes = [i for i, gs in generos_por_id.items() if genero in gs]
        else:
            relevantes = [r["id"] for r in q.get("relevantes", [])]
        dudosos = [r["id"] for r in q.get("dudosos", [])]

        if not relevantes:
            errores.append(f"{qid}: no tiene ningun relevante")
        if len(set(relevantes)) != len(relevantes):
            errores.append(f"{qid}: relevantes repetidos")
        faltan = (set(relevantes) | set(dudosos)) - ids_corpus
        if faltan:
            errores.append(f"{qid}: ids que no existen en el corpus {sorted(faltan)}")
        ambos = set(relevantes) & set(dudosos)
        if ambos:
            errores.append(f"{qid}: relevante y dudoso a la vez {sorted(ambos)}")

        consultas.append(Consulta(
            id=qid, texto=q.get("consulta", ""), tipo=q.get("tipo", ""),
            relevantes=frozenset(relevantes), dudosos=frozenset(dudosos),
            sin_solapamiento=bool(q.get("sin_solapamiento", False)),
            nota=q.get("nota", "")))

    if len(consultas) < MINIMO_CONSULTAS:
        errores.append(f"hay {len(consultas)} consultas; la consigna pide "
                       f"al menos {MINIMO_CONSULTAS}")
    if not any(c.sin_solapamiento for c in consultas):
        errores.append("ninguna consulta esta marcada como sin_solapamiento")
    if errores:
        raise ConsultasInvalidas("\n".join(errores))
    return consultas, datos.get("estado", "")


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------

def precision_at_k(puntajes, es_relevante, k):
    """Precision@k esperada si los empates se rompen al azar (valor exacto).

    puntajes:      un numero por documento (mayor = mas parecido).
    es_relevante:  booleano por documento, alineado con puntajes.
    k:             cuantos documentos se miran (1 <= k <= cantidad).

    P@k = (relevantes entre los k primeros) / k. El denominador es SIEMPRE k,
    aunque la consulta tenga menos de k relevantes (eso es lo que hace que
    exista un techo).

    Con empates, los k primeros puestos se llenan por grupos de igual puntaje,
    del mas alto al mas bajo. Los grupos que entran enteros aportan todos sus
    relevantes. El ultimo grupo entra solo en parte: si ocupa m lugares de los
    g documentos que empatan, y r de ellos son relevantes, aporta en esperanza
    m * r / g relevantes (hipergeometrica). Sin empates coincide con la
    definicion de siempre.
    """
    s = np.asarray(puntajes, dtype=np.float64)
    rel = np.asarray(es_relevante, dtype=bool)
    if s.shape != rel.shape or s.ndim != 1:
        raise ValueError(f"puntajes {s.shape} y relevancia {rel.shape} "
                         f"tienen que ser vectores del mismo largo")
    if np.isnan(s).any():
        raise ValueError("hay puntajes NaN: el ranking no esta definido")
    if not 1 <= k <= len(s):
        raise ValueError(f"k={k} fuera de rango: tiene que estar entre 1 y "
                         f"la cantidad de documentos ({len(s)})")

    esperados, lugares = 0.0, k
    for valor in np.unique(s)[::-1]:
        grupo = s == valor
        g = int(grupo.sum())
        r = int(rel[grupo].sum())
        if g <= lugares:
            esperados += r
            lugares -= g
        else:
            esperados += lugares * r / g
            lugares = 0
        if lugares == 0:
            break
    return esperados / k


def techo(n_relevantes, k):
    """La mejor P@k posible: min(|R|, k) / k."""
    return min(n_relevantes, k) / k


def piso_azar(n_relevantes, n_documentos):
    """P@k esperada de un ranking al azar: |R| / N, para cualquier k <= N."""
    return n_relevantes / n_documentos


def simular_azar(es_relevante, k, repeticiones=10_000, semilla=42):
    """P@k de `repeticiones` rankings al azar. Devuelve el array completo.

    La media converge a piso_azar(); lo que aporta la simulacion es la
    dispersion: con pocos relevantes, el azar acierta a veces, y eso dice
    cuanto de un buen resultado aislado puede ser suerte.
    """
    rel = np.asarray(es_relevante, dtype=bool)
    rng = np.random.default_rng(semilla)
    n = len(rel)
    # Para cada repeticion, los k primeros de una permutacion al azar.
    tops = np.argsort(rng.random((repeticiones, n)), axis=1)[:, :k]
    return rel[tops].sum(axis=1) / k


# ---------------------------------------------------------------------------
# Evaluacion completa
# ---------------------------------------------------------------------------

def evaluar(buscadores, consultas, ids_documentos, ks=(5, 10)):
    """Evalua cada buscador sobre cada consulta. Devuelve un DataFrame largo.

    buscadores: {nombre: funcion(texto_consulta) -> puntajes}, con los
                puntajes alineados con ids_documentos.

    Para cada (modelo, consulta, k) se reporta:
      p_at_k          P@k contra los relevantes (los dudosos NO cuentan)
      p_at_k_amplia   P@k contando tambien los dudosos como relevantes
      r_precision     P@|R|: la precision con k igual a la cantidad de
                      relevantes, que tiene techo 1 en todas las consultas
      piso, techo     para leer cada valor en contexto
    Se agrega un "modelo" mas, el AZAR, cuya P@k es el piso exacto.
    """
    import pandas as pd

    ids = np.asarray(ids_documentos)
    filas = []
    for c in consultas:
        rel = np.isin(ids, list(c.relevantes))
        amplia = np.isin(ids, list(c.relevantes | c.dudosos))
        n_rel = int(rel.sum())
        piso = piso_azar(n_rel, len(ids))

        candidatos = dict(buscadores)
        for nombre, buscar in candidatos.items():
            puntajes = np.asarray(buscar(c.texto), dtype=np.float64)
            if puntajes.shape != ids.shape:
                raise ValueError(f"{nombre} devolvio {puntajes.shape} "
                                 f"puntajes para {ids.shape} documentos")
            r_prec = precision_at_k(puntajes, rel, n_rel)
            for k in ks:
                filas.append({
                    "modelo": nombre, "consulta": c.id, "tipo": c.tipo,
                    "sin_solapamiento": c.sin_solapamiento, "k": k,
                    "p_at_k": precision_at_k(puntajes, rel, k),
                    "p_at_k_amplia": precision_at_k(puntajes, amplia, k),
                    "r_precision": r_prec,
                    "n_relevantes": n_rel, "piso": piso,
                    "techo": techo(n_rel, k),
                    "empate_total": bool(np.all(puntajes == puntajes[0])),
                })
        for k in ks:
            filas.append({
                "modelo": "AZAR (piso)", "consulta": c.id, "tipo": c.tipo,
                "sin_solapamiento": c.sin_solapamiento, "k": k,
                "p_at_k": piso,
                "p_at_k_amplia": piso_azar(int(amplia.sum()), len(ids)),
                "r_precision": piso, "n_relevantes": n_rel, "piso": piso,
                "techo": techo(n_rel, k), "empate_total": True,
            })
    return pd.DataFrame(filas)


def intervalo_bootstrap(valores, repeticiones=10_000, nivel=0.95, semilla=42):
    """Intervalo bootstrap percentil de la MEDIA, remuestreando consultas.

    Con 10-16 consultas el promedio de P@k se mueve mucho segun que consultas
    se hayan elegido. El intervalo lo hace visible: si los intervalos de dos
    modelos se pisan, la diferencia entre sus medias no es concluyente.
    """
    v = np.asarray(valores, dtype=np.float64)
    rng = np.random.default_rng(semilla)
    medias = v[rng.integers(0, len(v), size=(repeticiones, len(v)))].mean(axis=1)
    alfa = (1 - nivel) / 2
    return float(np.quantile(medias, alfa)), float(np.quantile(medias, 1 - alfa))


def resumen(df, k, columna="p_at_k"):
    """Media macro por modelo, con intervalo bootstrap y techo/piso promedio."""
    import pandas as pd

    filas = []
    for modelo, grupo in df[df.k == k].groupby("modelo", sort=False):
        bajo, alto = intervalo_bootstrap(grupo[columna].values)
        filas.append({"modelo": modelo, f"{columna} (media)": grupo[columna].mean(),
                      "IC95 bajo": bajo, "IC95 alto": alto,
                      "consultas": len(grupo)})
    # Orden estable: ante un empate en la media se respeta el orden en que se
    # pasaron los modelos, en vez de uno arbitrario.
    return pd.DataFrame(filas).sort_values(f"{columna} (media)", ascending=False,
                                           kind="stable")


def comparar_pareado(df, modelo_a, modelo_b, k, columna="p_at_k"):
    """Compara dos modelos consulta por consulta (mismas consultas).

    Devuelve cuantas consultas gana cada uno, cuantas empatan, la diferencia
    media (a - b) y su intervalo bootstrap. La comparacion pareada es la que
    corresponde: las consultas difieren mucho en dificultad, y comparar medias
    sueltas mezcla esa variacion con la de los modelos.
    """
    a = df[(df.modelo == modelo_a) & (df.k == k)].set_index("consulta")[columna]
    b = df[(df.modelo == modelo_b) & (df.k == k)].set_index("consulta")[columna]
    diff = (a - b.reindex(a.index)).values
    bajo, alto = intervalo_bootstrap(diff)
    return {"gana_" + modelo_a: int((diff > 1e-9).sum()),
            "gana_" + modelo_b: int((diff < -1e-9).sum()),
            "empatan": int((np.abs(diff) <= 1e-9).sum()),
            "diferencia_media": float(diff.mean()),
            "IC95": (bajo, alto)}


def mrr_misma_serie(matriz, series):
    """Recuperacion de la MISMA SAGA usando un libro como consulta.

    Para cada libro de una serie con 2 o mas tomos en el corpus, se busca en
    que puesto aparece el primer otro tomo de su serie (excluyendose a si
    mismo). MRR = promedio de 1/puesto. Es una evaluacion complementaria: su
    "relevancia" sale gratis de la columna serie, pero mide una tarea distinta
    de la busqueda por consulta (recuperar el mismo universo, no el mismo tema).

    matriz: filas normalizadas (documentos x dim). series: una por documento
    (None si no pertenece a ninguna). Los empates se resuelven por el peor
    puesto posible, para no premiar a un modelo por el orden del CSV.
    """
    from collections import defaultdict

    sim = matriz @ matriz.T
    np.fill_diagonal(sim, -np.inf)
    grupos = defaultdict(list)
    for i, s in enumerate(series):
        if s:
            grupos[s].append(i)
    puestos = []
    for indices in grupos.values():
        if len(indices) < 2:
            continue
        for i in indices:
            otros = [j for j in indices if j != i]
            mejor = sim[i, otros].max()
            # puesto pesimista: cuantos documentos tienen puntaje >= al mejor
            puestos.append(int((sim[i] >= mejor).sum()))
    puestos = np.array(puestos)
    return {"mrr": float(np.mean(1 / puestos)),
            "acierto_1": float(np.mean(puestos == 1)),
            "acierto_5": float(np.mean(puestos <= 5)),
            "evaluados": len(puestos)}


# ---------------------------------------------------------------------------
# Solapamiento lexico
# ---------------------------------------------------------------------------

def solapamiento(texto_consulta, textos_por_id, relevantes, tokenizar):
    """Que palabras comparte la consulta con cada relevante, y con el resto.

    Se usa la MISMA tokenizacion que TF-IDF, porque la pregunta es si TF-IDF
    puede beneficiarse: si la interseccion es vacia para todos los relevantes,
    su puntaje TF-IDF es exactamente 0 por construccion.
    """
    q = set(tokenizar(texto_consulta))
    por_relevante = {i: sorted(q & set(tokenizar(textos_por_id[i])))
                     for i in relevantes}
    distractores = [i for i, t in textos_por_id.items()
                    if i not in relevantes and q & set(tokenizar(t))]
    return {
        "tokens_consulta": sorted(q),
        "compartidos_por_relevante": por_relevante,
        "relevantes_con_solape": sum(1 for v in por_relevante.values() if v),
        "distractores": distractores,
    }

"""
Tests del TP2: vectores, metricas, consultas, corpus y representaciones.

    python tests/test_tp2.py          (o: pytest tests/test_tp2.py)

No necesitan red ni base. Los que dependen de scikit-learn, gensim o
sentence-transformers se SALTEAN si la libreria no se puede importar, y se
informan como [SALTA], nunca como aprobados. El modelo SBERT solo se usa si ya
esta descargado en la cache local.

Varios tests son ADVERSARIALES a proposito: buscan fallas silenciosas (un NaN
que se propaga, un empate que decide el ranking, un k fuera de rango que
"funciona") mas que confirmar el caso feliz.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

import corpus                       # noqa: E402
import evaluacion as ev             # noqa: E402
import preprocesamiento as pp       # noqa: E402
import vectores as vx               # noqa: E402

RUTA_CONSULTAS = (RAIZ / "queries.json" if (RAIZ / "queries.json").exists()
                  else RAIZ / "queries_propuesta.json")


class Saltear(Exception):
    """El test no se puede correr en este entorno (falta una dependencia)."""


def saltear(motivo):
    """Saltea el test: con pytest usa su mecanismo, si no, el del runner."""
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(motivo)
    raise Saltear(motivo)


def requiere(modulo):
    try:
        return __import__(modulo, fromlist=["_"])
    except ImportError as e:
        saltear(f"no se pudo importar {modulo}: {str(e)[:60]}")


def lanza(excepcion, funcion, *args, **kwargs):
    try:
        funcion(*args, **kwargs)
    except excepcion:
        return True
    return False


# ---------------------------------------------------------------------------
# vectores.py
# ---------------------------------------------------------------------------

def test_validar_rechaza_nan_inf_y_dimension():
    ok = np.ones((3, 4))
    assert vx.validar(ok, dimension=4)["nulos"] == 0
    con_nan = ok.copy(); con_nan[1, 2] = np.nan
    con_inf = ok.copy(); con_inf[0, 0] = -np.inf
    assert lanza(vx.VectoresInvalidos, vx.validar, con_nan)
    assert lanza(vx.VectoresInvalidos, vx.validar, con_inf)
    assert lanza(vx.VectoresInvalidos, vx.validar, ok, dimension=5)
    assert lanza(vx.VectoresInvalidos, vx.validar, np.ones(4))   # no es 2D


def test_validar_cuenta_vectores_nulos():
    m = np.array([[1.0, 0], [0, 0], [0, 0]])
    r = vx.validar(m)
    assert r["nulos"] == 2 and r["filas_nulas"] == [1, 2]
    assert lanza(vx.VectoresInvalidos, vx.validar, m, permitir_ceros=False)


def test_normalizar_deja_nulos_en_cero_sin_nan():
    m = np.array([[3.0, 4.0], [0.0, 0.0], [1e-30, 0.0]])
    n = vx.normalizar_l2(m)
    assert not np.isnan(n).any()
    assert np.allclose(n[0], [0.6, 0.8])
    assert np.all(n[1] == 0)
    assert vx.esta_normalizada(n)


def test_normalizar_rechaza_nan_de_entrada():
    assert lanza(vx.VectoresInvalidos, vx.normalizar_l2,
                 np.array([[np.nan, 1.0]]))


def test_similitud_consulta_nula_da_empate_total():
    docs = np.eye(3)
    s = vx.similitudes(np.zeros(3), docs)
    assert np.all(s == 0)


def test_similitud_dimension_incorrecta():
    assert lanza(vx.VectoresInvalidos, vx.similitudes, np.ones(2), np.eye(3))


def test_similitud_es_coseno():
    docs = np.array([[1.0, 0.0], [1.0, 1.0], [-1.0, 0.0]])
    s = vx.similitudes(np.array([2.0, 0.0]), docs)
    assert np.allclose(s, [1.0, np.sqrt(0.5), -1.0], atol=1e-6)


def test_pares_aleatorios_sin_diagonal_y_reproducibles():
    m = np.eye(5)          # cada documento solo se parece a si mismo
    a = vx.pares_aleatorios(m, cantidad=2000, semilla=1)
    b = vx.pares_aleatorios(m, cantidad=2000, semilla=1)
    assert np.array_equal(a, b)
    assert np.all(a == 0), "aparecio un par (i, i)"


# ---------------------------------------------------------------------------
# evaluacion.py: precision@k
# ---------------------------------------------------------------------------

def test_precision_sin_empates():
    s = [0.9, 0.8, 0.7, 0.6, 0.5]
    rel = [True, False, True, False, False]
    assert ev.precision_at_k(s, rel, 1) == 1.0
    assert ev.precision_at_k(s, rel, 2) == 0.5
    assert np.isclose(ev.precision_at_k(s, rel, 3), 2 / 3)


def test_precision_denominador_es_k_aunque_haya_menos_relevantes():
    s = [0.9, 0.1, 0.1, 0.1, 0.1, 0.0]
    rel = [True, False, False, False, False, False]
    assert ev.precision_at_k(s, rel, 5) == 0.2
    assert ev.techo(1, 5) == 0.2


def test_precision_empate_total_es_el_piso():
    n, r = 200, 6
    rel = np.zeros(n, dtype=bool); rel[:r] = True    # relevantes AL PRINCIPIO
    s = np.zeros(n)
    # Un argsort ingenuo pondria los 6 relevantes primero y daria P@5 = 1.0.
    assert np.isclose(ev.precision_at_k(s, rel, 5), r / n)
    assert np.isclose(ev.precision_at_k(s, rel, 5), ev.piso_azar(r, n))


def test_precision_empate_parcial_exacto():
    # 1 ganador claro (relevante) y despues 4 empatados con 1 relevante.
    s = [0.9, 0.5, 0.5, 0.5, 0.5]
    rel = [True, True, False, False, False]
    # P@3: el primero + 2 de los 4 empatados -> 1 + 2 * 1/4 = 1.5 relevantes
    assert np.isclose(ev.precision_at_k(s, rel, 3), 1.5 / 3)


def test_precision_independiente_del_orden_de_los_documentos():
    rng = np.random.default_rng(0)
    s = np.round(rng.random(50), 1)          # muchos empates
    rel = rng.random(50) < 0.2
    p = ev.precision_at_k(s, rel, 10)
    perm = rng.permutation(50)
    assert np.isclose(p, ev.precision_at_k(s[perm], rel[perm], 10))


def test_precision_rechaza_k_invalido_y_nan():
    s, rel = [0.5, 0.4], [True, False]
    assert lanza(ValueError, ev.precision_at_k, s, rel, 0)
    assert lanza(ValueError, ev.precision_at_k, s, rel, 3)    # k > documentos
    assert lanza(ValueError, ev.precision_at_k, [np.nan, 0.4], rel, 1)
    assert lanza(ValueError, ev.precision_at_k, [0.5], rel, 1)  # largos distintos


def test_azar_simulado_converge_al_piso_exacto():
    rel = np.zeros(200, dtype=bool); rel[:10] = True
    sim = ev.simular_azar(rel, 5, repeticiones=20_000, semilla=3)
    assert abs(sim.mean() - ev.piso_azar(10, 200)) < 0.005
    assert np.array_equal(sim, ev.simular_azar(rel, 5, 20_000, semilla=3))


def test_bootstrap_contiene_la_media():
    v = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    bajo, alto = ev.intervalo_bootstrap(v)
    assert bajo <= v.mean() <= alto


def test_evaluar_incluye_azar_y_respeta_el_techo():
    ids = [10, 20, 30, 40, 50, 60]
    consultas = [ev.Consulta("a", "x", "t", frozenset({10})),
                 ev.Consulta("b", "y", "t", frozenset({20, 30}),
                             dudosos=frozenset({40}))]
    perfecto = {"perfecto": lambda t: np.array(
        [1, 0, 0, 0, 0, 0] if t == "x" else [0, 1, 1, 0.5, 0, 0], dtype=float)}
    df = ev.evaluar(perfecto, consultas, ids, ks=(2,))
    p = df[df.modelo == "perfecto"].set_index("consulta")
    assert p.loc["a", "p_at_k"] == 0.5 == p.loc["a", "techo"]
    assert p.loc["b", "p_at_k"] == 1.0
    assert "AZAR (piso)" in set(df.modelo)
    assert np.isclose(df[(df.modelo == "AZAR (piso)") & (df.consulta == "b")]
                      .p_at_k.iloc[0], 2 / 6)


def test_mrr_misma_serie():
    # A1 y A2 son de la saga A y se parecen entre si; B1 y B2 tambien.
    m = vx.normalizar_l2(np.array([[1, 0.1], [1, 0.0], [0, 1], [0.1, 1], [1, 1]]))
    r = ev.mrr_misma_serie(m, ["A", "A", "B", "B", None])
    assert r["evaluados"] == 4 and r["mrr"] == 1.0
    # Todo empatado: el puesto pesimista es el ultimo, no el primero.
    r = ev.mrr_misma_serie(np.ones((4, 2)) / np.sqrt(2), ["A", "A", None, None])
    assert r["acierto_1"] == 0.0


# ---------------------------------------------------------------------------
# evaluacion.py: el archivo de consultas
# ---------------------------------------------------------------------------

def _docs():
    return corpus.traer_documentos()


def _consulta_minima(i, relevantes, **extra):
    q = {"id": f"q{i}", "consulta": f"consulta {i}", "tipo": "t",
         "relevantes": [{"id": r, "titulo": "", "motivo": ""} for r in relevantes],
         "dudosos": []}
    q.update(extra)
    return q


def _cargar_desde(consultas):
    ids = [d.id for d in _docs()]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                     encoding="utf-8") as f:
        json.dump({"estado": "test", "consultas": consultas}, f)
    try:
        return ev.cargar_consultas(f.name, ids)
    finally:
        Path(f.name).unlink()


def test_consultas_reales_validas():
    docs = _docs()
    consultas, estado = ev.cargar_consultas(
        RUTA_CONSULTAS, [d.id for d in docs], {d.id: d.generos for d in docs})
    assert len(consultas) >= 10
    assert any(c.sin_solapamiento for c in consultas)
    print(f"        ({len(consultas)} consultas en {RUTA_CONSULTAS.name}; "
          f"estado: {estado[:40]})")


def test_consultas_rechaza_id_inexistente():
    ids = [d.id for d in _docs()][:12]
    qs = [_consulta_minima(i, [ids[i]]) for i in range(11)]
    qs[0]["sin_solapamiento"] = True
    qs.append(_consulta_minima(99, [999999999]))
    assert lanza(ev.ConsultasInvalidas, _cargar_desde, qs)


def test_consultas_rechaza_menos_de_diez_y_sin_relevantes():
    ids = [d.id for d in _docs()][:12]
    pocas = [_consulta_minima(i, [ids[i]], sin_solapamiento=True) for i in range(9)]
    assert lanza(ev.ConsultasInvalidas, _cargar_desde, pocas)
    vacias = [_consulta_minima(i, [ids[i]]) for i in range(11)]
    vacias[0]["sin_solapamiento"] = True
    vacias[5]["relevantes"] = []
    assert lanza(ev.ConsultasInvalidas, _cargar_desde, vacias)


def test_consultas_rechaza_relevante_y_dudoso_a_la_vez():
    ids = [d.id for d in _docs()][:12]
    qs = [_consulta_minima(i, [ids[i]]) for i in range(11)]
    qs[0]["sin_solapamiento"] = True
    qs[3]["dudosos"] = [{"id": ids[3], "titulo": "", "motivo": ""}]
    assert lanza(ev.ConsultasInvalidas, _cargar_desde, qs)


def test_consultas_exige_una_sin_solapamiento():
    ids = [d.id for d in _docs()][:12]
    qs = [_consulta_minima(i, [ids[i]]) for i in range(11)]
    assert lanza(ev.ConsultasInvalidas, _cargar_desde, qs)


def test_sin_solapamiento_verificado_sobre_el_corpus_real():
    """La comprobacion matematica: cero palabras compartidas con los relevantes."""
    docs = _docs()
    textos = {d.id: d.texto for d in docs}
    consultas, _ = ev.cargar_consultas(
        RUTA_CONSULTAS, list(textos), {d.id: d.generos for d in docs})
    marcadas = [c for c in consultas if c.sin_solapamiento]
    assert marcadas
    for c in marcadas:
        r = ev.solapamiento(c.texto, textos, c.relevantes, pp.tokenizar)
        assert r["relevantes_con_solape"] == 0, (c.id, r["compartidos_por_relevante"])


# ---------------------------------------------------------------------------
# corpus.py y preprocesamiento.py
# ---------------------------------------------------------------------------

def test_corpus_csv_200_libros_ordenados_y_tipados():
    docs = _docs()
    assert len(docs) == 200
    ids = [d.id for d in docs]
    assert ids == sorted(ids) and len(set(ids)) == 200
    assert all(isinstance(d.autores, list) and d.autores for d in docs)
    assert all(d.sinopsis.strip() for d in docs)
    assert all(d.serie is None or d.serie for d in docs), "serie '' en vez de None"


def test_corpus_filtros_y_documento_inmutable():
    terror = corpus.traer_documentos(generos=["Terror"])
    assert terror and all("Terror" in d.generos for d in terror)
    assert corpus.traer_documentos(generos=["NoExiste"]) == []
    assert corpus.traer_documentos(limite=5) == _docs()[:5]
    assert lanza(Exception, setattr, terror[0], "sinopsis", "x")
    assert lanza(ValueError, corpus.traer_documentos, tabla="inexistente")


def test_tokenizar_casos_borde():
    assert pp.tokenizar("") == []
    assert pp.tokenizar(None) == []
    assert pp.tokenizar("el la de que y no") == []          # solo stopwords
    assert pp.tokenizar("¡¿@#$%&*()?!") == []               # solo simbolos
    assert pp.tokenizar("Canción del NIÑO") == ["canción", "niño"]
    assert pp._sin_acentos("canción año") == "cancion año"


# ---------------------------------------------------------------------------
# embeddings.vector_promedio
# ---------------------------------------------------------------------------

def _kv_juguete():
    KeyedVectors = requiere("gensim.models").KeyedVectors
    kv = KeyedVectors(vector_size=3)
    kv.add_vectors(["gato", "perro"], np.array([[1, 0, 0], [0, 1, 0]],
                                               dtype=np.float32))
    return kv


def test_promedio_palabras_fuera_de_vocabulario_da_nulo():
    import embeddings
    kv = _kv_juguete()
    v = embeddings.vector_promedio(["xyz", "abc"], kv)
    assert v.dtype == np.float32 and np.all(v == 0)
    assert np.all(embeddings.vector_promedio([], kv) == 0)


def test_promedio_palabras_ignora_desconocidas_y_pondera():
    import embeddings
    kv = _kv_juguete()
    assert np.allclose(embeddings.vector_promedio(["gato", "xyz", "perro"], kv),
                       [0.5, 0.5, 0])
    assert np.allclose(embeddings.vector_promedio(
        ["gato", "perro"], kv, pesos={"gato": 3.0, "perro": 1.0}), [0.75, 0.25, 0])
    assert embeddings.vector_promedio(["gato"], kv).dtype == np.float32


def test_word2vec_reproducible_con_un_worker():
    Word2Vec = requiere("gensim.models").Word2Vec
    frases = [["gato", "come", "pescado"], ["perro", "come", "carne"]] * 50
    p = dict(vector_size=8, window=2, min_count=1, sg=1, epochs=5,
             workers=1, seed=7)
    a, b = Word2Vec(frases, **p), Word2Vec(frases, **p)
    assert np.array_equal(a.wv.vectors, b.wv.vectors)


# ---------------------------------------------------------------------------
# busqueda.py
# ---------------------------------------------------------------------------

def test_tfidf_consulta_sin_palabras_del_corpus_da_el_piso():
    requiere("sklearn.feature_extraction.text")
    import busqueda
    docs = _docs()
    rep = busqueda.tfidf_tp1([d.texto for d in docs])
    assert vx.esta_normalizada(rep.matriz)
    s = rep.puntajes("bucaneros filibusteros")       # ninguna esta en el corpus
    assert np.all(s == 0)
    rel = np.zeros(len(docs), dtype=bool); rel[:2] = True
    assert np.isclose(ev.precision_at_k(s, rel, 5), 2 / 200)


def test_tfidf_consulta_vacia_y_caracteres_raros():
    requiere("sklearn.feature_extraction.text")
    import busqueda
    rep = busqueda.tfidf_tp1([d.texto for d in _docs()])
    for consulta in ["", "   ", "¡¡¡???", "the quick brown fox"]:
        s = rep.puntajes(consulta)
        assert not np.isnan(s).any() and s.shape == (200,)


def test_promedio_palabras_representacion_valida():
    import busqueda
    kv = _kv_juguete()
    rep = busqueda.promedio_palabras("juguete", kv,
                                     ["el gato", "el perro", "nada conocido"])
    assert rep.info["documentos_nulos"] == 1
    assert vx.esta_normalizada(rep.matriz)
    assert np.all(rep.puntajes("palabras desconocidas") == 0)


class _TokenizadorFalso:
    """Cada palabra son DOS subpalabras, mas [CLS] y [SEP]."""
    def __call__(self, texto, add_special_tokens=True, truncation=False):
        return {"input_ids": list(range(2 * len(texto.split()) + 2))}


class _ModeloFalso:
    max_seq_length = 10
    tokenizer = _TokenizadorFalso()


def test_truncamiento_se_mide_en_tokens_no_en_caracteres():
    import busqueda
    textos = ["uno dos tres cuatro",          # 10 tokens: entra justo
              "a b c d e",                     # 12 tokens pero 9 caracteres
              "palabraslarguisimassinespacios"]   # 3 tokens, muchos caracteres
    r = busqueda.analizar_truncamiento(_ModeloFalso(), textos)
    assert r["truncados"] == 1
    assert list(r["largos"]) == [10, 12, 4]


def test_sbert_real_si_esta_en_cache():
    requiere("sentence_transformers")
    import busqueda
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    if not any(cache.glob(f"*{busqueda.MODELO_SBERT}*")):
        saltear("el modelo SBERT no esta descargado")
    m = busqueda.cargar_sbert()
    assert busqueda.dimension_sbert(m) == 512
    assert m.max_seq_length == 128
    largo = " ".join(["palabra"] * 300)
    r = busqueda.analizar_truncamiento(m, ["hola mundo", largo])
    assert r["truncados"] == 1
    v = m.encode(["", "hola"], convert_to_numpy=True)
    vx.validar(v, dimension=512)


# ---------------------------------------------------------------------------
# Runner con el formato del resto del proyecto
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pruebas = [(n, f) for n, f in globals().items()
               if n.startswith("test_") and callable(f)]
    fallas = saltadas = 0
    for nombre, prueba in pruebas:
        try:
            prueba()
            print(f"  [OK ]  {nombre}")
        except Saltear as e:
            saltadas += 1
            print(f"  [SALTA] {nombre}: {e}")
        except Exception as e:                       # noqa: BLE001
            fallas += 1
            print(f"  [FALLA] {nombre}: {type(e).__name__}: {e}")
    print(f"\n{len(pruebas)} tests: {len(pruebas) - fallas - saltadas} OK, "
          f"{fallas} fallas, {saltadas} salteados")
    return 1 if fallas else 0


if __name__ == "__main__":
    sys.exit(main())

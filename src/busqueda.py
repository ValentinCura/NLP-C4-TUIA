"""
Representaciones para la busqueda: TF-IDF, promedio de word vectors y SBERT.

TP2 - Procesamiento del Lenguaje Natural (TUIA, FCEIA-UNR).

Todas las representaciones exponen lo mismo: una matriz de documentos ya
normalizada (fila i = documento i del corpus, en el orden de corpus.py) y una
funcion que convierte una consulta en un vector del MISMO espacio. Asi la
evaluacion trata a todos los modelos con exactamente la misma operacion
(producto punto entre vectores de norma 1 = similitud coseno) y las
diferencias solo pueden venir de la representacion.

Cada modelo recibe el texto con SU preprocesamiento (el preprocesamiento
pertenece al modelo, no al corpus):
  - TF-IDF y los promedios de word vectors: texto tokenizado (minusculas, sin
    stopwords), igual que en el TP1. La consulta pasa por la misma funcion.
  - SBERT: texto CRUDO, documento y consulta. Fue entrenado sobre texto
    natural, con su propio tokenizador de subpalabras.
"""

import re
from dataclasses import dataclass, field

import numpy as np

import preprocesamiento
import vectores

MODELO_SBERT = "distiluse-base-multilingual-cased-v1"


@dataclass
class Representacion:
    nombre: str
    familia: str                 # "lexica", "promedio de palabras", "oracion"
    matriz: np.ndarray           # documentos x dimension, filas de norma 1 o 0
    codificar: callable          # texto de consulta -> vector (sin normalizar)
    info: dict = field(default_factory=dict)

    def puntajes(self, texto):
        """Similitud coseno de la consulta contra cada documento."""
        return vectores.similitudes(self.codificar(texto), self.matriz)

    def ranking(self, texto, k=5):
        """Los indices de los k documentos mas parecidos, de mayor a menor.

        Para MOSTRAR resultados. La evaluacion no usa este orden sino
        evaluacion.precision_at_k, que trata los empates como corresponde;
        aca, ante un empate, se desempata por posicion (orden estable).
        """
        s = self.puntajes(texto)
        return np.argsort(-s, kind="stable")[:k]


def _finalizar(nombre, familia, matriz, codificar, info=None):
    """Valida, normaliza y empaqueta. Ningun modelo se salta este paso."""
    matriz = np.asarray(matriz, dtype=np.float32)
    antes = vectores.validar(matriz, nombre=nombre)
    normalizada = vectores.normalizar_l2(matriz)
    info = dict(info or {})
    info.update({"dimension": matriz.shape[1],
                 "documentos_nulos": antes["nulos"],
                 "filas_nulas": antes["filas_nulas"]})
    return Representacion(nombre, familia, normalizada, codificar, info)


# ---------------------------------------------------------------------------
# Familia lexica
# ---------------------------------------------------------------------------

def tfidf_tp1(textos):
    """TF-IDF con el preprocesamiento del TP1: la linea de base de la consigna.

    Se ajusta (fit) sobre las 200 sinopsis; la consulta solo se transforma con
    el vocabulario y el IDF aprendidos. Una palabra de la consulta que no esta
    en el corpus no aporta nada, y una consulta sin ninguna palabra del corpus
    da el vector nulo: empate total.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer()
    matriz = vec.fit_transform(preprocesamiento.para_sklearn(textos)).toarray()

    def codificar(texto):
        return vec.transform(preprocesamiento.para_sklearn([texto])).toarray()[0]

    return _finalizar("TF-IDF (TP1)", "lexica", matriz, codificar,
                      {"vocabulario": len(vec.vocabulary_)})


def tfidf_caracteres(textos):
    """TF-IDF sobre n-gramas de caracteres (3 a 5), como CONTROL.

    No es la linea de base que pide la consigna: es una pregunta de control.
    TF-IDF del TP1 no lematiza, asi que 'vampiros' y 'vampiro' son palabras
    distintas. Con n-gramas de caracteres comparten 'vamp', 'ampi', 'mpir'...
    Si esta variante recupera lo que el TF-IDF del TP1 pierde, el problema era
    la morfologia, no la falta de semantica. Sigue siendo puramente lexica:
    'bucanero' y 'pirata' no comparten ningun n-grama util.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                          lowercase=True, min_df=2, sublinear_tf=True)
    matriz = vec.fit_transform(textos).toarray()

    def codificar(texto):
        return vec.transform([texto]).toarray()[0]

    return _finalizar("TF-IDF n-gramas de caracteres", "lexica", matriz,
                      codificar, {"vocabulario": len(vec.vocabulary_)})


# ---------------------------------------------------------------------------
# Familia: promedio de word vectors
# ---------------------------------------------------------------------------

def promedio_palabras(nombre, kv, textos):
    """Vector de documento = promedio de los vectores de sus palabras.

    kv son los KeyedVectors del modelo (Word2Vec propio, FastText propio o
    SBW). Se usa embeddings.vector_promedio, el mismo de la Unidad 2: las
    palabras fuera del vocabulario se ignoran, y si no queda ninguna el vector
    es nulo (ver su docstring).

    Se registra, por documento, cuantas palabras se usaron: un documento que
    se representa con 3 palabras de 80 no es comparable a uno con 70.

    Ojo con FastText: 'palabra in kv' es True para CUALQUIER palabra, porque
    compone el vector con n-gramas. Para el, nunca hay palabras fuera de
    vocabulario, y tampoco vectores nulos.
    """
    import embeddings

    tokens = [preprocesamiento.tokenizar(t) for t in textos]
    matriz = np.array([embeddings.vector_promedio(t, kv) for t in tokens])
    usadas = [sum(1 for p in t if p in kv) for t in tokens]
    cobertura = [u / len(t) if t else 0.0 for u, t in zip(usadas, tokens)]

    def codificar(texto):
        return embeddings.vector_promedio(preprocesamiento.tokenizar(texto), kv)

    return _finalizar(nombre, "promedio de palabras", matriz, codificar,
                      {"cobertura_media": float(np.mean(cobertura)),
                       "cobertura_minima": float(np.min(cobertura))})


def cobertura_consulta(texto, kv):
    """Que palabras de la consulta conoce el modelo y cuales no."""
    tokens = preprocesamiento.tokenizar(texto)
    return {"conocidas": [t for t in tokens if t in kv],
            "desconocidas": [t for t in tokens if t not in kv]}


# ---------------------------------------------------------------------------
# Familia: modelo de oracion
# ---------------------------------------------------------------------------

def cargar_sbert(nombre=MODELO_SBERT):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(nombre, device="cpu")


def dimension_sbert(modelo):
    """Dimension del embedding. sentence-transformers 6 renombro el metodo, y
    Colab puede traer otra version: se usa el que exista."""
    metodo = (getattr(modelo, "get_embedding_dimension", None)
              or modelo.get_sentence_embedding_dimension)
    return metodo()


def contar_tokens(modelo, textos):
    """Tokens de subpalabra de cada texto, CON los especiales ([CLS], [SEP]).

    El limite del modelo (max_seq_length) se cuenta en estas unidades, no en
    palabras ni en caracteres. Una palabra castellana poco frecuente se parte
    en varias subpalabras, asi que contar palabras SUBESTIMA el largo real.
    Se tokeniza sin truncar para ver el largo completo.
    """
    return [len(modelo.tokenizer(t, add_special_tokens=True,
                                 truncation=False)["input_ids"])
            for t in textos]


def analizar_truncamiento(modelo, textos):
    """Cuantos documentos superan el limite del modelo, y cuanto pierden.

    El modelo no avisa: encode() corta en silencio todo lo que pase de
    max_seq_length tokens, y el embedding resultante representa solo el
    principio del texto.
    """
    limite = modelo.max_seq_length
    largos = np.array(contar_tokens(modelo, textos))
    truncados = largos > limite
    perdidos = np.clip(largos - limite, 0, None)
    return {
        "limite_tokens": limite,
        "documentos": len(textos),
        "truncados": int(truncados.sum()),
        "fraccion_truncados": float(truncados.mean()),
        "tokens_mediana": float(np.median(largos)),
        "tokens_max": int(largos.max()),
        "fraccion_texto_perdida_media": float(
            (perdidos / largos)[truncados].mean()) if truncados.any() else 0.0,
        "largos": largos,
    }


def sbert(textos, modelo=None):
    """Embeddings de oracion del texto CRUDO (lo que el modelo ve, truncado)."""
    modelo = modelo or cargar_sbert()
    matriz = modelo.encode(textos, batch_size=32, show_progress_bar=False,
                           convert_to_numpy=True)

    def codificar(texto):
        return modelo.encode([texto], convert_to_numpy=True)[0]

    return _finalizar("SBERT (truncado)", "oracion", matriz, codificar,
                      {"limite_tokens": modelo.max_seq_length,
                       "modelo": MODELO_SBERT})


def _fragmentar(modelo, texto, limite):
    """Parte un texto en fragmentos de oraciones completas que entran enteros."""
    oraciones = [o for o in re.split(r"(?<=[.!?…»])\s+", texto) if o.strip()]
    fragmentos, actual = [], ""
    for o in oraciones:
        candidato = f"{actual} {o}".strip()
        if actual and contar_tokens(modelo, [candidato])[0] > limite:
            fragmentos.append(actual)
            actual = o
        else:
            actual = candidato
    if actual:
        fragmentos.append(actual)
    return fragmentos


def sbert_fragmentos(textos, modelo=None):
    """SBERT sin perder texto: promedio de los embeddings de cada fragmento.

    Responde la pregunta que deja el truncamiento: ¿importa lo que se corta?
    Si esta variante y la truncada rinden igual, el principio de la sinopsis
    alcanzaba. El fragmentado es por oraciones completas, para no partir una
    oracion al medio. Una oracion sola mas larga que el limite se trunca igual.
    """
    modelo = modelo or cargar_sbert()
    limite = modelo.max_seq_length
    filas, n_fragmentos = [], []
    for t in textos:
        partes = _fragmentar(modelo, t, limite)
        emb = modelo.encode(partes, convert_to_numpy=True)
        filas.append(vectores.normalizar_l2(emb).mean(axis=0))
        n_fragmentos.append(len(partes))

    def codificar(texto):
        return modelo.encode([texto], convert_to_numpy=True)[0]

    return _finalizar("SBERT (por fragmentos)", "oracion", np.array(filas),
                      codificar, {"fragmentos_media": float(np.mean(n_fragmentos)),
                                  "fragmentos_max": int(np.max(n_fragmentos))})


# ---------------------------------------------------------------------------
# Mostrar resultados
# ---------------------------------------------------------------------------

def lado_a_lado(representaciones, texto, titulos, relevantes_idx=(), k=5):
    """Top-k de cada representacion para una consulta, en columnas paralelas.

    Marca con un tilde los documentos que el conjunto de evaluacion considera
    relevantes, para ver de un vistazo quien acierta.
    """
    import pandas as pd

    relevantes_idx = set(relevantes_idx)
    columnas = {}
    for rep in representaciones:
        s = rep.puntajes(texto)
        orden = rep.ranking(texto, k)
        columnas[rep.nombre] = [
            f"{'✓ ' if i in relevantes_idx else ''}{titulos[i][:38]} ({s[i]:.2f})"
            for i in orden]
    return pd.DataFrame(columnas, index=range(1, k + 1))

"""
Operaciones sobre matrices de vectores: validar, normalizar, comparar.

Unidad 2 - Procesamiento del Lenguaje Natural (TUIA, FCEIA-UNR).

Todas las representaciones del TP2 (TF-IDF, promedio de word vectors, modelo de
oracion) terminan en una matriz documentos x dimensiones. Este modulo es el
unico lugar donde esas matrices se validan y se normalizan, para que la regla
sea la misma para todos los modelos y para que haya UN test que la cubra.

Solo depende de numpy: se puede probar sin modelos, sin red y sin base.
"""

import numpy as np


class VectoresInvalidos(ValueError):
    """Una matriz de embeddings no cumple lo minimo para usarse."""


def validar(matriz, dimension=None, nombre="matriz", permitir_ceros=True):
    """Revisa una matriz de vectores y devuelve un resumen de lo que encontro.

    Falla RUIDOSAMENTE (VectoresInvalidos) ante lo que nunca deberia pasar:
      - una dimension distinta de la esperada;
      - NaN o infinitos, que contaminan cualquier similitud que los toque.

    Los vectores de ceros NO son un error en si: tienen un significado ("de
    este texto no se sabe nada", ver embeddings.vector_promedio). Pero se
    cuentan y se devuelven, porque un modelo con muchos vectores nulos esta
    funcionando mal aunque no explote. Con permitir_ceros=False tambien son
    error.
    """
    m = np.asarray(matriz)
    if m.ndim != 2:
        raise VectoresInvalidos(f"{nombre}: se esperaba una matriz 2D y "
                                f"llego una de {m.ndim} dimensiones")
    if dimension is not None and m.shape[1] != dimension:
        raise VectoresInvalidos(f"{nombre}: dimension {m.shape[1]}, "
                                f"se esperaba {dimension}")

    filas_nan = np.where(np.isnan(m).any(axis=1))[0]
    filas_inf = np.where(np.isinf(m).any(axis=1))[0]
    if len(filas_nan) or len(filas_inf):
        raise VectoresInvalidos(
            f"{nombre}: {len(filas_nan)} filas con NaN "
            f"{filas_nan[:5].tolist()} y {len(filas_inf)} con infinitos "
            f"{filas_inf[:5].tolist()}")

    normas = np.linalg.norm(m, axis=1)
    filas_cero = np.where(normas == 0)[0]
    if len(filas_cero) and not permitir_ceros:
        raise VectoresInvalidos(f"{nombre}: {len(filas_cero)} vectores nulos "
                                f"en las filas {filas_cero[:5].tolist()}")

    return {
        "nombre": nombre,
        "filas": m.shape[0],
        "dimension": m.shape[1],
        "nulos": len(filas_cero),
        "filas_nulas": filas_cero.tolist(),
        "norma_min": float(normas.min()) if len(normas) else 0.0,
        "norma_max": float(normas.max()) if len(normas) else 0.0,
    }


def normalizar_l2(matriz):
    """Divide cada fila por su norma, dejando en cero las filas nulas.

    Con todas las filas de norma 1, el producto punto ES la similitud coseno, y
    los rankings de todos los modelos se calculan con la misma operacion.

    El caso de la fila nula es el que importa: dividir por cero daria NaN, y
    un NaN se propaga a todas las similitudes que lo tocan. Se deja la fila en
    cero, que da similitud 0 con todo, coherente con su significado.

    Valida antes (para no normalizar NaN) y despues (para garantizar el
    resultado).
    """
    m = np.asarray(matriz, dtype=np.float32)
    if m.ndim == 1:
        return normalizar_l2(m[None, :])[0]
    validar(m, nombre="antes de normalizar")
    normas = np.linalg.norm(m, axis=1, keepdims=True)
    resultado = np.divide(m, normas, out=np.zeros_like(m), where=normas > 0)
    validar(resultado, nombre="despues de normalizar")
    return resultado


def esta_normalizada(matriz, tolerancia=1e-4):
    """True si cada fila tiene norma 1, o 0 si es nula."""
    normas = np.linalg.norm(np.asarray(matriz, dtype=np.float64), axis=1)
    return bool(np.all((np.abs(normas - 1) < tolerancia) | (normas == 0)))


def similitudes(consulta, documentos):
    """Similitud coseno de un vector contra cada fila de una matriz.

    Normaliza los dos lados, asi que funciona aunque no vengan normalizados.
    Si la consulta es nula, devuelve todos ceros (empate total), que la
    evaluacion trata como un ranking sin informacion.
    """
    q = normalizar_l2(np.asarray(consulta, dtype=np.float32).reshape(1, -1))[0]
    d = normalizar_l2(documentos)
    if q.shape[0] != d.shape[1]:
        raise VectoresInvalidos(f"la consulta tiene dimension {q.shape[0]} y "
                                f"los documentos {d.shape[1]}")
    return d @ q


def pares_aleatorios(matriz, cantidad=5000, semilla=42):
    """Similitud coseno entre pares aleatorios de documentos distintos.

    Sirve para ver si el espacio discrimina: si todo se parece a todo (media
    alta, desvio chico), el ranking ordena diferencias minimas aunque devuelva
    resultados. Los pares se sortean con reposicion entre pares, pero nunca un
    documento consigo mismo.
    """
    d = normalizar_l2(matriz)
    n = d.shape[0]
    rng = np.random.default_rng(semilla)
    i = rng.integers(0, n, size=cantidad)
    j = rng.integers(0, n - 1, size=cantidad)
    j = np.where(j >= i, j + 1, j)      # salta la diagonal sin sesgar
    return np.einsum("ij,ij->i", d[i], d[j])

# Traspaso del proyecto a otra máquina

> Esta rama (`borrar`) es donde se trabaja: tiene la Unidad 2 y el TP2 completos. `main`
> quedó en el final de la Unidad 1 (`141a4ca`); antes de entregar hay que decidir si se
> mergea. Incluye los buffers `.jsonl`, que en `main` están ignorados, para no tener que
> re-scrapear nada.
>
> El `.env` **ya no** se versiona: se sacó del índice para que nunca se publique por
> accidente una credencial real. Hay que crearlo a partir de `.env.example` (paso 3).
>
> **Para el TP2 no hace falta la base de datos.** La cátedra habilitó trabajar desde el
> CSV, y el notebook corre en Colab. Los pasos 3 y 4 solo sirven si se quiere usar
> PostgreSQL.

## Qué NO viene en el repositorio, y por qué

Tres cosas se quedaron afuera. Las tres se regeneran, así que no se pierde nada.

| Qué falta | Tamaño | Por qué no está | Cómo se recupera |
|---|---|---|---|
| `venv/` | 797 MB | Tiene rutas absolutas de Windows adentro: en otra máquina no funcionaría aunque se copiara | `pip install -r requirements.txt` |
| `data/modelos/` | 777 MB | Uno de los archivos pesa 762 MB y GitHub **rechaza** todo lo que supere 100 MB | `python src/embeddings.py entrenar` (segundos) |
| `SBW-vectors-300-min5.bin.gz` | 1,07 GB | Mismo límite, y además vive fuera del repositorio | Descarga, ver paso 5 |

## Pasos

### 1. Clonar la rama

```bash
git clone -b borrar https://github.com/ValentinCura/NLP-C4-TUIA.git
```

### 2. Crear el entorno de Python

Requiere **Python 3.10 o superior**.

```bash
python -m venv venv
```

```bash
venv/Scripts/activate
```

```bash
pip install -r requirements.txt
```

```bash
python -m playwright install chromium
```

> En Linux o macOS el activador es `source venv/bin/activate`. El `playwright install`
> solo hace falta si se va a volver a scrapear.

### 3. Levantar la base de datos

Requiere **Docker Desktop abierto** (el ícono de la ballena en verde).

```bash
docker compose up -d
```

```bash
docker compose ps
```

Tiene que decir `healthy` antes de seguir.

> El `.env` no viene en el repositorio. Crearlo con `copy .env.example .env` (en Linux o
> macOS, `cp .env.example .env`).

### 4. Cargar el corpus

```bash
python src/etl.py
```

```bash
python src/etl.py --csv data/libros_ampliado.csv --tabla libros_ampliado
```

Ambos tienen que terminar con `CARGA OK`.

### 5. El modelo pre-entrenado (solo para la comparación de embeddings)

Descargar `SBW-vectors-300-min5.bin.gz` (1,07 GB) de
[crscardellino.ar/SBWCE](https://crscardellino.ar/SBWCE/) y dejarlo **en la carpeta que
contiene al repositorio**, no adentro:

```
una-carpeta-cualquiera/
├── SBW-vectors-300-min5.bin.gz     <- acá
└── NLP-C4-TUIA/                    <- el repositorio clonado
```

Sin este archivo todo lo demás funciona igual: los módulos avisan que falta y usan solo
los modelos propios.

### 6. Entrenar los embeddings propios

```bash
python src/embeddings.py entrenar
```

## Verificar que quedó todo bien

Sin base de datos (alcanza para el TP2):

```bash
python tests/test_tp2.py
```

```bash
python tests/test_parsers.py
```

```bash
python src/corpus.py
```

```bash
python src/preprocesamiento.py --demo
```

Con la base levantada (solo Unidad 2): `python src/db.py` y
`python src/etl.py --solo-verificar`. El resto de los comandos está documentado en el
[README](README.md#ejecución).

## Continuar el TP2: qué falta y cómo hacerlo

**Estado al 2026-10-07.** El pipeline del TP2 está completo y probado: 35 tests, y el
notebook ejecutado de punta a punta en un kernel limpio. **Pero la evaluación es
PROVISIONAL**: las consultas de `queries_propuesta.json` las propuso un asistente de IA y
el grupo todavía no las validó. Ningún número de la sección 6 del notebook ni del borrador
del informe es definitivo. El detalle completo de lo que se auditó, corrigió e implementó
está en [`docs/AUDITORIA_TP2.md`](docs/AUDITORIA_TP2.md).

Los pasos que faltan, en orden:

1. **Validar las consultas, sin mirar los resultados del notebook.** Abrir
   [`docs/propuesta_queries.md`](docs/propuesta_queries.md):
   - tildar los relevantes con los que se está de acuerdo;
   - decidir cada dudoso (pasa a relevante o se descarta);
   - agregar libros que falten (el id sale de `data/libros.csv`);
   - revisar que las consultas marcadas "sin solapamiento" sigan sin compartir palabras con
     sus relevantes.

   Si se agrega un relevante a esas consultas, el test
   `test_sin_solapamiento_verificado_sobre_el_corpus_real` lo detecta.

2. **Generar `queries.json`.** Copiar `queries_propuesta.json` a `queries.json` (en la raíz)
   y aplicarle las correcciones. El formato es el mismo:
   - `relevantes` y `dudosos` son listas de `{"id", "titulo", "motivo"}`;
   - los dudosos que queden se tratan como **no** relevantes, y se informan aparte.

   Cambiar el campo `estado` a algo como `"VALIDADO por el grupo el <fecha>"`, sin la
   palabra PROVISIONAL. El notebook y los tests usan `queries.json` automáticamente si
   existe. Después correr `python tests/test_tp2.py`, que valida ids, formato, duplicados,
   mínimo de 10 consultas y solapamiento cero.

   **Las consultas y la relevancia no se tocan después de ver resultados.**

3. **Ejecutar el notebook en Colab** con "Ejecutar todo", desde un entorno nuevo. La
   primera celda clona esta rama (`borrar`) e imprime el commit, que tiene que coincidir con
   el de GitHub. Al final, la celda **"Verificación de la ejecución"** comprueba 15 etapas.
   Con las consultas validadas tienen que pasar las 15; con las provisorias falla solo la
   14, a propósito. Descargar el `.ipynb` ejecutado y subirlo al repositorio.

4. **Escribir las conclusiones.** Va en la sección 7 del notebook y en el texto del caso de
   fallo. Después cerrar el informe: partir de
   [`docs/informe_borrador.md`](docs/informe_borrador.md), reemplazar todos los números por
   los definitivos (están marcados con **[R]**) y generar `informe.pdf`, de 3 páginas como
   máximo. El párrafo de uso de IA lo completa el grupo.

5. **Auditoría final** contra la consigna: la checklist está en `docs/AUDITORIA_TP2.md`.

### Regenerar o ejecutar el notebook fuera de Colab

```bash
python tools/generar_notebook.py TP2_Beltramo_Cortinas_Cura_Maragliano.ipynb
```

```bash
python tools/ejecutar_notebook.py TP2_Beltramo_Cortinas_Cura_Maragliano.ipynb
```

El primero lo reescribe **desde cero** a partir de `tools/generar_notebook.py`, y pisa
cualquier edición hecha a mano en el `.ipynb`. Si se escriben las conclusiones directamente
en el notebook, ese archivo pasa a ser la fuente de verdad y no hay que volver a
regenerarlo. El segundo equivale a "Reiniciar y ejecutar todo". Necesita
`pip install nbclient ipykernel`.

### Si la PC tiene Windows Smart App Control

Algunas PCs con Windows 11 traen activado *Smart App Control*, que bloquea DLLs sin firma de
scikit-learn (`sklearn.metrics`), spaCy y sentence-transformers. Los tests lo informan como
`[SALTA]`, nunca como aprobado. Desactivarlo es irreversible sin reinstalar Windows, así que
se trabajó en un contenedor Linux con Docker Desktop. Desde la raíz del repositorio, con el
SBW en la carpeta de arriba:

```bash
docker run --rm -it -v "${PWD}:/work" -v "${PWD}/..:/parent:ro" -e SBW_PATH=/parent/SBW-vectors-300-min5.bin.gz -w /work python:3.12-slim bash
```

Adentro, primero el torch de CPU (si no, pip baja la versión con CUDA, de varios GB):
`pip install torch --index-url https://download.pytorch.org/whl/cpu`. Después
`pip install -r requirements.txt nbclient ipykernel`, y los comandos de arriba. Python 3.12
es la misma versión que usa Colab.

## Dos cosas que suelen trabar

**Todo funciona pero lentísimo.** Revisar que el `.env` diga `PGHOST=127.0.0.1` y no
`localhost`: en Windows, `localhost` resuelve primero a IPv6, donde la base no escucha, y
cada conexión espera el *timeout* de TCP. Son 130 segundos contra 0,02.

**`No se pudo conectar a PostgreSQL`.** Casi siempre es que Docker Desktop está cerrado.
El propio mensaje de error lista qué revisar, en orden.

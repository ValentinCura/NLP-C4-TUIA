# Traspaso del proyecto a otra máquina

> Esta rama (`borrar`) existe para mover el proyecto entre computadoras. Incluye
> algunos archivos que en `main` están ignorados a propósito (el `.env` y los buffers
> `.jsonl`), para que no haya que reconfigurar ni re-scrapear nada.
>
> **No es la rama de entrega.** La rama de trabajo es `main`.

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

> El `.env` ya viene en esta rama, así que no hay que crearlo. Si algún día falta:
> `copy .env.example .env`

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

Estos comandos no modifican nada y tienen que pasar los cinco:

```bash
python tests/test_parsers.py
```

```bash
python src/db.py
```

```bash
python src/etl.py --solo-verificar
```

```bash
python src/corpus.py
```

```bash
python src/preprocesamiento.py --demo
```

Si los cinco andan, el proyecto está listo. El resto de los comandos está documentado en
el [README](README.md#ejecución).

## Dos cosas que suelen trabar

**Todo funciona pero lentísimo.** Revisar que el `.env` diga `PGHOST=127.0.0.1` y no
`localhost`: en Windows, `localhost` resuelve primero a IPv6, donde la base no escucha, y
cada conexión espera el *timeout* de TCP. Son 130 segundos contra 0,02.

**`No se pudo conectar a PostgreSQL`.** Casi siempre es que Docker Desktop está cerrado.
El propio mensaje de error lista qué revisar, en orden.

-- Esquema del corpus de libros.
--
-- Unidad 2 - Procesamiento del Lenguaje Natural (TUIA, FCEIA-UNR).
--
-- Lo aplica src/etl.py, no el directorio /docker-entrypoint-initdb.d/ del
-- contenedor: ese directorio solo se ejecuta cuando el volumen esta vacio, asi
-- que al editar el DDL y reiniciar no pasaria nada y se perderia media tarde
-- entendiendo por que. Un solo camino, y ademas funciona igual contra un
-- PostgreSQL nativo.


-- ---------------------------------------------------------------------------
-- Tabla principal
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS libros (
    -- El id propio de Lectulandia (postid-N del <body>). Es unico y estable, y
    -- permite volver a la ficha de origen: una PK sintetica no agregaria nada y
    -- romperia esa trazabilidad.
    id                integer     PRIMARY KEY,

    titulo            text        NOT NULL,
    titulo_serie      text        NOT NULL,

    -- Los campos multivaluados viajan como jsonb, que es exactamente para lo
    -- que el scraper los escribio como array JSON en el CSV. Llegan a Python
    -- como list[str] sin que nadie haga un json.loads a mano.
    autores           jsonb       NOT NULL,
    n_autores         smallint    NOT NULL,
    generos           jsonb       NOT NULL,

    -- Las tres columnas que admiten NULL son aquellas donde la ausencia es
    -- informacion real: el libro no pertenece a ninguna serie, o no tiene
    -- portada. Ver la nota sobre '' vs NULL mas abajo.
    serie             text,
    serie_num         smallint,

    sinopsis          text        NOT NULL,
    url_libro         text        NOT NULL UNIQUE,
    portada           text,
    categoria_origen  jsonb       NOT NULL,
    fecha_extraccion  date        NOT NULL,

    -- ----- Invariantes que la base hace cumplir -----

    CONSTRAINT titulo_no_vacio        CHECK (btrim(titulo) <> ''),
    CONSTRAINT titulo_serie_no_vacio  CHECK (btrim(titulo_serie) <> ''),
    CONSTRAINT sinopsis_no_vacia      CHECK (btrim(sinopsis) <> ''),

    CONSTRAINT autores_es_array   CHECK (jsonb_typeof(autores) = 'array'),
    CONSTRAINT generos_es_array   CHECK (jsonb_typeof(generos) = 'array'),
    CONSTRAINT categoria_es_array CHECK (jsonb_typeof(categoria_origen) = 'array'),

    -- categoria_origen nunca puede ser [] : todo libro llego por alguna
    -- categoria semilla. generos SI puede ser [] (el sitio podria no declarar
    -- ninguno), asi que ahi no se exige longitud minima.
    CONSTRAINT categoria_no_vacia CHECK (jsonb_array_length(categoria_origen) >= 1),

    -- n_autores es un dato derivado que ya viene calculado en el CSV. En vez de
    -- recalcularlo con una columna GENERATED, se VERIFICA: si alguna vez no
    -- coincidiera, queremos enterarnos de que el dato de origen estaba mal, no
    -- taparlo silenciosamente.
    CONSTRAINT n_autores_coherente CHECK (n_autores = jsonb_array_length(autores)),

    -- Prohibir la cadena vacia en las columnas nullables es la pieza que hace
    -- que exista UNA SOLA forma de representar la ausencia. Sin esto, la
    -- politica de NULL seria un parrafo del informe; con esto, es un mecanismo.
    CONSTRAINT serie_no_vacia   CHECK (serie <> ''),
    CONSTRAINT portada_no_vacia CHECK (portada <> ''),

    -- Ojo: va >= 0 y no > 0. Hay 4 libros con serie_num = 0, que son omnibus y
    -- precuelas ("Dune: La saga completa", "El Archivo de las Tormentas 1-5",
    -- "Amanecer en la cosecha", "Las flores en llamas"). Un CHECK (> 0), que es
    -- lo que uno escribiria por reflejo, rompe la carga.
    CONSTRAINT serie_num_valido CHECK (serie_num >= 0),

    -- Un numero de tomo sin serie no significa nada.
    CONSTRAINT serie_num_implica_serie
        CHECK (serie_num IS NULL OR serie IS NOT NULL),

    -- El CSV guarda solo el slug; el prefijo vive en una constante de Python.
    CONSTRAINT url_es_slug CHECK (url_libro NOT LIKE 'http%')
);


-- ---------------------------------------------------------------------------
-- Indices
-- ---------------------------------------------------------------------------
--
-- Nota honesta: con 200 filas el planificador va a elegir Seq Scan siempre y
-- ninguno de estos indices se va a usar. Estan porque el diseno tiene que
-- escalar y porque son la forma correcta de consultar jsonb. En
-- sql/02_consultas.sql hay un EXPLAIN que lo demuestra.
--
-- Se usa jsonb_ops (el default) y NO jsonb_path_ops: este ultimo es mas chico y
-- mas rapido, pero solo soporta el operador @>. Con jsonb_path_ops, un
-- "generos ?| array['Terror','Novela']" (filtrar por CUALQUIERA de varios
-- generos, que es justo lo que necesita la capa de corpus) no podria usar el
-- indice. A 200 filas la diferencia de tamano es irrelevante; la de
-- flexibilidad no.

CREATE INDEX IF NOT EXISTS libros_generos_gin
    ON libros USING gin (generos);

CREATE INDEX IF NOT EXISTS libros_categoria_gin
    ON libros USING gin (categoria_origen);

-- Indice parcial: no tiene sentido indexar las 94 filas con serie NULL, que
-- nadie busca.
CREATE INDEX IF NOT EXISTS libros_serie_idx
    ON libros (serie) WHERE serie IS NOT NULL;


-- ---------------------------------------------------------------------------
-- Vista de conveniencia
-- ---------------------------------------------------------------------------
--
-- La forma relacional de "un libro tiene muchos generos", para quien prefiera
-- consultar con GROUP BY en vez de con operadores de jsonb. Es una vista y no
-- una tabla: no duplica almacenamiento y no puede quedar desincronizada.

CREATE OR REPLACE VIEW libro_genero AS
    SELECT l.id, l.titulo, g.valor AS genero
    FROM libros l, jsonb_array_elements_text(l.generos) AS g(valor);


-- ---------------------------------------------------------------------------
-- Documentacion dentro de la propia base
-- ---------------------------------------------------------------------------
--
-- Aparece al hacer \d+ libros en psql y en el panel de DBeaver. La decision mas
-- importante del trabajo queda escrita donde la va a leer quien este tentado de
-- hacer el UPDATE.

COMMENT ON TABLE libros IS
    'Corpus de sinopsis de libros extraido de Lectulandia (Unidad 1). '
    'Se carga desde data/libros.csv con src/etl.py.';

COMMENT ON COLUMN libros.sinopsis IS
    'Texto CRUDO. No aplicar lower(), stemming, quita de acentos ni de '
    'stopwords sobre esta columna: el preprocesamiento pertenece al MODELO, no '
    'al corpus, y cada modelo necesita uno distinto. Ver src/preprocesamiento.py';

COMMENT ON COLUMN libros.serie IS
    'NULL (no cadena vacia) cuando el libro es autoconclusivo. El CSV usa '''' '
    'porque el formato CSV no tiene NULL; el ETL traduce entre ambos idiomas.';

COMMENT ON COLUMN libros.serie_num IS
    'Numero de tomo. Puede ser 0: hay omnibus y precuelas.';

COMMENT ON COLUMN libros.generos IS
    'Generos tal como los publica el sitio, SIN canonicalizar (incluye el typo '
    'del propio sitio "Publicaciones periodicas"). Conservar el valor crudo es '
    'lo que permite auditar contra el origen.';

COMMENT ON COLUMN libros.categoria_origen IS
    'Categorias semilla desde las que el scraper llego al libro. No sale de la '
    'ficha: depende de que categorias y que cupo se usaron al extraer.';

COMMENT ON COLUMN libros.url_libro IS
    'Solo el slug. La URL completa se reconstruye con el prefijo constante que '
    'vive en src/scraper.py (PREFIJO_FICHA).';

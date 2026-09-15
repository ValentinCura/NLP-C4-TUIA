-- Consultas de verificacion y exploracion del corpus.
--
-- Unidad 2 - Procesamiento del Lenguaje Natural (TUIA, FCEIA-UNR).
--
-- Pensadas para ejecutarse a mano desde DBeaver, bloque por bloque. Las
-- comprobaciones automaticas equivalentes estan en la funcion verificar() de
-- src/etl.py; estas sirven para mirar los datos con los ojos.
--
-- Conexion desde DBeaver:  host 127.0.0.1 (NO "localhost"), puerto 5433,
-- base tuia, usuario tuia, password tuia.


-- ===========================================================================
-- 1. Que hay cargado
-- ===========================================================================

SELECT count(*) AS libros,
       count(DISTINCT id) AS ids_unicos,
       count(serie) AS con_serie,               -- count() ignora los NULL
       count(*) - count(serie) AS autoconclusivos,
       min(fecha_extraccion) AS extraido_desde,
       max(fecha_extraccion) AS extraido_hasta
FROM libros;


-- Longitudes de las sinopsis: es lo que determina si hay senal suficiente para
-- TF-IDF. La mediana importa mas que el promedio, porque un par de sinopsis muy
-- largas lo estirarian.
SELECT count(*) AS libros,
       min(length(sinopsis))    AS mas_corta,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY length(sinopsis))::int
                                AS mediana,
       max(length(sinopsis))    AS mas_larga,
       sum(length(sinopsis))    AS checksum
FROM libros;


-- ===========================================================================
-- 2. La politica de valores ausentes
-- ===========================================================================
--
-- La ausencia se representa SIEMPRE con NULL, nunca con cadena vacia. Estas dos
-- consultas tienen que devolver 0 y 0: si alguna diera distinto, significaria
-- que sobrevivio una '' del CSV y que existirian dos formas de decir "no hay
-- dato", que es justo lo que el CHECK de la tabla impide.

SELECT count(*) AS cadenas_vacias_que_no_deberian_existir
FROM libros
WHERE '' IN (titulo, titulo_serie, sinopsis, url_libro)
   OR serie = '' OR portada = '';

-- serie y serie_num van siempre juntas: un numero de tomo sin serie no
-- significaria nada.
SELECT count(*) AS incoherencias
FROM libros
WHERE (serie IS NULL) <> (serie_num IS NULL);


-- ===========================================================================
-- 3. Consultar los campos multivaluados (jsonb)
-- ===========================================================================

-- Distribucion de generos. Notar el desbalance: "Novela" aparece en el 85% de
-- los libros, o sea que como etiqueta de clasificacion casi no discrimina.
SELECT genero, count(*) AS libros
FROM libro_genero            -- la vista definida en 01_esquema.sql
GROUP BY genero
ORDER BY libros DESC;


-- Cuantas etiquetas tiene cada libro. El minimo es 2: no hay ni un solo libro
-- con un unico genero, asi que la clasificacion es multi-etiqueta por
-- construccion, no por decision nuestra.
SELECT jsonb_array_length(generos) AS cantidad_de_generos,
       count(*) AS libros
FROM libros
GROUP BY 1 ORDER BY 1;


-- Operador @>  : "el array CONTIENE este valor"
SELECT id, titulo FROM libros
WHERE generos @> '["Terror"]'
ORDER BY id LIMIT 5;

-- Operador ?|  : "el array contiene CUALQUIERA de estas claves"
-- Es el que usa traer_documentos() de src/corpus.py, y la razon por la que el
-- indice GIN se creo con jsonb_ops en vez de jsonb_path_ops (que solo soporta @>).
SELECT count(*) AS scifi_o_terror FROM libros
WHERE generos ?| array['Ciencia ficción', 'Terror'];


-- Libros que llegaron por mas de una categoria semilla: son los que justifican
-- haber deduplicado en la Fase A del scraper antes de visitar las fichas.
SELECT id, titulo, categoria_origen
FROM libros
WHERE jsonb_array_length(categoria_origen) > 1
ORDER BY jsonb_array_length(categoria_origen) DESC, id
LIMIT 10;


-- ===========================================================================
-- 4. Los indices: por que hoy no se usan, y como comprobar que funcionan
-- ===========================================================================
--
-- Con 200 filas la tabla entera entra en una pagina o dos, asi que recorrerla
-- completa es mas barato que consultar un indice y despues ir a buscar las
-- filas. El planificador elige Seq Scan, y hace bien.

EXPLAIN ANALYZE
SELECT id FROM libros WHERE generos @> '["Terror"]';
-- Esperado: Seq Scan on libros


-- Forzando a ignorar el Seq Scan se ve que el indice GIN existe y es usable.
-- Esto NO es una optimizacion: es una comprobacion de que el diseno es correcto
-- y de que escalaria. A 200 filas va a dar un tiempo igual o peor.
SET enable_seqscan = off;

EXPLAIN ANALYZE
SELECT id FROM libros WHERE generos @> '["Terror"]';
-- Esperado: Bitmap Index Scan on libros_generos_gin

RESET enable_seqscan;


-- ===========================================================================
-- 5. Comprobar que el texto NO esta normalizado
-- ===========================================================================
--
-- El corpus guarda el texto crudo: el preprocesamiento pertenece al modelo.
-- Estas consultas tienen que devolver filas, porque demuestran que siguen ahi
-- las mayusculas, las tildes y la puntuacion.

SELECT count(*) FILTER (WHERE sinopsis ~ '[A-ZÁÉÍÓÚÑ]')  AS con_mayusculas,
       count(*) FILTER (WHERE sinopsis ~ '[áéíóúñü]')     AS con_tildes,
       count(*) FILTER (WHERE sinopsis ~ '[.,;:¿?¡!]')    AS con_puntuacion,
       count(*) FILTER (WHERE sinopsis ~* '\mno\M')       AS con_negaciones
FROM libros;

-- Las negaciones son el caso mas delicado: "no" es stopword, asi que la version
-- tokenizada las pierde y "no es un thriller" queda igual que "es un thriller".
-- Por eso el corpus conserva el original y cada modelo decide que descartar.
SELECT id, titulo, substring(sinopsis from '[^.]*\mno\M[^.]*\.') AS oracion
FROM libros
WHERE sinopsis ~* '\mno\M'
ORDER BY id LIMIT 5;


-- ===========================================================================
-- 6. Ground truth gratis: las series
-- ===========================================================================
--
-- Los libros de una misma saga deberian salir como las primeras
-- recomendaciones de un recomendador que funcione. Es material de evaluacion
-- que no hubo que etiquetar a mano.

SELECT serie, count(*) AS tomos,
       string_agg(titulo, ' | ' ORDER BY serie_num) AS libros
FROM libros
WHERE serie IS NOT NULL
GROUP BY serie
HAVING count(*) > 1
ORDER BY tomos DESC;

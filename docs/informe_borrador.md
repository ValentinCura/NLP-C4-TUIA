# TP2 — Embeddings y búsqueda semántica: informe

**Grupo:** Sebastián Beltramo, Bautista Cortinas, Valentín Cura, Franco Maragliano.
**Corpus:** 200 sinopsis de Lectulandia (ciencia ficción, fantástico, terror, histórico).

> **BORRADOR.** Los números salen del notebook ejecutado con el conjunto de consultas
> **PROVISIONAL** (`queries_propuesta.json`, todavía sin validar por el grupo). Hay que
> reemplazarlos por los de la corrida definitiva antes de entregar. Los marcados con **[R]**
> tienen que revisarse contra el notebook final.

## 1. Qué se comparó y cómo se midió

Se compararon tres familias de representaciones sobre las mismas 16 consultas:
**léxica** (TF-IDF con el preprocesamiento del TP1, que es la línea de base, y TF-IDF de
n-gramas de caracteres como control), **promedio de word vectors** (Word2Vec y FastText
propios, entrenados sobre 1700 sinopsis, y SBW pre-entrenado) y **modelo de oración**
(SBERT `distiluse-base-multilingual-cased-v1`). Cada modelo recibe su propio preprocesamiento:
texto tokenizado para los léxicos y los promedios, texto crudo para SBERT.

La métrica es **precision@5**: la fracción de relevantes entre los 5 primeros, promediada
sobre las consultas. Para que el número signifique algo se fijaron tres cosas:

- **El piso de azar** es exacto: |R|/N. Con 16 consultas da 0,041 en promedio. La consulta
  amplia "algo que me dé miedo" tiene 54 relevantes y un piso de **0,27**: un modelo con
  0,30 ahí apenas supera al azar.
- **Los empates** (TF-IDF da 0 a todo lo que no comparte palabras) se resuelven con la
  precisión *esperada* bajo desempate aleatorio. `argsort` los ordenaría por posición en el
  CSV, y eso es arbitrario.
- **La variabilidad**: con 16 consultas el promedio depende de cuáles se eligieron. Se
  informa un intervalo bootstrap al 95 % y se compara cada modelo contra TF-IDF consulta por
  consulta.

| Representación | P@5 | IC 95 % | vs. TF-IDF (gana/empata/pierde) |
|---|---|---|---|
| SBERT (truncado) | **0,363** | [0,225; 0,513] | 8 / 7 / 1 — dif. +0,185 [+0,061; +0,335] |
| TF-IDF n-gramas de caracteres | 0,288 | [0,150; 0,425] | 5 / 8 / 3 — dif. +0,110 [+0,010; +0,235] |
| Word2Vec propio (promedio) | 0,200 | [0,088; 0,338] | 4 / 8 / 4 — dif. +0,023 (IC incluye 0) |
| FastText propio (promedio) | 0,200 | [0,112; 0,288] | 4 / 8 / 4 — dif. +0,023 (IC incluye 0) |
| **TF-IDF (TP1), línea de base** | 0,177 | [0,065; 0,314] | — |
| SBW pre-entrenado (promedio) | 0,163 | [0,088; 0,238] | 5 / 6 / 5 — dif. −0,015 (IC incluye 0) |
| Azar (piso) | 0,041 | | |

## 2. ¿Cuánto mejor es que TF-IDF, y en qué consultas?

SBERT supera a TF-IDF en **+0,185 de P@5** (casi el doble), y el intervalo de la diferencia
pareada no incluye el 0. Gana en 8 consultas, empata en 7 y pierde en 1. **[R]** La
ventaja **no es uniforme**:

- **Consultas léxicas** ("caballeros Jedi", "templarios"): **empate** (0,70 contra 0,70).
  Cuando la consulta usa las palabras raras del texto, TF-IDF ya acierta.
- **Consultas sin solapamiento léxico** (por ejemplo "bucaneros, filibusteros y abordajes en
  el océano", verificada con cero palabras en común con sus relevantes): TF-IDF da 0,005,
  *por debajo* del azar (0,017). Las palabras de la consulta aparecen en libros no
  relevantes y esos quedan primero. SBERT da 0,20. Es la diferencia de familia que la
  consigna quería hacer visible. **[R]**
- **Temáticas** ("viajes en el tiempo", "fantasmas y casas encantadas"): 0,12 contra 0,32.

Hay dos controles que matizan la conclusión:

1. **TF-IDF de n-gramas de caracteres** recupera parte de la brecha (0,288). Una parte de lo
   que pierde TF-IDF es *morfología*, no semántica. "Novelas de vampiros" no comparte
   ninguna palabra con 5 de sus 6 relevantes, que dicen "vampiro" o "vampira".
2. **Los promedios de word vectors no superan a TF-IDF**: los tres intervalos incluyen el 0.
   El SBW, el modelo con más datos, es el peor: su distribución de similitudes entre pares
   al azar tiene media 0,85 y desvío 0,06, o sea que todo se parece a todo y el ranking
   ordena diferencias mínimas. Es la dilución del promedio. SBERT tiene media 0,26 y desvío
   0,09.

En la tarea complementaria de **recuperar otro tomo de la misma saga**, el orden se invierte:
TF-IDF tiene MRR 0,881 y SBERT 0,828. Los tomos de una saga comparten nombres propios
rarísimos, que es justo lo que TF-IDF pondera al máximo.

## 3. Qué mide y qué no mide la métrica

**Mide** qué fracción de los primeros 5 resultados son libros que *nosotros* juzgamos
relevantes leyendo sus sinopsis.

**No mide:**

- el orden dentro de los 5 primeros;
- cuántos relevantes quedan afuera (recall);
- grados de relevancia: el juicio es binario;
- la diversidad, porque devolver 5 tomos de la misma saga cuenta igual que 5 libros
  distintos.

Además:

- **Tiene techo:** con 2 relevantes, P@5 no pasa de 0,4. Por eso se informa también
  R-precision, donde SBERT obtiene 0,379 y TF-IDF 0,226.
- **Depende de los juicios:** que son de un solo grupo, sobre la sinopsis y no sobre el
  libro. Los casos dudosos se trataron como no relevantes. Si se cuentan, SBERT sigue primero
  (0,475 contra 0,290 de TF-IDF), pero TF-IDF pasa a superar a los tres promedios de word
  vectors. **La posición relativa de TF-IDF y de los promedios depende de esa decisión;
  la de SBERT no.** **[R]**
- **Es un promedio de 16 consultas:** los intervalos de casi todos los modelos se pisan.
  Solo la ventaja de SBERT sobre TF-IDF es robusta.

## 4. Un caso donde la búsqueda falló

**[R] — revisar con la corrida definitiva.** La peor consulta de SBERT es *"historias de
viajes en el tiempo"*: **P@5 = 0**. Devolvió *La historia interminable*, *Cuentos del
antiguo Egipto* y antologías de relatos. Los 5 relevantes quedaron en los puestos 16 a 147.

**Hipótesis:** el modelo representa el *tipo de texto* (colecciones de "historias") más que
el motivo argumental. En los relevantes, el viaje en el tiempo no se nombra como tal: se
deduce de "año 1355", de un videojuego que lleva al año 1215, de "arrastrada en el tiempo".
Y queda subordinado a lo que domina la sinopsis: un romance escocés, una cruzada, Warcraft.

**Lo que se descartó con datos:** los 5 relevantes superan los 128 tokens y SBERT los
trunca. Pero la variante que codifica el texto completo por fragmentos **no** los mejora
(quedan entre los puestos 24 y 159). El problema no es el truncamiento.

## 5. ¿Qué modelo para producción?

**SBERT `distiluse-base-multilingual-cased-v1`**, con TF-IDF como complemento para
consultas con nombres propios. El criterio no es solo la métrica:

- **Costo:** 135 M de parámetros y 543 MB en disco, y corre en CPU, sin GPU. Medido en CPU:
  6 s para codificar las 200 sinopsis y 17 ms por consulta. TF-IDF cuesta prácticamente
  cero.
- **Privacidad:** el modelo corre **local**. Ni las consultas ni el catálogo salen a una API
  de terceros. Una alternativa por API (embeddings de un proveedor) sí enviaría cada consulta
  afuera.
- **Reproducibilidad:** la inferencia es determinista en CPU (dos codificaciones del corpus
  dan vectores idénticos), y el entrenamiento propio es reproducible byte a byte con
  `workers=1` y semilla fija. Las dos cosas están verificadas. El riesgo es que el
  modelo se descarga de Hugging Face y una nueva versión podría cambiar los vectores. Hay
  que fijar la revisión del modelo y guardar los embeddings en disco, como hace el notebook.
- **Dependencia de terceros:** depende de que Hugging Face siga publicando el modelo y de
  torch y sentence-transformers. TF-IDF solo depende de scikit-learn.
- **Límite conocido:** trunca el 86 % de las sinopsis (172 de 200, contando tokens de
  subpalabra; contando palabras se habrían estimado 118). En esta evaluación el truncamiento
  no cambió la P@5: la variante por fragmentos da 0,363 igual. Pero no es gratis en otras
  tareas.

**Contra elegir un promedio de word vectors:** no superaron a la línea base, y el modelo
propio, entrenado con solo 119 mil tokens, devuelve como vecinos nombres de personajes
(memoriza libros más que significados).

## 6. Otros hallazgos del trabajo

- **Idioma:** el corpus tiene una sinopsis en **gallego** (*Morning Star*). El detector
  (`lingua`) no tiene modelo de gallego y la etiquetó como castellano con confianza 0,998:
  un falso negativo silencioso. Sus dos términos TF-IDF de mayor peso son *dun* y *dunha*.
- **Sesgo promocional:** quitar el vocabulario publicitario no cambia el clasificador de
  género. La diferencia pareada en 20 splits es −0,000 ± 0,009. TF-IDF ya castiga esas
  palabras, porque aparecen en todas las sinopsis.

## 7. Uso de asistentes de IA

Usamos un asistente de IA (Claude, de Anthropic) para:

- auditar el código existente, lo que detectó tres errores que corregimos: la ablación
  promocional ignoraba las palabras con tilde, la conclusión de idioma era falsa y el
  entrenamiento no era reproducible;
- implementar el pipeline de evaluación y sus tests;
- **proponer** las consultas y los juicios de relevancia, leyendo las 200 sinopsis antes de
  ejecutar los modelos.

**[Completar por el grupo:** qué juicios de relevancia se validaron, se cambiaron o se
agregaron, y quién lo hizo.**]** Las conclusiones de este informe las revisamos contra los
números del notebook.

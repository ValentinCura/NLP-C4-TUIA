# Propuesta de `queries.json` — PROVISIONAL, para validar

> **Estado: pendiente de validación del grupo.** Hasta que la validen, la evaluación del notebook
> se marca como provisoria y no se pueden sacar conclusiones académicas de ella.

## Cómo se armó

Las consultas y los juicios de relevancia los propuso un asistente de IA leyendo las 200 sinopsis completas de data/libros.csv, ANTES de ejecutar cualquier modelo de busqueda. Los candidatos NO se buscaron por coincidencia de palabras (eso sesgaria la evaluacion a favor de TF-IDF): cada consulta se juzgo contra el corpus entero. Las consultas sin solapamiento se reformularon hasta tener cero palabras en comun con sus relevantes, verificado con la tokenizacion de TF-IDF, sin mirar resultados de ningun modelo.

**Criterio de relevancia.** Un libro es relevante si alguien que escribe la consulta quedaria satisfecho de encontrarlo, juzgando por lo que dice su sinopsis (no por el titulo ni por la etiqueta de genero, salvo en q06, donde se explicita). Juicio binario.

## Qué tienen que hacer ustedes

1. Para cada consulta: ¿la consulta está bien redactada? ¿la buscaría un lector real?
2. Para cada **relevante**: tildar si están de acuerdo. Si no, tacharlo.
3. Para cada **dudoso**: decidir si pasa a relevante o se descarta.
4. Si conocen un libro del corpus que falta, agregarlo (con su id de `data/libros.csv`).
5. **Háganlo sin mirar resultados de los modelos.** Después de validar, las consultas y la relevancia
   quedan congeladas: no se tocan aunque un modelo salga mal.

Al terminar, me pasan las correcciones y genero `queries.json` definitivo.

---

## q01 — «novelas de vampiros»

Tipo: `tematica`

> Parecía una consulta léxica, pero solo 1 de los 6 relevantes contiene alguna palabra de la consulta: los textos dicen 'vampiro' o 'vampira', y TF-IDF sin lematizar trata a 'vampiros' como otra palabra.

- [ ] Consulta aprobada

**Relevantes**

- [ ] `124988` *El invitado de Drácula* — relatos de Stoker en torno a Drácula
- [ ] `125304` *La sangre del vampiro* — protagonista vampira psíquica
- [ ] `125387` *Beber en rojo* — reescritura del mito de Drácula
- [ ] `125138` *El pájaro y el corazón de piedra* — protagonista convertida en vampira
- [ ] `125007` *Cuando caían las noches* — vampiro protagonista
- [ ] `125173` *La muerta enamorada y otros relatos fantásticos* — el vampirismo es uno de sus temas; el relato titular es un clásico de vampiros

**Dudosos** (decidir: ¿relevante o no?)

- [ ] `124530` *Alfa* — los vampiros aparecen como facción, pero la protagonista es híbrida humano-licántropo

---

## q02 — «criaturas chupasangres que acechan de noche»

Tipo: `sin_solapamiento` · **sin solapamiento léxico**

> Misma necesidad de información que q01 con otras palabras: aísla el efecto léxico. Comparte palabras con 19 libros NO relevantes (distractores léxicos). Dos relevantes contienen variantes morfológicas (criatura, noches) que TF-IDF no aprovecha porque no lematiza.

- [ ] Consulta aprobada

**Relevantes**

- [ ] `124988` *El invitado de Drácula* — igual que q01
- [ ] `125304` *La sangre del vampiro* — igual que q01
- [ ] `125387` *Beber en rojo* — igual que q01
- [ ] `125138` *El pájaro y el corazón de piedra* — igual que q01
- [ ] `125007` *Cuando caían las noches* — igual que q01
- [ ] `125173` *La muerta enamorada y otros relatos fantásticos* — igual que q01

**Dudosos** (decidir: ¿relevante o no?)

- [ ] `124530` *Alfa* — igual que q01

---

## q03 — «historias de viajes en el tiempo»

Tipo: `tematica`

- [ ] Consulta aprobada

**Relevantes**

- [ ] `124111` *Los años del cuervo* — el juego Hyperversum transporta a los personajes al año 1215
- [ ] `124512` *Norby salva al universo* — viaje por distintas épocas para restaurar el orden temporal
- [ ] `124684` *Buscando al Lobo de las Highlands* — la protagonista pasa de 2021 a 1355
- [ ] `125061` *El canto del cuerno* — Leonor vuelve a ser arrastrada en el tiempo
- [ ] `124915` *El pozo de la eternidad* — una brecha arrastra a los protagonistas a una época anterior

**Dudosos** (decidir: ¿relevante o no?)

- [ ] `124729` *Sombras en la eternidad* — regresiones mentales a las Cruzadas, no un viaje físico
- [ ] `124771` *Siete cuentos imposibles* — solo uno de los siete cuentos trata de viajar al futuro
- [ ] `124694` *Poshumanas y distópicas Vol. 1* — antología que menciona máquinas del tiempo entre sus temas
- [ ] `124819` *Poshumanas y distópicas Vol. 2* — ídem

---

## q04 — «extraterrestres que amenazan con invadir la Tierra»

Tipo: `tematica`

- [ ] Consulta aprobada

**Relevantes**

- [ ] `124376` *El problema de los Tres Cuerpos* — contacto con extraterrestres y amenaza de extinción
- [ ] `124375` *El bosque oscuro* — la Tierra tiene cuatro siglos para defenderse de Trisolaris
- [ ] `124373` *El fin de la muerte* — tercera parte del conflicto con Trisolaris
- [ ] `124433` *Señales desde las estrellas* — la humanidad es suplantada en silencio por algo que no es de la Tierra
- [ ] `124505` *Los ladrones de cerebros* — los Narks llegan de otra galaxia para apoderarse de la Tierra
- [ ] `124599` *Islas de la ascuaoscura* — invasores de las estrellas amenazan con conquistar al pueblo protagonista

**Dudosos** (decidir: ¿relevante o no?)

- [ ] `125331` *Triplanetario* — razas alienígenas manipulan a la humanidad, pero no es una invasión
- [ ] `124860` *El gran espectáculo* — relatos de primeros contactos alienígenas
- [ ] `124753` *Abominación atlántica* — ser no humano que amenaza a la humanidad, pero hallado en el fondo del mar

---

## q05 — «historias de fantasmas y casas encantadas»

Tipo: `tematica`

- [ ] Consulta aprobada

**Relevantes**

- [ ] `123734` *Esa maldita voz y otros relatos fantasmagóricos* — relatos de fantasmas
- [ ] `123757` *El Monte de las Ánimas y otras leyendas góticas* — leyendas con fantasmas y maldiciones
- [ ] `124102` *El Otro* — el espectro del marido muerto acosa a la viuda
- [ ] `124513` *Té para los fantasmas* — una pasadora de fantasmas busca el fantasma de su madre
- [ ] `124544` *La casa zafiro* — mansión con el fantasma de un asesino
- [ ] `124077` *Gótico* — casa que invade los sueños de la protagonista
- [ ] `124177` *La habitación secreta* — caserón con una habitación cerrada donde se oyen pasos y voces
- [ ] `124585` *Hay algo malo en casa* — un ente vive en la habitación de una niña
- [ ] `124991` *Venus en las tinieblas* — antología gótica con 'El espectro' y 'La casa encantada'

**Dudosos** (decidir: ¿relevante o no?)

- [ ] `125241` *El secreto de la señorita Primrose* — una médium ve a los muertos, pero es una intriga romántica
- [ ] `124210` *La chica nueva* — la chica de la que se enamora el protagonista está muerta
- [ ] `125315` *Despertar* — una madre cree ver el fantasma de su hijo; es una novela de posguerra
- [ ] `124539` *Ombria oculta* — ciudad subterránea llena de fantasmas, pero es fantasía épica
- [ ] `125173` *La muerta enamorada y otros relatos fantásticos* — apariciones y ultratumba entre sus temas

---

## q06 — «quiero leer algo que me dé miedo»

Tipo: `amplia`

> Consulta amplia a propósito, para mostrar el piso de azar: la relevancia es la etiqueta de género 'Terror' del sitio (54 libros, el 27% del corpus). Es la única consulta cuya relevancia sale de metadata y no de la lectura. Si se la deja, el piso de azar de P@k ronda 0,27.

- [ ] Consulta aprobada

**Relevantes:** todos los libros con el género `Terror` del sitio.

---

## q07 — «una historia de amor en la alta sociedad de la Inglaterra victoriana»

Tipo: `tematica`

- [ ] Consulta aprobada

**Relevantes**

- [ ] `125076` *Por una vida amándote* — romance con el heredero de un marquesado en Londres
- [ ] `125353` *La nobleza del conde* — condesa viuda y heredero, matrimonio de conveniencia
- [ ] `125354` *El corazón del duque* — romance entre una lady y un futuro duque
- [ ] `125394` *La pasión del marqués* — hija de barón y marqués
- [ ] `125396` *La redención del vizconde* — debutante y vizconde seductor
- [ ] `125406` *Una oveja descarriada para Norfolk* — la reina busca esposa para un duque
- [ ] `125407` *El regreso del caballero* — nuevo conde de Pembroke y una señorita
- [ ] `125408` *El destino de una rosa inglesa* — matrimonio arreglado entre una lady y un marqués
- [ ] `125170` *El lirio de Ludgate Hill* — lady de luto y un caballero
- [ ] `125316` *Una dulce propuesta para un conde* — conde y princesa en Londres
- [ ] `125314` *El corazón de Ruby* — lord que busca esposa en Londres
- [ ] `125379` *El hechizo de Agatha* — joven pobre y un lord en Londres
- [ ] `125473` *Aquel misterio que nos unió* — cozy mystery victoriano con romance con un vizconde

**Dudosos** (decidir: ¿relevante o no?)

- [ ] `125351` *Un corazón en peligro* — romance, pero la sinopsis no sitúa época ni clase social
- [ ] `125117` *Costuras de amor* — costurera y modisto de la aristocracia londinense; no son nobles
- [ ] `125241` *El secreto de la señorita Primrose* — Londres victoriano, pero centrada en la desaparición
- [ ] `125391` *Los vecinos de lady Chester* — comedia social victoriana sin historia de amor explícita
- [ ] `125105` *Una novia dada por muerta* — Belle Époque (posterior a la era victoriana); venganza más que romance
- [ ] `125273` *El mensajero* — amor prohibido en 1900 visto por un niño; no es novela romántica

---

## q08 — «novelas ambientadas en la Segunda Guerra Mundial»

Tipo: `tematica`

- [ ] Consulta aprobada

**Relevantes**

- [ ] `125431` *Mi querida Irene* — Irène Némirovsky y la deportación en la Francia ocupada
- [ ] `125527` *Extraños en el tiempo* — niños en el Londres del Blitz
- [ ] `125374` *En un hotel de Malmö* — Suecia 1940-1943, soldado en la frontera finlandesa
- [ ] `125265` *El nombre en el muro* — un maquis de la Resistencia durante la Ocupación

---

## q09 — «novelas sobre la guerra civil española»

Tipo: `tematica`

- [ ] Consulta aprobada

**Relevantes**

- [ ] `125155` *Tierra de sueños* — niños evacuados a México en 1937 y el tesoro de la República
- [ ] `125164` *El último caso de Unamuno* — Salamanca 1936 ocupada por los sublevados

**Dudosos** (decidir: ¿relevante o no?)

- [ ] `125161` *Los amores paralelos* — Asturias en los años 30 y la revolución de 1934, anterior a la guerra

---

## q10 — «robots e inteligencias artificiales»

Tipo: `tematica`

> Ninguno de los 6 relevantes contiene una palabra de la consulta: dicen 'robot', 'robótica', 'androide', 'IA'. Además, dos relevantes (Norby 124496 y 124498) tienen la sinopsis idéntica por un error del sitio.

- [ ] Consulta aprobada

**Relevantes**

- [ ] `124496` *La gran aventura de Norby* — robot protagonista, leyes de la robótica
- [ ] `124498` *Norby regresa a la Tierra* — ídem (sinopsis duplicada)
- [ ] `124512` *Norby salva al universo* — robot extraterrestre protagonista
- [ ] `124524` *Protocolo rebelde* — androide / IA asesina protagonista
- [ ] `124860` *El gran espectáculo* — un tanque autoconsciente
- [ ] `125075` *La máquina se detiene* — una Máquina omnisciente gobierna a la humanidad

**Dudosos** (decidir: ¿relevante o no?)

- [ ] `124694` *Poshumanas y distópicas Vol. 1* — antología que menciona inteligencias artificiales entre sus temas
- [ ] `124819` *Poshumanas y distópicas Vol. 2* — ídem
- [ ] `125358` *Imperio (Víctor Conde)* — aparece una inteligencia no orgánica, como elemento secundario

---

## q11 — «un mundo de fantasía con dragones»

Tipo: `tematica`

- [ ] Consulta aprobada

**Relevantes**

- [ ] `125132` *Las flores en llamas* — el Ejército Dracónico y un gran wyrm
- [ ] `124599` *Islas de la ascuaoscura* — una joven dragona encadenada en forma humana
- [ ] `124915` *El pozo de la eternidad* — los Dragones Aspectos en el apogeo de su poder
- [ ] `124870` *La historia interminable* — Fantasia, tierras de dragones y gigantes

**Dudosos** (decidir: ¿relevante o no?)

- [ ] `124512` *Norby salva al universo* — aparece la reina de las tierras de los grandes dragones, de forma secundaria

---

## q12 — «bucaneros, filibusteros y abordajes en el océano»

Tipo: `sin_solapamiento` · **sin solapamiento léxico**

> Cero palabras en común con los relevantes, y tampoco variantes morfológicas. Solo 2 libros del corpus comparten alguna palabra con la consulta.

- [ ] Consulta aprobada

**Relevantes**

- [ ] `125216` *La canción de Hands* — el tesoro del pirata Barbanegra
- [ ] `125274` *Malditos* — incluye relatos de piratas que aterrorizaron los mares

**Dudosos** (decidir: ¿relevante o no?)

- [ ] `124546` *Tu sangre en mis manos* — corsarios en el Mediterráneo del siglo XVI, como trasfondo
- [ ] `125417` *Patrulla galáctica* — piratas, pero espaciales

---

## q13 — «una epidemia que diezma a la población»

Tipo: `sin_solapamiento` · **sin solapamiento léxico**

> Cero palabras en común con los relevantes. En 125224 la peste ni siquiera se nombra: aparece como 'el Ángel Negro de la Muerte que asola el reino'.

- [ ] Consulta aprobada

**Relevantes**

- [ ] `125136` *Entre dos fuegos* — 1348, huérfana de la peste negra
- [ ] `125224` *Don Juan Manuel: El guardián de las palabras* — una mortandad que abate a campesinos, nobles y reyes

**Dudosos** (decidir: ¿relevante o no?)

- [ ] `124229` *Ahora todo es mejor* — una pandemia, pero de felicidad, no mortal

---

## q14 — «una sociedad futura sometida a un poder totalitario»

Tipo: `tematica`

- [ ] Consulta aprobada

**Relevantes**

- [ ] `124851` *El talón de hierro* — primera distopía política: una oligarquía aplasta a la sociedad
- [ ] `125075` *La máquina se detiene* — humanidad confinada y sometida a la Máquina
- [ ] `124768` *Amanecer en la cosecha* — el Capitolio somete a los distritos de Panem
- [ ] `125042` *Las sendas púrpuras* — la oligarquía del Purpurado condena planetas al aislamiento
- [ ] `125002` *Los sicarios de Dios* — planeta gobernado por el fanatismo, donde la intolerancia es norma

**Dudosos** (decidir: ¿relevante o no?)

- [ ] `124229` *Ahora todo es mejor* — etiquetada como distopía, pero el conflicto es corporativo, no totalitario
- [ ] `124694` *Poshumanas y distópicas Vol. 1* — antología con relatos distópicos
- [ ] `124819` *Poshumanas y distópicas Vol. 2* — ídem

---

## q15 — «la orden de los templarios»

Tipo: `lexica`

- [ ] Consulta aprobada

**Relevantes**

- [ ] `125128` *El guardián de los dioses* — tesoro y lista de maestres templarios en Orvieto
- [ ] `124729` *Sombras en la eternidad* — pergaminos que prueban que la Orden del Temple sobrevivió

---

## q16 — «aventuras de caballeros Jedi»

Tipo: `lexica`

> Consulta donde TF-IDF tiene todo a favor: 'Jedi' es un nombre propio raro que solo aparece en estos libros.

- [ ] Consulta aprobada

**Relevantes**

- [ ] `124737` *Cataclismo* — Star Wars: High Republic
- [ ] `124738` *Convergencia* — Star Wars: High Republic
- [ ] `124739` *La batalla de Jedha* — Star Wars: High Republic
- [ ] `124740` *El ojo de la oscuridad* — Star Wars: High Republic
- [ ] `124741` *Estrella caída* — Star Wars: High Republic


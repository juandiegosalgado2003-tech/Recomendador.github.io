from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

DOMINIOS_META: dict[str, dict[str, str]] = {
    "todo": {
        "nombre": "Todas las categorías",
        "subtitulo": "Exploración transversal de intereses combinados",
        "icono": "sparkles",
    },
    "peliculas": {
        "nombre": "Cine y Películas",
        "subtitulo": "Largometrajes de diversos géneros y cinematografías",
        "icono": "film",
    },
    "series": {
        "nombre": "Series de Televisión",
        "subtitulo": "Producciones seriadas, miniseries y temporadas",
        "icono": "tv",
    },
    "libros": {
        "nombre": "Literatura y Libros",
        "subtitulo": "Narrativa, divulgación científica, ensayo y clásicos",
        "icono": "book",
    },
    "musica": {
        "nombre": "Música y Álbumes",
        "subtitulo": "Obras discográficas, géneros y composiciones sonoras",
        "icono": "music",
    },
    "videojuegos": {
        "nombre": "Videojuegos",
        "subtitulo": "Obras interactivas, narrativa y diseño de juego",
        "icono": "gamepad",
    },
    "comida": {
        "nombre": "Gastronomía y Cocina",
        "subtitulo": "Recetas representativas, ingredientes y tradiciones culinarias",
        "icono": "utensils",
    },
    "viajes": {
        "nombre": "Viajes y Rutas",
        "subtitulo": "Destinos geográficos, patrimonio cultural y naturaleza",
        "icono": "compass",
    },
    "hobbies_regalos": {
        "nombre": "Actividades y Aficiones",
        "subtitulo": "Prácticas creativas, proyectos manuales y exploración",
        "icono": "gift",
    },
}


def _it(
    id_num: int,
    nombre: str,
    categoria: str,
    subcategoria: str,
    descripcion: str,
    etiquetas: list[str],
    texto_extra: str,
    tono: str,
    presupuesto: str,
    duracion: str,
    compania: list[str],
    icono: str,
) -> dict[str, Any]:
    texto_completo = " ".join([
        nombre,
        subcategoria,
        descripcion,
        " ".join(etiquetas),
        texto_extra,
        tono,
        presupuesto,
        duracion,
        " ".join(compania),
    ])
    return {
        "id": id_num,
        "nombre": nombre,
        "categoria": categoria,
        "subcategoria": subcategoria,
        "descripcion": descripcion,
        "etiquetas": ", ".join(etiquetas),
        "texto": texto_completo,
        "tono": tono,
        "presupuesto": presupuesto,
        "duracion": duracion,
        "compania": ", ".join(compania),
        "icono": icono,
    }


def obtener_catalogo_base() -> list[dict[str, Any]]:
    """Devuelve el catálogo curado multi-dominio listo para recomendación local."""
    items: list[dict[str, Any]] = []
    idx = 1

    def add(*args: Any) -> None:
        nonlocal idx
        items.append(_it(idx, *args))
        idx += 1

    # ---------------- PELÍCULAS ----------------
    add("Interestelar", "peliculas", "Ciencia ficción",
        "Un viaje espacial épico sobre el tiempo, la física y el vínculo entre un padre y su hija.",
        ["ciencia ficcion", "espacio", "emotivo", "banda sonora", "futuro"],
        "universo astronomia agujero negro familia drama reflexivo tecnologia nolan",
        "emotivo", "economico", "largo", ["solo", "pareja", "amigos"], "film")

    add("Spider-Man: Un nuevo universo", "peliculas", "Animación / Acción",
        "Animación vibrante con ritmo vertiginoso, humor fresco y corazón sobre encontrar tu identidad.",
        ["animacion", "accion", "superheroes", "dinamico", "colorido"],
        "aventura musica hip hop comedia familia amigos ritmo rapido energia",
        "divertido", "economico", "medio", ["solo", "amigos", "familia"], "film")

    add("Entre navajas y secretos (Knives Out)", "peliculas", "Misterio / Comedia",
        "Un detective excéntrico investiga un asesinato en una mansión llena de sospechosos brillantes.",
        ["misterio", "giros", "comedia inteligente", "detective", "suspenso"],
        "enigma investigacion familia herencia ingenio entretenida ritmo agil",
        "curioso", "economico", "medio", ["pareja", "amigos", "familia"], "film")

    add("El viaje de Chihiro", "peliculas", "Animación / Fantasía",
        "Fábula mágica de Studio Ghibli sobre un mundo espiritual lleno de criaturas inolvidables.",
        ["animacion", "fantasia", "magia", "nostalgico", "arte"],
        "ghibli japon espiritual amistad aventura relajado inspirador familia",
        "creativo", "economico", "medio", ["solo", "pareja", "familia"], "film")

    add("Mad Max: Furia en el camino", "peliculas", "Acción",
        "Persecución implacable por el desierto con dirección visual espectacular y pura adrenalina.",
        ["accion", "persecucion", "adrenalina", "postapocaliptico", "intenso"],
        "autos velocidad combate ritmo rapido energia explosion violencia",
        "adrenalina", "economico", "medio", ["solo", "amigos"], "film")

    add("Cuestión de tiempo (About Time)", "peliculas", "Romance / Comedia",
        "Un joven descubre que puede viajar en el tiempo y aprende a valorar cada día cotidiano.",
        ["romance", "emotivo", "familia", "comedia", "pareja"],
        "amor vida cotidiana londres calido sensible nostalgia corazon",
        "emotivo", "economico", "medio", ["solo", "pareja"], "film")

    add("Parásitos", "peliculas", "Thriller / Drama",
        "Sátira mordaz y llena de tensión sobre dos familias de mundos opuestos que se entrelazan.",
        ["suspenso", "giros", "critica social", "oscuro", "inteligente"],
        "corea tension thriller drama sorpresa oscuro psicologico",
        "oscuro", "economico", "medio", ["solo", "pareja", "amigos"], "film")

    add("El legado del diablo (Hereditary)", "peliculas", "Terror psicológico",
        "Terror atmosférico perturbador sobre secretos familiares y una presencia implacable.",
        ["terror", "miedo", "sustos", "oscuro", "psicologico"],
        "horror casa familia maldicion tension miedo intenso oscuro",
        "oscuro", "economico", "medio", ["solo", "amigos"], "film")

    add("La La Land", "peliculas", "Musical / Romance",
        "Historia luminosa de dos soñadores en Los Ángeles entre jazz, cine y ambiciones artísticas.",
        ["musical", "romance", "jazz", "arte", "nostalgico"],
        "musica baile suenos amor colores fotografia melancolia creativo",
        "emotivo", "economico", "medio", ["solo", "pareja"], "film")

    add("Mi maestro el pulpo", "peliculas", "Documental / Naturaleza",
        "Relato íntimo de un cineasta que construye una amistad insólita con un pulpo salvaje.",
        ["documental", "naturaleza", "mar", "relajado", "inspirador"],
        "oceano fotografia calma reflexion animales biologo lento contemplativo",
        "relajado", "economico", "corto", ["solo", "pareja", "familia"], "film")

    add("Todo en todas partes al mismo tiempo", "peliculas", "Ciencia ficción / Comedia",
        "Aventura multiversal absurda, emotiva y desbordante de creatividad visual.",
        ["ciencia ficcion", "multiverso", "comedia", "creativo", "accion"],
        "locura familia artes marciales giros original energia emotivo",
        "creativo", "economico", "medio", ["solo", "pareja", "amigos"], "film")

    add("Chef (A fuego lento)", "peliculas", "Comedia / Gastronomía",
        "Un cocinero deja un restaurante rígido para recorrer Estados Unidos en un food truck con su hijo.",
        ["comida", "comedia", "relajado", "familia", "viaje"],
        "cocina cubanos musica viaje carretera optimista ligero comida",
        "relajado", "economico", "medio", ["solo", "pareja", "familia"], "film")

    add("Duna: Parte Dos", "peliculas", "Ciencia ficción / Épica",
        "Epopeya visual monumental en el desierto de Arrakis sobre poder, profecía y destino.",
        ["ciencia ficcion", "epica", "aventura", "visual", "intenso"],
        "desierto arena imperio espacio futuro musica hans zimmer largo",
        "adrenalina", "economico", "largo", ["solo", "amigos"], "film")

    add("El Gran Hotel Budapest", "peliculas", "Comedia / Aventura",
        "Comedia estilizada de Wes Anderson llena de simetría, humor seco y un robo extravagante.",
        ["comedia", "arte", "estetica", "misterio", "creativo"],
        "europa hotel pintura aventura ingenio colorido divertido rapido",
        "divertido", "economico", "medio", ["solo", "pareja", "amigos"], "film")

    add("Intensamente (Inside Out)", "peliculas", "Animación / Familia",
        "Viaje tierno e ingenioso al interior de la mente y las emociones humanas.",
        ["animacion", "familia", "emotivo", "psicologia", "divertido"],
        "ninos mente sentimientos alegria tristeza risa corazon",
        "emotivo", "economico", "medio", ["solo", "familia", "pareja"], "film")

    add("Blade Runner 2049", "peliculas", "Ciencia ficción / Noir",
        "Neón, lluvia y preguntas filosóficas sobre memoria e identidad en un futuro distópico.",
        ["ciencia ficcion", "cyberpunk", "oscuro", "visual", "reflexivo"],
        "futuro inteligencia artificial estetica fotografia lento contemplativo",
        "oscuro", "economico", "largo", ["solo", "pareja"], "film")

    # ---------------- SERIES ----------------
    add("Dark", "series", "Ciencia ficción / Misterio",
        "Un enigma intergeneracional de viajes en el tiempo en un pueblo alemán lleno de secretos.",
        ["misterio", "ciencia ficcion", "oscuro", "giros", "complejo"],
        "tiempo paradoja suspenso rompecabezas alemania oscuro intenso",
        "oscuro", "economico", "largo", ["solo", "pareja"], "tv")

    add("Ted Lasso", "series", "Comedia / Feel-good",
        "Un optimista entrenador estadounidense dirige un club de fútbol inglés conquistando corazones.",
        ["comedia", "optimista", "deportes", "amistad", "ligero"],
        "humor futbol bienestar trabajo en equipo calido divertido facil",
        "divertido", "economico", "corto", ["solo", "pareja", "familia"], "tv")

    add("Severance (Separación)", "series", "Thriller / Ciencia ficción",
        "Empleados dividen quirúrgicamente sus recuerdos laborales de los personales con inquietantes consecuencias.",
        ["suspenso", "ciencia ficcion", "oficina", "misterio", "psicologico"],
        "tecnologia trabajo misterio estetica minimalista giros tension",
        "curioso", "economico", "medio", ["solo", "pareja"], "tv")

    add("The Bear", "series", "Drama / Gastronomía",
        "Un joven chef de alta cocina intenta rescatar el humilde local familiar de sándwiches en Chicago.",
        ["comida", "drama", "intenso", "cocina", "ritmo rapido"],
        "restaurante pasion estres familia superacion cocina intenso",
        "adrenalina", "economico", "corto", ["solo", "pareja"], "tv")

    add("Avatar: La leyenda de Aang", "series", "Animación / Aventura",
        "Aventura épica de amistad, equilibrio y dominio elemental ideal para cualquier edad.",
        ["animacion", "fantasia", "aventura", "amistad", "humor"],
        "elementos viaje equipo sabiduria accion familia divertido",
        "creativo", "economico", "medio", ["solo", "amigos", "familia"], "tv")

    add("Nuestro Planeta", "series", "Documental / Naturaleza",
        "Imágenes asombrosas de ecosistemas salvajes narradas con calma y belleza cinematográfica.",
        ["documental", "naturaleza", "relajado", "fotografia", "animales"],
        "tierra bosques oceanos calma aprender contemplativo familia",
        "relajado", "economico", "medio", ["solo", "pareja", "familia"], "tv")

    add("Arcane", "series", "Animación / Acción",
        "Obra visual deslumbrante sobre dos hermanas en ciudades enfrentadas entre magia y tecnología.",
        ["animacion", "accion", "steampunk", "arte", "emotivo"],
        "musica arte conflicto fantasia drama intenso visual",
        "adrenalina", "economico", "medio", ["solo", "amigos"], "tv")

    add("Only Murders in the Building", "series", "Misterio / Comedia",
        "Tres vecinos fanáticos del true crime resuelven asesinatos mientras graban su propio pódcast.",
        ["misterio", "comedia", "ligero", "nueva york", "enigma"],
        "podcast detectives diversion vecinos humor suave",
        "divertido", "economico", "corto", ["solo", "pareja", "familia"], "tv")

    add("Black Mirror", "series", "Antología / Tecnología",
        "Episodios independientes que exploran el lado inquietante de la tecnología y la sociedad moderna.",
        ["tecnologia", "ciencia ficcion", "oscuro", "reflexivo", "giros"],
        "futuro redes sociales inteligencia artificial distopia corto",
        "oscuro", "economico", "medio", ["solo", "amigos"], "tv")

    add("Frieren: Más allá del final del viaje", "series", "Anime / Fantasía",
        "Una elfa inmortal reflexiona sobre la amistad y el paso del tiempo tras derrotar al rey demonio.",
        ["anime", "fantasia", "relajado", "emotivo", "viaje"],
        "magia nostalgia paisajes calma filosofia aventura contemplativo",
        "relajado", "economico", "corto", ["solo", "pareja"], "tv")

    add("Succession", "series", "Drama / Sátira",
        "Lucha despiadada por el control de un imperio mediático con diálogos afilados.",
        ["drama", "negocios", "humor negro", "familia", "intenso"],
        "empresa poder dinero dialogo ingenioso lujo conflicto",
        "oscuro", "economico", "largo", ["solo", "pareja"], "tv")

    add("Chef's Table", "series", "Documental / Gastronomía",
        "Retratos íntimos y sensoriales de los cocineros más creativos del mundo y sus platos.",
        ["comida", "documental", "arte", "inspirador", "viajes"],
        "gastronomia ingredientes recetas creatividad cultura relajado",
        "creativo", "economico", "medio", ["solo", "pareja"], "tv")

    # ---------------- LIBROS ----------------
    add("Proyecto Hail Mary — Andy Weir", "libros", "Ciencia ficción",
        "Un astronauta despierta solo en una nave espacial y debe usar la ciencia para salvar la Tierra.",
        ["ciencia ficcion", "ciencia", "espacio", "aventura", "optimista"],
        "fisica quimica ingenio humor misterio amistad astronauta",
        "curioso", "economico", "largo", ["solo"], "book")

    add("Sapiens: De animales a dioses — Yuval Noah Harari", "libros", "Historia / Divulgación",
        "Un recorrido fascinante por cómo la humanidad construyó imperios, dinero, ciencia y mitos.",
        ["historia", "ciencia", "sociedad", "curiosidad", "ensayo"],
        "evolucion imperios cultura filosofia aprender reflexivo",
        "curioso", "economico", "largo", ["solo"], "book")

    add("La sombra del viento — Carlos Ruiz Zafón", "libros", "Novela / Misterio",
        "En la Barcelona de posguerra, un joven descubre un libro maldito que cambia su destino.",
        ["novela", "misterio", "nostalgico", "barcelona", "romance"],
        "libros secretos amor suspenso historia atmosfera emotivo",
        "emotivo", "economico", "largo", ["solo"], "book")

    add("Hábitos Atómicos — James Clear", "libros", "Crecimiento / Productividad",
        "Guía práctica y clara para transformar rutinas diarias con mejoras pequeñas y medibles.",
        ["habitos", "productividad", "psicologia", "practico", "negocios"],
        "mejora personal metas organizacion trabajo disciplina claro",
        "creativo", "economico", "medio", ["solo"], "book")

    add("El Hobbit — J.R.R. Tolkien", "libros", "Fantasía / Aventura",
        "Un hobbit hogareño emprende un viaje inesperado lleno de dragones, mapas y camaradería.",
        ["fantasia", "aventura", "viaje", "magia", "clasico"],
        "dragones amistad montanas aventura acogedor entretenido",
        "creativo", "economico", "medio", ["solo", "familia"], "book")

    add("Sal, grasa, ácido, calor — Samin Nosrat", "libros", "Cocina / Gastronomía",
        "Domina los cuatro pilares universales de la buena cocina con ilustraciones y humor.",
        ["cocina", "comida", "recetas", "aprender", "ilustrado"],
        "ingredientes tecnicas sabor cocinar creativo casa",
        "relajado", "moderado", "medio", ["solo", "pareja"], "book")

    add("El problema de los tres cuerpos — Cixin Liu", "libros", "Ciencia ficción dura",
        "Una señal enviada al espacio desencadena una conspiración científica a escala cósmica.",
        ["ciencia ficcion", "fisica", "misterio", "universo", "tecnologia"],
        "espacio alienigenas historia china ciencia complejo futuro",
        "curioso", "economico", "largo", ["solo"], "book")

    add("Cien años de soledad — Gabriel García Márquez", "libros", "Novela / Realismo mágico",
        "La saga mítica de la familia Buendía en Macondo, entre memoria, amor y destino.",
        ["novela", "realismo magico", "clasico", "familia", "latinoamerica"],
        "literatura poesia historia fantasia emotivo profundo",
        "emotivo", "economico", "largo", ["solo"], "book")

    add("El hombre en busca de sentido — Viktor Frankl", "libros", "Psicología / Filosofía",
        "Testimonio conmovedor y lección profunda sobre cómo hallar propósito incluso en la adversidad.",
        ["psicologia", "filosofia", "emotivo", "reflexivo", "corto"],
        "resiliencia vida sentido historia superacion",
        "emotivo", "economico", "corto", ["solo"], "book")

    add("Diez negritos (Y no quedó ninguno) — Agatha Christie", "libros", "Misterio / Policial",
        "Diez desconocidos aislados en una isla van cayendo uno a uno bajo un plan calculado.",
        ["misterio", "suspenso", "detective", "giros", "rapido"],
        "enigma crimen isla suspenso facil de leer entretenido",
        "oscuro", "economico", "corto", ["solo"], "book")

    add("La psicología del dinero — Morgan Housel", "libros", "Finanzas / Negocios",
        "Historias breves que explican cómo nuestras emociones gobiernan nuestras decisiones financieras.",
        ["finanzas", "negocios", "psicologia", "practico", "inversion"],
        "dinero ahorro estrategia mente empresa claro",
        "curioso", "economico", "corto", ["solo"], "book")

    add("Cosmos — Carl Sagan", "libros", "Ciencia / Astronomía",
        "Un canto poético a la astronomía, la evolución y nuestro lugar en el universo.",
        ["ciencia", "universo", "astronomia", "inspirador", "historia"],
        "estrellas planetas filosofia curiosidad maravilla",
        "relajado", "economico", "medio", ["solo"], "book")

    # ---------------- MÚSICA ----------------
    add("Lofi Beats & Jazz de Medianoche", "musica", "Lo-Fi / Chillhop",
        "Ritmos cálidos de vinilo, piano eléctrico suave y texturas relajantes para estudiar o desconectar.",
        ["relajado", "estudio", "jazz", "instrumental", "calma"],
        "concentracion cafe lluvia noche tranquilo sin letra",
        "relajado", "economico", "corto", ["solo", "pareja"], "music")

    add("Discovery — Daft Punk", "musica", "Electrónica / French House",
        "Clásico luminoso de sintetizadores, groove retrofuturista y energía positiva de principio a fin.",
        ["electronica", "bailar", "energia", "retro", "futurista"],
        "fiesta sintetizadores ritmo divertido carretera amigos",
        "divertido", "economico", "medio", ["solo", "amigos"], "music")

    add("Kind of Blue — Miles Davis", "musica", "Jazz clásico",
        "Elegancia atemporal de trompeta y saxofón perfecta para una noche serena o una cena especial.",
        ["jazz", "clasico", "nocturno", "sofisticado", "instrumental"],
        "vinilo calma pareja copa elegante reflexivo",
        "relajado", "economico", "medio", ["solo", "pareja"], "music")

    add("El Madrileño — C. Tangana", "musica", "Pop latino / Fusión",
        "Fusión audaz de bolero, flamenco, rumba y sonidos urbanos contemporáneos con gran producción.",
        ["latino", "fusion", "emotivo", "fiesta", "espanol"],
        "guitarra ritmo nostalgia moderna amigos voz",
        "creativo", "economico", "medio", ["solo", "pareja", "amigos"], "music")

    add("Bandas Sonoras Épicas (Hans Zimmer / Joe Hisaishi)", "musica", "Orquestal / Cine",
        "Piezas sinfónicas cargadas de emoción para inspirarse, leer o imaginar mundos gigantes.",
        ["orquestal", "cine", "epico", "emotivo", "instrumental"],
        "piano cuerdas aventura inspiracion concentracion fantasia",
        "emotivo", "economico", "medio", ["solo"], "music")

    add("Synthwave Nocturno (Midnight Drive)", "musica", "Synthwave / Retro 80s",
        "Bajos analógicos y melodías neón ideales para conducir de noche o programar con enfoque.",
        ["synthwave", "retro", "nocturno", "energia", "cyberpunk"],
        "coche ciudad luces tecnologia concentracion ritmo",
        "adrenalina", "economico", "medio", ["solo", "amigos"], "music")

    add("Currents — Tame Impala", "musica", "Indie / Psicodélico",
        "Capas hipnóticas de pop psicodélico y sintetizadores sobre el cambio personal.",
        ["indie", "psicodelico", "moderno", "creativo", "melodico"],
        "rock suave sintetizador reflexivo vibra nocturna",
        "creativo", "economico", "medio", ["solo", "pareja", "amigos"], "music")

    add("Un Verano Sin Ti — Bad Bunny", "musica", "Urbano / Caribeño",
        "Ritmos playeros, mambo y reguetón festivo para subir el ánimo al instante.",
        ["urbano", "playa", "fiesta", "verano", "divertido"],
        "baile amigos coche sol energia alegre",
        "divertido", "economico", "medio", ["amigos", "pareja"], "music")

    add("Folklore — Taylor Swift", "musica", "Indie Folk / Pop",
        "Baladas acústicas íntimas llenas de relatos cotidianos, piano suave y melancolía otoñal.",
        ["acustico", "folk", "emotivo", "tranquilo", "letras"],
        "otono lluvia historias guitarra sensible",
        "emotivo", "economico", "medio", ["solo", "pareja"], "music")

    add("Random Access Memories — Daft Punk", "musica", "Disco / Funk Electrónico",
        "Producción orgánica impecable con guitarras funk, vocoders y atmósfera cinematográfica.",
        ["funk", "disco", "electronica", "bailar", "calidad"],
        "fiesta groove energia amigos elegante",
        "divertido", "economico", "medio", ["solo", "pareja", "amigos"], "music")

    add("Bossa Nova de Domingo (João Gilberto & Tom Jobim)", "musica", "Bossa Nova / Acústico",
        "Guitarra suave brasileña y brisa cálida para desayunos tranquilos o tardes de sol.",
        ["bossa nova", "acustico", "relajado", "sol", "cafe"],
        "brasil calma desayuno playa pareja ligero",
        "relajado", "economico", "corto", ["solo", "pareja", "familia"], "music")

    add("Dark Side of the Moon — Pink Floyd", "musica", "Rock progresivo",
        "Viaje sonoro conceptual sobre el tiempo, la mente y la existencia humana.",
        ["rock", "clasico", "reflexivo", "espacio", "conceptual"],
        "vinilo guitarra psicodelico profundo nocturno",
        "curioso", "economico", "medio", ["solo"], "music")

    # ---------------- VIDEOJUEGOS ----------------
    add("Stardew Valley", "videojuegos", "Simulación / Cozy",
        "Hereda una granja tranquila, cultiva, explora cuevas y haz amistad con un pueblo encantador.",
        ["relajado", "cozy", "granja", "creativo", "cooperativo"],
        "pixel art calma gestion sin estres amigos pareja economico",
        "relajado", "economico", "largo", ["solo", "pareja", "amigos"], "gamepad")

    add("The Legend of Zelda: Breath of the Wild", "videojuegos", "Aventura / Mundo abierto",
        "Libertad total para escalar montañas, resolver santuarios y explorar un reino vasto a tu ritmo.",
        ["aventura", "mundo abierto", "exploracion", "fantasia", "puzles"],
        "naturaleza descubrimiento ingenio combate suave epico",
        "creativo", "moderado", "largo", ["solo"], "gamepad")

    add("Hades", "videojuegos", "Acción / Roguelike",
        "Combate frenético, arte increíble y diálogos brillantes mientras escapas del inframundo.",
        ["accion", "adrenalina", "mitologia", "rapido", "rejugable"],
        "combate ritmo rapido energia musica dinamico",
        "adrenalina", "economico", "corto", ["solo"], "gamepad")

    add("It Takes Two", "videojuegos", "Cooperativo / Aventura",
        "Aventura creativa pensada exclusivamente para jugar de a dos con mecánicas siempre nuevas.",
        ["cooperativo", "pareja", "amigos", "divertido", "creativo"],
        "puzles humor aventura compartir pantalla juntos",
        "divertido", "moderado", "medio", ["pareja", "amigos", "familia"], "gamepad")

    add("Outer Wilds", "videojuegos", "Exploración / Misterio",
        "Explora un sistema solar en miniatura atrapado en un bucle temporal para descifrar un enigma ancestral.",
        ["misterio", "espacio", "ciencia ficcion", "exploracion", "inteligente"],
        "astronomia curiosidad puzles emotivo musica sin combate",
        "curioso", "economico", "medio", ["solo"], "gamepad")

    add("Hollow Knight", "videojuegos", "Metroidvania / Atmosférico",
        "Explora un reino subterráneo melancólico dibujado a mano con combates exigentes y secretos.",
        ["aventura", "oscuro", "desafiante", "arte", "exploracion"],
        "plataformas precision misterio banda sonora profundo",
        "oscuro", "economico", "largo", ["solo"], "gamepad")

    add("Portal 2", "videojuegos", "Puzles / Comedia",
        "Acertijos brillantes con portales espaciales acompañados de un humor mordaz inolvidable.",
        ["puzles", "comedia", "ciencia ficcion", "cooperativo", "ingenio"],
        "logica laboratorio robot divertido pareja amigos",
        "curioso", "economico", "medio", ["solo", "pareja", "amigos"], "gamepad")

    add("Minecraft", "videojuegos", "Sandbox / Creatividad",
        "Construye cualquier mundo imaginable, sobrevive o explora en solitario o con amigos.",
        ["creativo", "construccion", "familia", "amigos", "infinito"],
        "bloques aventura relajado cooperativo imaginacion",
        "creativo", "economico", "largo", ["solo", "amigos", "familia"], "gamepad")

    add("Baldur's Gate 3", "videojuegos", "RPG / Fantasía",
        "Aventura de rol profunda donde cada decisión transforma la historia y tus compañeros.",
        ["rpg", "fantasia", "estrategia", "narrativa", "cooperativo"],
        "magia decisiones combates por turnos historia larga amigos",
        "curioso", "moderado", "largo", ["solo", "amigos"], "gamepad")

    add("Overcooked! All You Can Eat", "videojuegos", "Party / Cooperativo",
        "Caos culinario divertidísimo en equipo preparando pedidos antes de que arda la cocina.",
        ["fiesta", "amigos", "familia", "divertido", "rapido"],
        "cocina risas multijugador equipo corto energia",
        "divertido", "economico", "corto", ["pareja", "amigos", "familia"], "gamepad")

    add("Alan Wake 2", "videojuegos", "Survival Horror / Thriller",
        "Thriller psicológico oscuro y cinematográfico que mezcla investigación policial y pesadillas.",
        ["terror", "misterio", "oscuro", "narrativa", "sustos"],
        "horror suspenso historia adulto miedo violencia",
        "oscuro", "moderado", "largo", ["solo"], "gamepad")

    add("Civilization VI", "videojuegos", "Estrategia por turnos",
        "Guía una civilización desde la Edad de Piedra hasta la era espacial con táctica e historia.",
        ["estrategia", "historia", "pensar", "gestion", "largo"],
        "imperio diplomacia ciencia mapas calma mental",
        "curioso", "economico", "largo", ["solo", "amigos"], "gamepad")

    # ---------------- COMIDA Y RECETAS ----------------
    add("Tacos al Pastor Caseros con Piña Asada", "comida", "Mexicana / Intensa",
        "Carne especiada con achiote, piña caramelizada, cilantro fresco y salsa picante vibrante.",
        ["picante", "especias", "amigos", "intenso", "mexicana"],
        "tacos tortilla limon sabor fiesta compartir cena",
        "divertido", "economico", "medio", ["pareja", "amigos", "familia"], "utensils")

    add("Pasta Cacio e Pepe Auténtica", "comida", "Italiana / Rápida",
        "Crema sedosa de queso pecorino y pimienta negra recién molida lista en 15 minutos.",
        ["rapido", "italiana", "pasta", "reconfortante", "economico"],
        "queso cena facil casera pocos ingredientes delicioso",
        "relajado", "economico", "corto", ["solo", "pareja"], "utensils")

    add("Ramen Casero de Miso y Setas", "comida", "Asiática / Reconfortante",
        "Caldo umami aromático con fideos, huevo marinado, setas salteadas y aceite de sésamo.",
        ["asiatica", "sopa", "caliente", "reconfortante", "noche"],
        "japon fideos caldo invierno sabroso lluvia",
        "relajado", "moderado", "medio", ["solo", "pareja"], "utensils")

    add("Bowl Mediterráneo de Salmón, Aguacate y Quinoa", "comida", "Saludable / Fresca",
        "Plato colorido lleno de energía limpia, limón, hierbas frescas y semillas tostadas.",
        ["saludable", "fresco", "rapido", "nutritivo", "ligero"],
        "ensalada pescado verano fit equilibrado almuerzo",
        "relajado", "moderado", "corto", ["solo", "pareja"], "utensils")

    add("Brownie de Chocolate Oscuro con Sal Marina", "comida", "Postres / Repostería",
        "Centro húmedo de cacao intenso con corteza crujiente y un toque de flor de sal.",
        ["dulce", "postre", "chocolate", "horno", "antojo"],
        "reposteria hornear casa tarde cafe pareja amigos",
        "emotivo", "economico", "medio", ["solo", "pareja", "amigos", "familia"], "utensils")

    add("Curry Verde Tailandés de Verduras y Coco", "comida", "Asiática / Vegetariana",
        "Aromático, ligeramente picante y cremoso con leche de coco, albahaca y arroz jazmín.",
        ["vegetariano", "picante", "asiatica", "especias", "arroz"],
        "tailandia coco verduras colorido cena sabroso",
        "creativo", "economico", "medio", ["solo", "pareja", "amigos"], "utensils")

    add("Shakshuka (Huevos en Salsa de Tomate Especiada)", "comida", "Brunch / Casera",
        "Sartén burbujeante de tomate, pimiento, comino y huevos escalfados para mojar pan crujiente.",
        ["desayuno", "rapido", "vegetariano", "sarten", "economico"],
        "huevos brunch tomate especias facil fin de semana",
        "relajado", "economico", "corto", ["solo", "pareja", "familia"], "utensils")

    add("Pizza Napolitana Casera en Sartén y Horno", "comida", "Italiana / En grupo",
        "Masa aireada con salsa de tomate San Marzano, mozzarella fresca y albahaca aromática.",
        ["italiana", "amigos", "familia", "divertido", "horno"],
        "queso fin de semana cocinar juntos fiesta",
        "divertido", "economico", "medio", ["pareja", "amigos", "familia"], "utensils")

    add("Ceviche Clásico de Pescado Blanco y Cítricos", "comida", "Fresco / Marino",
        "Explosión cítrica de lima, cilantro, cebolla morada y ají fresco muy refrescante.",
        ["fresco", "marino", "verano", "picante", "saludable"],
        "limon pescado ligero entrada elegante amigos",
        "adrenalina", "moderado", "corto", ["pareja", "amigos"], "utensils")

    add("Affogato de Café Espresso y Helado de Vainilla", "comida", "Postre / Café",
        "El contraste perfecto entre espresso caliente recién hecho y helado artesanal frío.",
        ["dulce", "cafe", "rapido", "italiana", "postre"],
        "helado espresso elegante cinco minutos pareja",
        "relajado", "economico", "corto", ["solo", "pareja"], "utensils")

    add("Risotto Cremoso de Hongos Porcini", "comida", "Italiana / Gourmet",
        "Arroz arborio cocinado lentamente con caldo caliente, vino blanco y setas aromáticas.",
        ["gourmet", "italiana", "vegetariano", "pareja", "cocinar"],
        "cena especial setas queso vino paciente",
        "emotivo", "moderado", "medio", ["pareja", "amigos"], "utensils")

    add("Pollo Koreano Crujiente Dulce y Picante", "comida", "Asiática / Crujiente",
        "Bocados doblemente crujientes glaseados en salsa gochujang, miel, ajo y sésamo.",
        ["picante", "asiatica", "crujiente", "amigos", "intenso"],
        "corea cena pelicula noche sabor potente",
        "adrenalina", "economico", "medio", ["solo", "amigos"], "utensils")

    # ---------------- VIAJES Y ESCAPADAS ----------------
    add("Kioto, Japón — Templos, Bosques de Bambú y Té Matcha", "viajes", "Cultura y Tradición",
        "Calles tranquilas de madera, jardines zen, gastronomía refinada y santuarios milenarios.",
        ["cultura", "historia", "gastronomia", "tranquilo", "fotografia"],
        "japon templos arte caminar inspirador especial caro",
        "curioso", "especial", "largo", ["solo", "pareja"], "compass")

    add("Oaxaca, México — Mercados, Mezcal y Arquitectura Viva", "viajes", "Escapada Gastronómica",
        "Explosión de sabores tradicionales, artesanías textiles, plazas coloridas y sitios arqueológicos.",
        ["comida", "cultura", "colorido", "historia", "artesania"],
        "mercados cocina chocolate cafe amigos calido",
        "creativo", "moderado", "medio", ["solo", "pareja", "amigos"], "compass")

    add("Cabaña en la Montaña y Senderismo de Fin de Semana", "viajes", "Naturaleza / Desconexión",
        "Aire puro de bosque, fogata nocturna, caminatas entre pinos y cero notificaciones.",
        ["naturaleza", "montana", "relajado", "desconexion", "economico"],
        "bosque calma lectura pareja amigos aire libre",
        "relajado", "economico", "corto", ["solo", "pareja", "amigos", "familia"], "compass")

    add("Lisboa, Portugal — Tranvías, Miradores y Pastéis de Nata", "viajes", "Ciudad con Encanto",
        "Colinas soleadas frente al Atlántico, fado nocturno, cafés históricos y ritmo relajado.",
        ["ciudad", "historia", "comida", "romantico", "fotografia"],
        "europa mar caminar arquitectura cafe pareja",
        "emotivo", "moderado", "medio", ["solo", "pareja", "amigos"], "compass")

    add("Costa Caribeña / Playa Escondida de Aguas Turquesas", "viajes", "Playa y Descanso",
        "Hamacas bajo palmeras, snorkel en arrecifes y atardeceres dorados sin prisas.",
        ["playa", "sol", "relajado", "mar", "verano"],
        "nadar descanso pareja amigos calor vacaciones",
        "relajado", "moderado", "medio", ["pareja", "amigos", "familia"], "compass")

    add("Ruta en Caravana por la Patagonia o los Fiordos", "viajes", "Aventura Épica",
        "Glaciares imponentes, lagos turquesa y libertad absoluta recorriendo paisajes salvajes.",
        ["aventura", "naturaleza", "montana", "fotografia", "epico"],
        "senderismo carretera aire libre adrenalina paisaje largo",
        "adrenalina", "especial", "largo", ["solo", "pareja", "amigos"], "compass")

    add("Madrid — Museos, Tapas y Vida Nocturna", "viajes", "Ciudad Cultural",
        "Del Prado y el Retiro a terrazas animadas de tapas, librerías y teatros.",
        ["ciudad", "arte", "comida", "amigos", "divertido"],
        "museos historia caminar gastronomia fiesta cultura",
        "divertido", "moderado", "medio", ["solo", "pareja", "amigos"], "compass")

    add("Eje Cafetero, Colombia — Fincas de Café y Palma de Cera", "viajes", "Naturaleza y Café",
        "Montañas verdes, degustación de café de altura, pueblos coloridos y valles brumosos.",
        ["cafe", "naturaleza", "economico", "tranquilo", "cultura"],
        "montana senderismo finca gastronomia calido familia",
        "relajado", "economico", "medio", ["solo", "pareja", "familia"], "compass")

    add("Islandia — Auroras Boreales, Géiseres y Aguas Termales", "viajes", "Naturaleza Extrema",
        "Paisajes volcánicos de otro planeta, cascadas gigantes y baños geotermales bajo las estrellas.",
        ["aventura", "fotografia", "naturaleza", "espacial", "unico"],
        "auroras frio termas carretera caro especial",
        "curioso", "especial", "largo", ["solo", "pareja", "amigos"], "compass")

    add("Escapada Urbana a Cafeterías, Librerías y Cine Independiente", "viajes", "Plan Local / Micro-viaje",
        "Redescubre tu propia ciudad en un día sin coche: café de especialidad, libro nuevo y película.",
        ["economico", "corto", "cafe", "cultura", "relajado"],
        "ciudad fin de semana barato solo pareja inspirador",
        "relajado", "economico", "corto", ["solo", "pareja"], "compass")

    # ---------------- HOBBIES, PLANES Y REGALOS ----------------
    add("Kit de Café de Especialidad (V60 + Molinillo Manual)", "hobbies_regalos", "Gastronomía / Regalo",
        "Convierte cada mañana en un ritual aromático descubriendo notas de origen único.",
        ["cafe", "ritual", "regalo", "relajado", "casa"],
        "cocina manana aroma preciso artesanal disfrutar",
        "relajado", "moderado", "corto", ["solo", "pareja"], "gift")

    add("Juego de Mesa: Catan o Azul", "hobbies_regalos", "Juegos / Plan con Amigos",
        "Estrategia accesible, comercio o diseño de mosaicos ideal para noches divertidas en mesa.",
        ["juegos de mesa", "estrategia", "amigos", "familia", "divertido"],
        "regalo noche casa risas pensar sin pantallas",
        "divertido", "economico", "medio", ["pareja", "amigos", "familia"], "gift")

    add("Taller de Cerámica Artesanal en Torno", "hobbies_regalos", "Arte / Experiencia",
        "Desconecta las manos del teclado moldeando tazas y cuencos únicos con arcilla.",
        ["arte", "creativo", "manos", "relajado", "experiencia"],
        "taller regalo pareja calma mindfulness crear",
        "creativo", "moderado", "corto", ["solo", "pareja", "amigos"], "gift")

    add("Cámara Analógica Compacta de 35mm + Carrete", "hobbies_regalos", "Fotografía / Regalo",
        "Vuelve a fotografiar momentos con intención, grano orgánico y la emoción de revelar.",
        ["fotografia", "arte", "nostalgico", "viajes", "regalo"],
        "retro creativo recuerdos paseos mirada",
        "creativo", "moderado", "medio", ["solo", "pareja"], "gift")

    add("Telescopio Portátil o Prismáticos Astronómicos", "hobbies_regalos", "Ciencia / Aire Libre",
        "Observa cráteres lunares, los anillos de Saturno y cúmulos estelares en noches despejadas.",
        ["astronomia", "ciencia", "universo", "naturaleza", "regalo"],
        "estrellas cielo noche curiosidad familia aprender",
        "curioso", "moderado", "medio", ["solo", "pareja", "familia"], "gift")

    add("Curso Práctico de Cocina o Repostería en Vivo", "hobbies_regalos", "Experiencia Culinaria",
        "Aprende técnicas reales de pasta fresca, ramen o pan de masa madre y disfruta lo cocinado.",
        ["comida", "cocina", "aprender", "experiencia", "pareja"],
        "regalo divertido cena sabor amigos",
        "divertido", "moderado", "corto", ["solo", "pareja", "amigos"], "gift")

    add("Diario de Dibujo Urbano + Acuarelas de Bolsillo", "hobbies_regalos", "Arte / Portátil",
        "Captura cafés, parques y viajes en pequeñas láminas de color sin necesidad de ser experto.",
        ["arte", "pintura", "viajes", "economico", "relajado"],
        "cuaderno creatividad calma parque regalo",
        "relajado", "economico", "corto", ["solo"], "gift")

    add("Raspberry Pi o Kit de Domótica y Electrónica DIY", "hobbies_regalos", "Tecnología / Maker",
        "Construye retro-consolas, estaciones meteorológicas o automatizaciones inteligentes en casa.",
        ["tecnologia", "programacion", "ingenio", "proyecto", "regalo"],
        "electronica aprender ordenador hogar curioso",
        "curioso", "moderado", "largo", ["solo", "amigos"], "gift")

    add("Noche de Observación de Estrellas con Manta y Termo", "hobbies_regalos", "Plan Económico / Aire Libre",
        "Escápate a un mirador oscuro con mapa estelar en el móvil y algo caliente para beber.",
        ["economico", "romantico", "naturaleza", "nocturno", "relajado"],
        "pareja estrellas cielo calma gratis plan",
        "emotivo", "economico", "corto", ["solo", "pareja", "amigos"], "gift")

    add("Auriculares de Estudio con Cancelación de Ruido", "hobbies_regalos", "Música / Regalo Tech",
        "Sumérgete en discos completos, pódcasts o trabajo profundo sin distracciones externas.",
        ["musica", "tecnologia", "trabajo", "viajes", "regalo"],
        "sonido concentracion vuelos calidad caro",
        "relajado", "especial", "largo", ["solo"], "gift")

    add("Escalada Indoor (Búlder) para Principiantes", "hobbies_regalos", "Deporte / Adrenalina",
        "Puzles físicos en la pared que combinan fuerza, equilibrio y comunidad amigable.",
        ["deporte", "adrenalina", "retos", "amigos", "energia"],
        "movimiento salud diversion puzle fisico plan",
        "adrenalina", "economico", "corto", ["solo", "pareja", "amigos"], "gift")

    add("Terrario de Plantas Autosustentable en Frasco de Cristal", "hobbies_regalos", "Naturaleza / Decoración",
        "Crea un mini bosque húmedo con musgo y helechos que decora tu escritorio casi sin cuidados.",
        ["naturaleza", "plantas", "creativo", "economico", "regalo"],
        "hogar verde calma artesanal decoracion",
        "relajado", "economico", "corto", ["solo", "pareja", "familia"], "gift")

    return items


def generar_interacciones_iniciales(items_df: pd.DataFrame, semilla: int = 42) -> pd.DataFrame:
    """Genera perfiles realistas iniciales para habilitar filtrado colaborativo (SVD) desde el primer minuto."""
    rng = np.random.default_rng(semilla)

    arquetipos = [
        ("Sofía (Cinéfila y Viajera)", ["ciencia ficcion", "emotivo", "viajes", "cultura", "fotografia"], ["terror", "sustos"]),
        ("Mateo (Gamer y Lector Sci-Fi)", ["ciencia ficcion", "espacio", "rpg", "estrategia", "tecnologia"], ["romance"]),
        ("Lucía (Foodie y Creativa)", ["comida", "cocina", "arte", "cafe", "postre"], ["terror", "violencia"]),
        ("Carlos (Amante del Jazz y Historia)", ["jazz", "historia", "clasico", "documental", "relajado"], ["adrenalina"]),
        ("Valentina (Plan con Amigos y Comedia)", ["divertido", "comedia", "amigos", "cooperativo", "fiesta"], ["oscuro", "triste"]),
        ("Diego (Adrenalina y Misterio)", ["accion", "misterio", "oscuro", "adrenalina", "giros"], ["lento"]),
        ("Elena (Bienestar y Naturaleza)", ["naturaleza", "relajado", "saludable", "animacion", "cozy"], ["terror", "violencia"]),
        ("Andrés (Curioso Multitema)", ["curioso", "ciencia", "misterio", "puzles", "gastronomia"], []),
    ]

    filas: list[dict[str, Any]] = []
    # 8 usuarios nombrados + 24 usuarios sintéticos variados para robustez de SVD y Precisión@K
    perfiles = list(arquetipos)
    clusters_extra = [
        (["ciencia ficcion", "misterio", "tecnologia", "espacio"], ["romance"]),
        (["relajado", "cafe", "naturaleza", "cozy", "jazz"], ["terror"]),
        (["comedia", "divertido", "amigos", "fiesta", "animacion"], ["oscuro"]),
        (["comida", "cocina", "italiana", "asiatica", "viajes"], ["sustos"]),
    ]
    for n_u in range(1, 25):
        likes, dislikes = clusters_extra[n_u % len(clusters_extra)]
        perfiles.append((f"Explorador_{n_u:02d}", likes, dislikes))

    for usuario, likes, dislikes in perfiles:
        # Cada usuario valora ~35% del catálogo con coherencia semántica + ruido real
        elegidos = rng.choice(items_df["id"].to_numpy(), size=max(12, int(len(items_df) * 0.36)), replace=False)
        for item_id in elegidos:
            fila = items_df[items_df["id"] == item_id].iloc[0]
            texto = str(fila["texto"]).lower()
            hits_pos = sum(1 for k in likes if k in texto)
            hits_neg = sum(1 for k in dislikes if k in texto)
            if hits_neg > 0:
                base = 1.6
            elif hits_pos >= 2:
                base = 4.7
            elif hits_pos == 1:
                base = 4.1
            else:
                base = 2.6
            calif = int(np.clip(round(base + rng.normal(0, 0.55)), 1, 5))
            filas.append({"usuario": usuario, "item": int(item_id), "calificacion": calif})

    return pd.DataFrame(filas)


def inicializar_datos(carpeta_datos: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Crea o carga los CSV maestros en data/ manteniendo compatibilidad con el formato del Colab."""
    os.makedirs(carpeta_datos, exist_ok=True)
    ruta_items = os.path.join(carpeta_datos, "items.csv")
    ruta_inter = os.path.join(carpeta_datos, "interacciones.csv")

    if os.path.exists(ruta_items) and os.path.exists(ruta_inter):
        items_df = pd.read_csv(ruta_items)
        inter_df = pd.read_csv(ruta_inter)
        return items_df, inter_df

    items_df = pd.DataFrame(obtener_catalogo_base())
    inter_df = generar_interacciones_iniciales(items_df)
    items_df.to_csv(ruta_items, index=False)
    inter_df.to_csv(ruta_inter, index=False)

    # También exporta carpetas separadas por dominio (formato exacto del Colab)
    for dom in DOMINIOS_META:
        if dom == "todo":
            continue
        sub_dir = os.path.join(carpeta_datos, dom)
        os.makedirs(sub_dir, exist_ok=True)
        sub_items = items_df[items_df["categoria"] == dom]
        sub_inter = inter_df[inter_df["item"].isin(sub_items["id"])]
        sub_items.to_csv(os.path.join(sub_dir, "items.csv"), index=False)
        sub_inter.to_csv(os.path.join(sub_dir, "interacciones.csv"), index=False)

    return items_df, inter_df

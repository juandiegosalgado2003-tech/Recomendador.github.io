# Criterio — Sistema de Recomendación Inteligente

Aplicación web moderna y completa para la recomendación multi-dominio basada en afinidad de gustos e inteligencia artificial local, evolucionada a partir de los fundamentos de aprendizaje automático de `Recomendador_general (1).ipynb`.

El sistema opera **100% en local** (sin requerir claves API externas ni servicios de terceros), utilizando procesamiento de lenguaje natural (NLP) con TF-IDF, filtrado colaborativo con descomposición en valores singulares (TruncatedSVD) y agrupamiento temático con K-Means.

---

## Estructura del Proyecto

El código está estrictamente organizado y depurado; contiene únicamente los archivos y módulos necesarios para el funcionamiento del recomendador:

```text
Recomendador/
├── backend/
│   ├── __init__.py
│   ├── catalogos.py     # Catálogo base de 98 obras en 8 categorías y generador de datos
│   ├── main.py          # Servidor FastAPI y endpoints de la API REST
│   ├── motor.py         # Motor de recomendación (TF-IDF, TruncatedSVD, K-Means, Precisión@K)
│   └── storage.py       # Persistencia local de perfiles e interacciones en disco
├── data/
│   ├── items.csv        # Catálogo maestro de obras culturales y de ocio
│   ├── interacciones.csv# Historial de calificaciones de usuarios
│   └── perfiles.json    # Perfiles guardados con preferencias del cuestionario
├── frontend/
│   ├── index.html       # Interfaz limpia de 3 vistas (Recomendador, Catálogo, Perfil)
│   ├── styles.css       # Sistema de diseño sobrio, oscuro y responsivo (sin emojis)
│   ├── app.js           # Lógica cliente modular para el flujo interactivo
│   └── questions.js     # Banco de preguntas y opciones realistas por categoría
├── tests/
│   └── test_motor_api.py# Pruebas automatizadas (compatibilidad Colab y endpoints API)
├── pyproject.toml       # Configuración del proyecto y dependencias
├── requirements.txt     # Lista de paquetes pip
├── run.bat              # Script de inicio rápido para Windows (doble clic)
└── run.ps1              # Script de arranque en PowerShell
```

---

## Funcionalidades Principales

### 1. Recomendador Interactivo por Cuestionario
- Guía al usuario a través de 6 pasos claros para conocer sus preferencias reales:
  1. **Categoría**: Cine, Series, Literatura, Música, Videojuegos, Gastronomía, Viajes, Actividades o Todo.
  2. **Referentes**: Obras o experiencias de cabecera con opción de añadir referencias personalizadas.
  3. **Cualidades**: Selección de atmósfera (giros de misterio, estética visual, calma sensorial, ritmo ágil, etc.).
  4. **Formato**: Filtros de duración, tipo de compañía (solitario, pareja, amigos, familia) y presupuesto.
  5. **Exclusiones**: Filtros negativos expresos (descartar terror, relatos depresivos, lentitud, etc.).
  6. **Afinado final**: Explicación en lenguaje natural y nombre del perfil de usuario.
- Entrega recomendaciones ordenadas por porcentaje de afinidad IA (% Afinidad) con motivos transparentes de la sugerencia.

### 2. Explorador de Catálogo y Colecciones Temáticas
- **Clustering K-Means integrado**: Agrupa orgánicamente las obras en grupos temáticos y extrae palabras clave representativas para facilitar la navegación y el descubrimiento semántico.
- Buscador semántico con procesamiento de lenguaje natural (NLP) para consultas libres.
- Filtros rápidos por categoría y fichas detalladas con sugerencias de obras similares.

### 3. Mi Perfil y Calibración
- Historial de calificaciones por estrellas (1 a 5).
- Recalibración del motor según los algoritmos:
  - **Híbrido** (Contenido + Colaborativo)
  - **Contenido** (TF-IDF y atributos)
  - **Colaborativo** (SVD de interacciones)
  - **Clustering** (Agrupamiento temático)
  - **Popularidad** (Obras destacadas)

---

## Cómo Ejecutar en Windows

### Opción 1: Inicio rápido (Recomendado)
Haz doble clic sobre el archivo **`run.bat`** o ejecuta en PowerShell:
```powershell
.\run.ps1
```

### Opción 2: Con el entorno virtual Python
```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Luego abre en tu navegador web:
```text
http://127.0.0.1:8000
```

---

## Pruebas Automatizadas

Para validar la integridad del motor y la compatibilidad con el Colab:
```powershell
.\.venv\Scripts\pytest.exe -v
```


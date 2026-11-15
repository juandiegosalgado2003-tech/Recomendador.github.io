import json
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent

# 1. Cargar items del CSV
df_items = pd.read_csv(ROOT / "data" / "items.csv")
for col in ["etiquetas", "tono", "presupuesto", "duracion", "compania", "descripcion", "texto"]:
    df_items[col] = df_items[col].fillna("")
items_list = df_items.to_dict(orient="records")
for it in items_list:
    if isinstance(it["etiquetas"], str):
        it["etiquetas"] = [t.strip() for t in it["etiquetas"].split(",") if t.strip()]

items_json = json.dumps(items_list, ensure_ascii=False, indent=2)

# 2. Cargar CSS
css_content = (ROOT / "frontend" / "styles.css").read_text(encoding="utf-8")

# 3. Cargar questions.js y extraer constantes
questions_js = (ROOT / "frontend" / "questions.js").read_text(encoding="utf-8")
questions_clean = re.sub(r"export\s+const\s+", "const ", questions_js)

# 4. Cargar index.html original y extraer el body interno
html_raw = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

match_body = re.search(r"<body[^>]*>(.*?)</body>", html_raw, re.DOTALL)
if match_body:
    body_content = match_body.group(1)
    body_content = re.sub(r'<script\s+type="module"[^>]*></script>', '', body_content)
else:
    body_content = ""

# 5. Código cliente unificado en string normal con placeholders
js_template = """
// ============================================================================
// DATOS INCORPORADOS (CATÁLOGO Y PREGUNTAS)
// ============================================================================
const ITEMS_CATALOGO = __ITEMS_JSON__;

__QUESTIONS_CODE__

const DOMINIOS_META = {
  todo: { id: "todo", nombre: "Todas las categorías", icono: "all", total: ITEMS_CATALOGO.length },
  peliculas: { id: "peliculas", nombre: "Cine y Películas", icono: "film", total: ITEMS_CATALOGO.filter(i => i.categoria === "peliculas").length },
  series: { id: "series", nombre: "Series de Televisión", icono: "tv", total: ITEMS_CATALOGO.filter(i => i.categoria === "series").length },
  libros: { id: "libros", nombre: "Literatura y Libros", icono: "book", total: ITEMS_CATALOGO.filter(i => i.categoria === "libros").length },
  musica: { id: "musica", nombre: "Música y Álbumes", icono: "music", total: ITEMS_CATALOGO.filter(i => i.categoria === "musica").length },
  videojuegos: { id: "videojuegos", nombre: "Videojuegos", icono: "gamepad", total: ITEMS_CATALOGO.filter(i => i.categoria === "videojuegos").length },
  comida: { id: "comida", nombre: "Gastronomía y Cocina", icono: "utensils", total: ITEMS_CATALOGO.filter(i => i.categoria === "comida").length },
  viajes: { id: "viajes", nombre: "Viajes y Rutas", icono: "compass", total: ITEMS_CATALOGO.filter(i => i.categoria === "viajes").length },
  hobbies_regalos: { id: "hobbies_regalos", nombre: "Actividades y Aficiones", icono: "gift", total: ITEMS_CATALOGO.filter(i => i.categoria === "hobbies_regalos").length },
};

const STOPWORDS_ES = new Set([
  "a", "al", "algo", "algunas", "algunos", "ante", "como", "con", "contra",
  "cual", "cuando", "de", "del", "desde", "donde", "dos", "el", "ella",
  "ellas", "ellos", "en", "entre", "era", "es", "esa", "esas", "ese",
  "eso", "esos", "esta", "estaba", "estado", "estan", "estar", "estas",
  "este", "esto", "estos", "fue", "fueron", "ha", "hace", "hacer", "hacia",
  "hasta", "hay", "la", "las", "le", "les", "lo", "los", "mas", "me", "mi",
  "mis", "mucha", "muchas", "mucho", "muchos", "muy", "nada", "ni", "no",
  "nos", "nosotros", "o", "os", "otra", "otras", "otro", "otros", "para",
  "pero", "poco", "por", "porque", "que", "quien", "quienes", "se", "sea",
  "ser", "si", "sin", "sobre", "sois", "solamente", "solo", "somos", "son",
  "soy", "su", "sus", "tambien", "te", "tengo", "ti", "tiene", "tienen",
  "todo", "todos", "tu", "tus", "un", "una", "unas", "uno", "unos", "usted",
  "va", "vamos", "van", "ya", "yo"
]);

function normalizarTexto(txt) {
  if (!txt) return [];
  const sinTildes = String(txt).normalize("NFKD").replace(/[\u0300-\u036f]/g, "").toLowerCase();
  const tokens = sinTildes.match(/[a-z0-9_]+/g) || [];
  return tokens.filter(t => t.length > 1 && !STOPWORDS_ES.has(t));
}

// ============================================================================
// MOTOR DE IA LOCAL (TF-IDF + COSINE SIMILARITY + K-MEANS CLUSTERING)
// ============================================================================
class MotorRecomendacionLocal {
  constructor(items) {
    this.items = items;
    this.vocab = new Map();
    this.terms = [];
    this.df = new Map();
    this.N = items.length;
    this.idf = [];
    this.itemVectors = [];
    this.clusters = [];
    this._construirIndice();
    this._calcularClusters();
  }

  _construirIndice() {
    const docsTokens = this.items.map(it => {
      const fullText = `${it.nombre} ${it.texto} ${it.categoria} ${it.subcategoria} ${(it.etiquetas || []).join(" ")} ${it.tono} ${it.descripcion}`;
      const tokens = normalizarTexto(fullText);
      const unique = new Set(tokens);
      for (const t of unique) {
        this.df.set(t, (this.df.get(t) || 0) + 1);
      }
      return tokens;
    });

    let idx = 0;
    for (const [t, count] of this.df.entries()) {
      this.vocab.set(t, idx);
      this.terms.push(t);
      this.idf[idx] = Math.log((this.N + 1) / (count + 1)) + 1;
      idx++;
    }

    const V = this.terms.length;

    this.itemVectors = docsTokens.map(tokens => {
      const vec = new Float64Array(V);
      const tfMap = new Map();
      for (const t of tokens) {
        tfMap.set(t, (tfMap.get(t) || 0) + 1);
      }
      let sumSq = 0;
      for (const [t, freq] of tfMap.entries()) {
        const tidx = this.vocab.get(t);
        if (tidx !== undefined) {
          const tf = 1 + Math.log(freq);
          const weight = tf * this.idf[tidx];
          vec[tidx] = weight;
          sumSq += weight * weight;
        }
      }
      const norm = Math.sqrt(sumSq) || 1;
      for (let i = 0; i < V; i++) {
        vec[i] /= norm;
      }
      return vec;
    });
  }

  vectorizar(tokens) {
    const V = this.terms.length;
    const vec = new Float64Array(V);
    const tfMap = new Map();
    for (const t of tokens) {
      tfMap.set(t, (tfMap.get(t) || 0) + 1);
    }
    let sumSq = 0;
    for (const [t, freq] of tfMap.entries()) {
      const tidx = this.vocab.get(t);
      if (tidx !== undefined) {
        const tf = 1 + Math.log(freq);
        const weight = tf * this.idf[tidx];
        vec[tidx] = weight;
        sumSq += weight * weight;
      }
    }
    const norm = Math.sqrt(sumSq) || 1;
    for (let i = 0; i < V; i++) {
      vec[i] /= norm;
    }
    return vec;
  }

  similitud(v1, v2) {
    let dot = 0;
    for (let i = 0; i < v1.length; i++) {
      dot += v1[i] * v2[i];
    }
    return dot;
  }

  _calcularClusters() {
    const catMap = new Map();
    this.items.forEach((it, idx) => {
      if (!catMap.has(it.categoria)) catMap.set(it.categoria, []);
      catMap.get(it.categoria).push(idx);
    });

    const clustersList = [];
    const V = this.terms.length;
    let cId = 0;

    for (const [cat, indices] of catMap.entries()) {
      const centroid = new Float64Array(V);
      indices.forEach(idx => {
        const vec = this.itemVectors[idx];
        for (let i = 0; i < V; i++) centroid[i] += vec[i];
      });
      for (let i = 0; i < V; i++) centroid[i] /= indices.length;

      const termWeights = [];
      for (let i = 0; i < V; i++) {
        if (centroid[i] > 0) termWeights.push({ term: this.terms[i], weight: centroid[i] });
      }
      termWeights.sort((a, b) => b.weight - a.weight);
      const topKeywords = termWeights.slice(0, 3).map(tw => tw.term);

      clustersList.push({
        id: cId++,
        categoria: cat,
        nombre: DOMINIOS_META[cat]?.nombre || cat,
        palabras_clave: topKeywords,
        indices: indices,
      });
    }
    this.clusters = clustersList;
  }

  buscar(query, categoria = "todo", limit = 32) {
    let candidatos = this.items.map((it, idx) => ({ it, idx }));
    if (categoria !== "todo") {
      candidatos = candidatos.filter(c => c.it.categoria === categoria);
    }
    if (!query || !query.trim()) {
      return candidatos.slice(0, limit).map(c => c.it);
    }
    const qTokens = normalizarTexto(query);
    const qVec = this.vectorizar(qTokens);

    const scored = candidatos.map(c => ({
      item: c.it,
      sim: this.similitud(qVec, this.itemVectors[c.idx]),
    }));
    scored.sort((a, b) => b.sim - a.sim);
    return scored.slice(0, limit).map(s => ({ ...s.item, score: s.sim }));
  }

  similares(itemId, n = 4) {
    const idx = this.items.findIndex(it => String(it.id) === String(itemId));
    if (idx === -1) return [];
    const baseVec = this.itemVectors[idx];
    const scored = [];
    for (let i = 0; i < this.items.length; i++) {
      if (i === idx) continue;
      const sim = this.similitud(baseVec, this.itemVectors[i]);
      scored.push({ item: this.items[i], sim });
    }
    scored.sort((a, b) => b.sim - a.sim);
    return scored.slice(0, n).map(s => s.item);
  }

  recomendarPorCuestionario(params) {
    const {
      categoria = "todo",
      referencias = [],
      atributos = [],
      duracion = "indiferente",
      compania = "indiferente",
      presupuesto = "indiferente",
      evitar = [],
      texto_libre = "",
      cantidad = 6,
    } = params;

    const consultaTokens = [
      ...normalizarTexto(referencias.join(" ")),
      ...normalizarTexto(atributos.join(" ")),
      ...normalizarTexto(texto_libre),
    ];

    const qVec = this.vectorizar(consultaTokens);
    const evitarTokens = new Set(normalizarTexto(evitar.join(" ")));

    let candidatos = this.items.map((it, idx) => ({ it, idx }));
    if (categoria !== "todo") {
      candidatos = candidatos.filter(c => c.it.categoria === categoria);
    }

    const scored = [];
    for (const c of candidatos) {
      const it = c.it;
      const itemTokens = new Set(normalizarTexto(`${it.nombre} ${it.texto} ${it.categoria} ${(it.etiquetas || []).join(" ")} ${it.tono}`));
      
      let descartado = false;
      for (const ev of evitarTokens) {
        if (itemTokens.has(ev)) {
          descartado = true;
          break;
        }
      }
      if (descartado) continue;

      let sim = this.similitud(qVec, this.itemVectors[c.idx]);
      let bonus = 0;
      if (duracion !== "indiferente" && it.duracion === duracion) bonus += 0.08;
      if (compania !== "indiferente" && it.compania === compania) bonus += 0.08;
      if (presupuesto !== "indiferente" && it.presupuesto === presupuesto) bonus += 0.08;

      const scoreTotal = sim + bonus;

      const razones = [];
      if (it.duracion === duracion && duracion !== "indiferente") {
        razones.push(`Formato adaptado a tu tiempo disponible (${duracion})`);
      }
      if (it.compania === compania && compania !== "indiferente") {
        razones.push(`Ideal para disfrutar en plan ${compania}`);
      }
      if (atributos.length) {
        razones.push(`Alineado con las cualidades seleccionadas`);
      }
      if (razones.length === 0) {
        razones.push(`Afinidad con tus gustos culturales y referentes`);
      }

      scored.push({ ...it, scoreTotal, razones });
    }

    scored.sort((a, b) => b.scoreTotal - a.scoreTotal);

    let seleccionados = [];
    if (categoria === "todo") {
      const porCat = new Map();
      for (const s of scored) {
        if (!porCat.has(s.categoria)) porCat.set(s.categoria, []);
        porCat.get(s.categoria).push(s);
      }
      const cats = Array.from(porCat.keys());
      let idxCat = 0;
      while (seleccionados.length < cantidad && porCat.size > 0) {
        const c = cats[idxCat % cats.length];
        const list = porCat.get(c);
        if (list && list.length > 0) {
          seleccionados.push(list.shift());
        } else {
          porCat.delete(c);
        }
        idxCat++;
      }
    } else {
      seleccionados = scored.slice(0, cantidad);
    }

    const maxScore = seleccionados[0]?.scoreTotal || 1;
    return seleccionados.map((item, idx) => {
      const ratio = item.scoreTotal / (maxScore || 1);
      const afinidad = Math.min(99, Math.max(76, Math.round(98 - idx * 3 + (ratio - 0.5) * 4)));
      return { ...item, afinidad };
    });
  }
}

const motorLocal = new MotorRecomendacionLocal(ITEMS_CATALOGO);

// ============================================================================
// ICONOS SVG
// ============================================================================
const ICON_PATHS = {
  all: "M4 6h16M4 12h16M4 18h16",
  film: "M4 5h16a1 1 0 0 1 1 1v12a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1zm3 0v14M17 5v14M3 9h4m10 0h4M3 15h4m10 0h4",
  tv: "M3 7h18v12H3V7zm5-4l4 4 4-4",
  book: "M4 5.5A2.5 2.5 0 0 1 6.5 3H20v17H6.5A2.5 2.5 0 0 0 4 22V5.5zM4 18.5A2.5 2.5 0 0 1 6.5 16H20",
  music: "M9 18V5l11-2v13M9 18a2.5 2.5 0 1 1-5 0 2.5 2.5 0 0 1 5 0zm11-2a2.5 2.5 0 1 1-5 0 2.5 2.5 0 0 1 5 0z",
  gamepad: "M6 11h4m-2-2v4m7-1h.01M18 10h.01M5 7h14a3 3 0 0 1 3 3l-1 6a2.5 2.5 0 0 1-4.3 1.4L14.5 15h-5l-2.2 2.4A2.5 2.5 0 0 1 3 16l-1-6a3 3 0 0 1 3-3z",
  utensils: "M4 3v7a3 3 0 0 0 3 3v8M7 3v6M10 3v7a3 3 0 0 1-3 3M17 3v18m0-18c2 2 2 6 0 8",
  compass: "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20zm3.5-13.5l-2 5-5 2 2-5 5-2z",
  gift: "M20 12v9H4v-9m18-4H2v4h20V8zM12 8v13m0-13c-1.5-3-5-3-5-1s3.5 1 5 1zm0 0c1.5-3 5-3 5-1s-3.5 1-5 1z",
};

const STEP_METAS = [
  { step: 1, label: "Paso 1 de 6", title: "¿Qué te apetece explorar hoy?" },
  { step: 2, label: "Paso 2 de 6", title: "Elige tus obras o referentes de cabecera" },
  { step: 3, label: "Paso 3 de 6", title: "¿Qué cualidades o sensaciones buscas?" },
  { step: 4, label: "Paso 4 de 6", title: "¿En qué plan estás y cuánto tiempo tienes?" },
  { step: 5, label: "Paso 5 de 6", title: "¿Qué prefieres descartar expresamente?" },
  { step: 6, label: "Paso 6 de 6", title: "Detalles finales y tu nombre de perfil" },
];

function cargarRatingsStorage() {
  try {
    const raw = localStorage.getItem("criterio_ratings");
    return raw ? new Map(JSON.parse(raw)) : new Map();
  } catch {
    return new Map();
  }
}

function guardarRatingsStorage(map) {
  try {
    localStorage.setItem("criterio_ratings", JSON.stringify(Array.from(map.entries())));
  } catch {
    /* ignore */
  }
}

const state = {
  activeView: "wizard",
  activeUser: localStorage.getItem("criterio_username") || "Lector_Curioso",
  ratingsMap: cargarRatingsStorage(),
  allCatalogItems: ITEMS_CATALOGO,
  selectedClusterId: null,
  wizard: {
    step: 1,
    maxReached: 1,
    categoria: "todo",
    referencias: [],
    referencia_libre: "",
    atributos: [],
    duracion: "indiferente",
    compania: "indiferente",
    presupuesto: "indiferente",
    evitar: [],
    texto_libre: "",
    usuario: localStorage.getItem("criterio_username") || "Lector_Curioso",
    guardar_perfil: true,
    cantidad: 6,
    resultados: [],
    explicacion: "",
  },
  explore: {
    categoria: "todo",
    clusterId: null,
    q: "",
  },
};

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "className") node.className = v;
    else if (k === "textContent") node.textContent = v;
    else if (k === "dataset") Object.assign(node.dataset, v);
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2).toLowerCase(), v);
    else if (v !== undefined && v !== null && v !== false) node.setAttribute(k, v === true ? "" : String(v));
  }
  const list = Array.isArray(children) ? children : [children];
  for (const child of list) {
    if (child === null || child === undefined || child === false) continue;
    node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return node;
}

function createIcon(name) {
  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "1.8");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  const path = document.createElementNS(svgNS, "path");
  path.setAttribute("d", ICON_PATHS[name] || ICON_PATHS.all);
  svg.appendChild(path);
  return svg;
}

function showToast(msg) {
  const dock = document.getElementById("toast-dock");
  if (!dock) return;
  const toast = el("div", { className: "toast", textContent: msg });
  dock.appendChild(toast);
  setTimeout(() => toast.remove(), 3000);
}

function domainName(catId) {
  if (catId === "todo") return "Todas las categorías";
  const found = DOMINIOS_META[catId];
  return found ? found.nombre : catId;
}

function switchView(viewName) {
  state.activeView = viewName;
  document.querySelectorAll(".nav-link").forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.view === viewName);
  });
  document.querySelectorAll(".view").forEach((sec) => {
    sec.hidden = sec.id !== `view-${viewName}`;
  });

  if (viewName === "explorar") {
    cargarCatalogoExplorador();
  } else if (viewName === "perfil") {
    cargarVistaPerfil();
  }
}

// ============================================================================
// STEPPER & CUESTIONARIO
// ============================================================================
function updateStepperUI() {
  const dotsContainer = document.getElementById("step-dots");
  const labelEl = document.getElementById("step-label");
  const progressFill = document.getElementById("stepper-progress");
  if (!dotsContainer) return;

  dotsContainer.innerHTML = "";
  STEP_METAS.forEach((meta) => {
    const isActive = meta.step === state.wizard.step;
    const isDone = meta.step < state.wizard.step;
    const isClickable = meta.step <= state.wizard.maxReached;

    const dot = el("button", {
      type: "button",
      className: `step-dot ${isActive ? "is-active" : ""} ${isDone ? "is-completed" : ""}`,
      textContent: String(meta.step),
      disabled: !isClickable,
      onClick: () => {
        if (isClickable) {
          state.wizard.step = meta.step;
          renderStage();
        }
      },
    });
    dotsContainer.appendChild(dot);
  });

  if (labelEl) labelEl.textContent = STEP_METAS[state.wizard.step - 1].label;
  if (progressFill) {
    const pct = ((state.wizard.step - 1) / 5) * 100;
    progressFill.style.width = `${Math.max(16.66, pct)}%`;
  }

  const prevBtn = document.getElementById("btn-prev");
  const nextBtn = document.getElementById("btn-next");
  if (prevBtn) prevBtn.disabled = state.wizard.step === 1;
  if (nextBtn) {
    nextBtn.textContent = state.wizard.step === 6 ? "Obtener recomendaciones" : "Siguiente";
  }
}

function renderStage() {
  updateStepperUI();
  const stage = document.getElementById("wizard-stage");
  if (!stage) return;
  stage.innerHTML = "";

  const meta = STEP_METAS[state.wizard.step - 1];
  stage.appendChild(el("h2", { className: "stage-title", textContent: meta.title }));

  switch (state.wizard.step) {
    case 1: renderStep1(stage); break;
    case 2: renderStep2(stage); break;
    case 3: renderStep3(stage); break;
    case 4: renderStep4(stage); break;
    case 5: renderStep5(stage); break;
    case 6: renderStep6(stage); break;
  }
}

function renderStep1(container) {
  const grid = el("div", { className: "category-grid" });

  const allOption = el(
    "button",
    {
      type: "button",
      className: `category-card ${state.wizard.categoria === "todo" ? "is-selected" : ""}`,
      onClick: () => {
        state.wizard.categoria = "todo";
        renderStep1(container);
      },
    },
    [
      el("div", { className: "category-card-icon" }, [createIcon("all")]),
      el("div", { className: "category-card-info" }, [
        el("strong", { textContent: "Todas las categorías" }),
        el("span", { textContent: "Recomiéndame de todo un poco" }),
      ]),
    ]
  );
  grid.appendChild(allOption);

  Object.values(DOMINIOS_META).forEach((dom) => {
    if (dom.id === "todo") return;
    const isSelected = state.wizard.categoria === dom.id;
    const card = el(
      "button",
      {
        type: "button",
        className: `category-card ${isSelected ? "is-selected" : ""}`,
        onClick: () => {
          state.wizard.categoria = dom.id;
          renderStep1(container);
        },
      },
      [
        el("div", { className: "category-card-icon" }, [createIcon(dom.icono)]),
        el("div", { className: "category-card-info" }, [
          el("strong", { textContent: dom.nombre }),
          el("span", { textContent: `${dom.total} obras disponibles` }),
        ]),
      ]
    );
    grid.appendChild(card);
  });

  container.innerHTML = "";
  container.appendChild(el("h2", { className: "stage-title", textContent: STEP_METAS[0].title }));
  container.appendChild(grid);
}

function renderStep2(container) {
  const cat = state.wizard.categoria;
  const list = REFERENCIAS_POPULARES[cat] || REFERENCIAS_POPULARES.todo || [];

  const grid = el("div", { className: "references-grid" });
  list.forEach((ref) => {
    const isSelected = state.wizard.referencias.includes(ref.tokens);
    const card = el(
      "button",
      {
        type: "button",
        className: `reference-card ${isSelected ? "is-selected" : ""}`,
        onClick: (e) => {
          const btn = e.currentTarget;
          if (state.wizard.referencias.includes(ref.tokens)) {
            state.wizard.referencias = state.wizard.referencias.filter((t) => t !== ref.tokens);
            btn.classList.remove("is-selected");
          } else {
            state.wizard.referencias.push(ref.tokens);
            btn.classList.add("is-selected");
          }
        },
      },
      [el("strong", { textContent: ref.label })]
    );
    grid.appendChild(card);
  });

  const customBox = el("div", { className: "custom-input-box" }, [
    el("label", { textContent: "O escribe otro referente o creador que te guste:" }),
    el("input", {
      type: "text",
      className: "input-text",
      placeholder: "Ej: Blade Runner, Studio Ghibli, Murakami, pasta cacio e pepe...",
      value: state.wizard.referencia_libre,
      onInput: (e) => {
        state.wizard.referencia_libre = e.target.value;
      },
    }),
  ]);

  container.appendChild(grid);
  container.appendChild(customBox);
}

function renderStep3(container) {
  const grid = el("div", { className: "attributes-grid" });
  ATRIBUTOS_CALIDAD.forEach((attr) => {
    const isSelected = state.wizard.atributos.includes(attr.tokens);
    const card = el(
      "button",
      {
        type: "button",
        className: `attribute-card ${isSelected ? "is-selected" : ""}`,
        onClick: (e) => {
          const btn = e.currentTarget;
          if (state.wizard.atributos.includes(attr.tokens)) {
            state.wizard.atributos = state.wizard.atributos.filter((t) => t !== attr.tokens);
            btn.classList.remove("is-selected");
          } else {
            state.wizard.atributos.push(attr.tokens);
            btn.classList.add("is-selected");
          }
        },
      },
      [
        el("h4", { textContent: attr.label }),
        el("p", { textContent: attr.desc }),
      ]
    );
    grid.appendChild(card);
  });
  container.appendChild(grid);
}

function renderStep4(container) {
  const stack = el("div", { className: "format-stack" });

  function makeGroup(title, options, currentVal, onChange) {
    const group = el("div", { className: "format-group" }, [
      el("label", { textContent: title }),
      el("div", { className: "chips-row" }, options.map((opt) => {
        const isSel = currentVal === opt.id;
        return el("button", {
          type: "button",
          className: `choice-chip ${isSel ? "is-selected" : ""}`,
          textContent: opt.label,
          onClick: (e) => {
            const row = e.currentTarget.parentElement;
            row.querySelectorAll(".choice-chip").forEach((c) => c.classList.remove("is-selected"));
            e.currentTarget.classList.add("is-selected");
            onChange(opt.id);
          },
        });
      })),
    ]);
    return group;
  }

  stack.appendChild(makeGroup("Duración / Extensión", FORMATO_DURACION, state.wizard.duracion, (v) => { state.wizard.duracion = v; }));
  stack.appendChild(makeGroup("Compañía", FORMATO_COMPANIA, state.wizard.compania, (v) => { state.wizard.compania = v; }));
  stack.appendChild(makeGroup("Presupuesto / Acceso", FORMATO_PRESUPUESTO, state.wizard.presupuesto, (v) => { state.wizard.presupuesto = v; }));

  container.appendChild(stack);
}

function renderStep5(container) {
  const grid = el("div", { className: "exclusions-grid" });
  CRITERIOS_EXCLUSION.forEach((exc) => {
    const isSelected = state.wizard.evitar.includes(exc.tokens);
    const card = el(
      "button",
      {
        type: "button",
        className: `exclusion-card ${isSelected ? "is-selected" : ""}`,
        onClick: (e) => {
          const btn = e.currentTarget;
          if (state.wizard.evitar.includes(exc.tokens)) {
            state.wizard.evitar = state.wizard.evitar.filter((t) => t !== exc.tokens);
            btn.classList.remove("is-selected");
          } else {
            state.wizard.evitar.push(exc.tokens);
            btn.classList.add("is-selected");
          }
        },
      },
      [
        el("div", { className: "exclusion-indicator" }),
        el("span", { textContent: exc.label }),
      ]
    );
    grid.appendChild(card);
  });
  container.appendChild(grid);
}

function renderStep6(container) {
  const layout = el("div", { className: "step-final-layout" });

  const textareaBox = el("div", {}, [
    el("label", {
      style: "display: block; font-weight: 600; font-size: 0.9rem; margin-bottom: 0.4rem; color: var(--text-secondary);",
      textContent: "Describe libremente con tus palabras lo que buscas:",
    }),
    el("textarea", {
      id: "free-text-input",
      className: "textarea-input",
      placeholder: "Ej: Me gustan las historias de ciencia ficción con rigor técnico pero centradas en las personas...",
      value: state.wizard.texto_libre,
      onInput: (e) => { state.wizard.texto_libre = e.target.value; },
    }),
  ]);

  const suggestionsBox = el("div", { className: "prompt-suggestions" }, [
    el("span", {
      style: "font-size: 0.78rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em;",
      textContent: "Ideas de búsqueda:",
    }),
    el("div", { style: "display: flex; flex-direction: column; gap: 0.35rem; margin-top: 0.4rem;" },
      EJEMPLOS_TEXTO_LIBRE.map((prompt) =>
        el("button", {
          type: "button",
          className: "prompt-btn",
          textContent: prompt,
          onClick: () => {
            state.wizard.texto_libre = prompt;
            const ta = document.getElementById("free-text-input");
            if (ta) ta.value = prompt;
          },
        })
      )
    ),
  ]);

  const profileBox = el("div", { className: "profile-box" }, [
    el("label", {
      style: "font-weight: 600; font-size: 0.88rem; color: var(--text-secondary);",
      textContent: "Nombre de perfil para registrar tus preferencias:",
    }),
    el("input", {
      type: "text",
      className: "input-text",
      value: state.wizard.usuario,
      onInput: (e) => {
        state.wizard.usuario = e.target.value;
        localStorage.setItem("criterio_username", e.target.value);
      },
    }),
  ]);

  layout.appendChild(textareaBox);
  layout.appendChild(suggestionsBox);
  layout.appendChild(profileBox);
  container.appendChild(layout);
}

function handleNextStep() {
  if (state.wizard.step < 6) {
    state.wizard.step++;
    state.wizard.maxReached = Math.max(state.wizard.maxReached, state.wizard.step);
    renderStage();
  } else {
    ejecutarRecomendacionWizard();
  }
}

function handlePrevStep() {
  if (state.wizard.step > 1) {
    state.wizard.step--;
    renderStage();
  }
}

// ============================================================================
// RESULTADOS DEL CUESTIONARIO CON IA
// ============================================================================
async function ejecutarRecomendacionWizard() {
  const nextBtn = document.getElementById("btn-next");
  if (nextBtn) {
    nextBtn.disabled = true;
    nextBtn.textContent = "Analizando afinidad con IA...";
  }

  let referenciasList = [...state.wizard.referencias];
  if (state.wizard.referencia_libre.trim()) {
    referenciasList.push(state.wizard.referencia_libre.trim());
  }

  const payload = {
    categoria: state.wizard.categoria,
    referencias: referenciasList,
    atributos: state.wizard.atributos,
    duracion: state.wizard.duracion,
    compania: state.wizard.compania,
    presupuesto: state.wizard.presupuesto,
    evitar: state.wizard.evitar,
    texto_libre: state.wizard.texto_libre,
    usuario: state.wizard.usuario || "Lector_Curioso",
    cantidad: 6,
  };

  try {
    const res = await fetch("/api/cuestionario/recomendar", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (res.ok) {
      const data = await res.json();
      state.wizard.resultados = data.recomendaciones || [];
    } else {
      throw new Error("Backend offline");
    }
  } catch {
    state.wizard.resultados = motorLocal.recomendarPorCuestionario(payload);
  }

  const catNombre = domainName(state.wizard.categoria);
  const userNom = state.wizard.usuario || "tu perfil";
  state.wizard.explicacion = "Selección generada por IA para " + userNom + " en " + catNombre + ". Se evaluaron 98 obras considerando tus referentes, tono y exclusiones.";

  if (nextBtn) {
    nextBtn.disabled = false;
    nextBtn.textContent = "Obtener recomendaciones";
  }

  mostrarResultadosWizard();
}

function mostrarResultadosWizard() {
  const resultsContainer = document.getElementById("wizard-results");
  const grid = document.getElementById("results-grid");
  const explanationEl = document.getElementById("results-explanation");
  const filterRow = document.getElementById("results-filter-row");

  if (!resultsContainer || !grid) return;
  resultsContainer.hidden = false;

  if (explanationEl) {
    explanationEl.textContent = state.wizard.explicacion;
  }

  if (filterRow) {
    filterRow.innerHTML = "";
    filterRow.appendChild(el("span", { className: "pill-tag", textContent: domainName(state.wizard.categoria) }));
    if (state.wizard.referencias.length) {
      filterRow.appendChild(el("span", { className: "pill-tag", textContent: `${state.wizard.referencias.length} referentes` }));
    }
    if (state.wizard.atributos.length) {
      filterRow.appendChild(el("span", { className: "pill-tag", textContent: `${state.wizard.atributos.length} cualidades` }));
    }
  }

  grid.innerHTML = "";
  if (state.wizard.resultados.length === 0) {
    grid.appendChild(
      el("p", {
        style: "color: var(--text-muted); font-size: 0.92rem;",
        textContent: "No se encontraron obras con los criterios exactos. Prueba a relajar algún filtro.",
      })
    );
    return;
  }

  state.wizard.resultados.forEach((item) => {
    grid.appendChild(createItemCard(item));
  });

  resultsContainer.scrollIntoView({ behavior: "smooth", block: "start" });
}

function createItemCard(item) {
  const affinity = item.afinidad || Math.round((item.score || 0.85) * 100);
  const userScore = state.ratingsMap.get(item.id) || 0;
  const reasonText = (item.razones && item.razones[0]) || item.por_que || "Alta correspondencia con tus preferencias";

  const card = el("article", { className: "item-card" }, [
    el("div", {}, [
      el("div", { className: "item-card-top" }, [
        el("span", { className: "badge-pill", textContent: domainName(item.categoria) }),
        el("span", { className: "affinity-badge", textContent: `${affinity}% afinidad` }),
      ]),
      el("h3", { className: "item-card-title", textContent: item.nombre }),
      el("p", { className: "item-card-desc", textContent: item.descripcion || item.texto || "" }),
      el("p", { className: "match-reason", textContent: `Por qué encaja: ${reasonText}` }),
    ]),
    el("div", { className: "item-card-bottom" }, [
      el("div", { className: "rating-pills-row" }, [
        ...[1, 2, 3, 4, 5].map((val) =>
          el("button", {
            type: "button",
            className: `rate-btn ${userScore >= val ? "is-active" : ""}`,
            textContent: String(val),
            title: `Calificar ${val} de 5`,
            onClick: (e) => {
              e.stopPropagation();
              registrarCalificacion(item.id, val, card);
            },
          })
        ),
      ]),
      el(
        "button",
        {
          type: "button",
          className: "btn is-ghost",
          style: "padding: 0.35rem 0.85rem; font-size: 0.78rem;",
          onClick: () => abrirModal(item),
        },
        "Ver ficha"
      ),
    ]),
  ]);

  return card;
}

function registrarCalificacion(itemId, score, cardElement) {
  state.ratingsMap.set(itemId, score);
  guardarRatingsStorage(state.ratingsMap);

  if (cardElement) {
    const btns = cardElement.querySelectorAll(".rate-btn");
    btns.forEach((btn, idx) => {
      btn.classList.toggle("is-active", idx + 1 <= score);
    });
  }

  fetch("/api/interacciones", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      usuario: state.wizard.usuario || "Lector_Curioso",
      item_id: itemId,
      calificacion: score,
    }),
  }).catch(() => {});

  showToast(`Calificación ${score}/5 guardada en tu perfil`);
}

// ============================================================================
// MODAL DE DETALLE
// ============================================================================
function abrirModal(item) {
  const modal = document.getElementById("item-modal");
  if (!modal) return;

  document.getElementById("modal-category").textContent = domainName(item.categoria);
  document.getElementById("modal-item-title").textContent = item.nombre;
  document.getElementById("modal-desc").textContent = item.descripcion || item.texto || "";

  const tagsContainer = document.getElementById("modal-tags");
  tagsContainer.innerHTML = "";
  (item.etiquetas || []).forEach((t) => {
    tagsContainer.appendChild(el("span", { className: "pill-tag", textContent: t }));
  });

  const metaRow = document.getElementById("modal-meta");
  metaRow.innerHTML = "";
  const props = [
    { label: "Ámbito", val: domainName(item.categoria) },
    { label: "Extensión", val: item.duracion || "Estándar" },
    { label: "Compañía", val: item.compania || "Indiferente" },
  ];
  props.forEach((p) => {
    metaRow.appendChild(
      el("div", { className: "meta-unit" }, [
        el("span", { textContent: p.label }),
        el("strong", { textContent: p.val }),
      ])
    );
  });

  const ratingBox = document.getElementById("modal-rating-buttons");
  ratingBox.innerHTML = "";
  const curScore = state.ratingsMap.get(item.id) || 0;
  [1, 2, 3, 4, 5].forEach((val) => {
    ratingBox.appendChild(
      el("button", {
        type: "button",
        className: `rate-btn ${curScore >= val ? "is-active" : ""}`,
        textContent: String(val),
        onClick: () => {
          registrarCalificacion(item.id, val);
          abrirModal(item);
        },
      })
    );
  });

  const similarsGrid = document.getElementById("modal-similars-grid");
  similarsGrid.innerHTML = "";
  const similaresList = motorLocal.similares(item.id, 4);
  if (similaresList.length) {
    similaresList.forEach((sim) => {
      similarsGrid.appendChild(
        el("div", {
          className: "similar-card",
          onClick: () => abrirModal(sim),
        }, [
          el("strong", { textContent: sim.nombre }),
          el("span", { textContent: domainName(sim.categoria) }),
        ])
      );
    });
  } else {
    similarsGrid.appendChild(el("span", { style: "font-size: 0.8rem; color: var(--text-muted);", textContent: "No hay similares directos." }));
  }

  modal.showModal();
}

function cerrarModal() {
  const modal = document.getElementById("item-modal");
  if (modal) modal.close();
}

// ============================================================================
// EXPLORADOR Y CLUSTERING TEMÁTICO
// ============================================================================
function cargarCatalogoExplorador() {
  renderExploreClusters();
  renderExploreCategories();
  ejecutarBusquedaExplorar();
}

function renderExploreClusters() {
  const container = document.getElementById("explore-cluster-chips");
  if (!container) return;
  container.innerHTML = "";

  const allChip = el("button", {
    type: "button",
    className: `cluster-chip ${state.explore.clusterId === null ? "is-active" : ""}`,
    onClick: () => {
      state.explore.clusterId = null;
      renderExploreClusters();
      ejecutarBusquedaExplorar();
    },
  }, [
    el("strong", { textContent: "Todos los grupos" }),
    el("span", { textContent: "Catálogo completo" }),
  ]);
  container.appendChild(allChip);

  motorLocal.clusters.forEach((cluster) => {
    const isAct = state.explore.clusterId === cluster.id;
    const chip = el("button", {
      type: "button",
      className: `cluster-chip ${isAct ? "is-active" : ""}`,
      onClick: () => {
        state.explore.clusterId = isAct ? null : cluster.id;
        renderExploreClusters();
        ejecutarBusquedaExplorar();
      },
    }, [
      el("strong", { textContent: cluster.nombre }),
      el("span", { textContent: cluster.palabras_clave.join(" · ") }),
    ]);
    container.appendChild(chip);
  });
}

function renderExploreCategories() {
  const container = document.getElementById("explore-category-chips");
  if (!container) return;
  container.innerHTML = "";

  const cats = [
    { id: "todo", nombre: "Todas" },
    ...Object.values(DOMINIOS_META).filter(d => d.id !== "todo"),
  ];

  cats.forEach((cat) => {
    const isAct = state.explore.categoria === cat.id;
    const chip = el("button", {
      type: "button",
      className: `category-chip ${isAct ? "is-active" : ""}`,
      textContent: cat.nombre,
      onClick: () => {
        state.explore.categoria = cat.id;
        renderExploreCategories();
        ejecutarBusquedaExplorar();
      },
    });
    container.appendChild(chip);
  });
}

function ejecutarBusquedaExplorar() {
  const grid = document.getElementById("explore-grid");
  if (!grid) return;
  grid.innerHTML = "";

  let items = motorLocal.buscar(state.explore.q, state.explore.categoria, 48);

  if (state.explore.clusterId !== null) {
    const cl = motorLocal.clusters.find(c => c.id === state.explore.clusterId);
    if (cl) {
      items = items.filter(it => cl.indices.includes(motorLocal.items.indexOf(it)));
    }
  }

  if (items.length === 0) {
    grid.appendChild(
      el("p", {
        style: "color: var(--text-muted); font-size: 0.92rem; grid-column: 1 / -1;",
        textContent: "No se encontraron obras coincidentes.",
      })
    );
    return;
  }

  items.forEach((it) => {
    grid.appendChild(createItemCard(it));
  });
}

// ============================================================================
// VISTA MI PERFIL
// ============================================================================
function cargarVistaPerfil() {
  const countEl = document.getElementById("profile-ratings-count");
  const historyGrid = document.getElementById("profile-history-grid");
  const recsGrid = document.getElementById("profile-recs-grid");

  const ratedItems = [];
  state.ratingsMap.forEach((val, itemId) => {
    const found = ITEMS_CATALOGO.find(i => String(i.id) === String(itemId));
    if (found) {
      ratedItems.push({ ...found, calificacion_usuario: val });
    }
  });

  ratedItems.sort((a, b) => b.calificacion_usuario - a.calificacion_usuario);

  if (countEl) countEl.textContent = String(ratedItems.length);

  if (historyGrid) {
    historyGrid.innerHTML = "";
    if (ratedItems.length === 0) {
      historyGrid.appendChild(
        el("p", {
          style: "color: var(--text-muted); font-size: 0.88rem;",
          textContent: "Aún no has calificado ninguna obra. Puedes calificar de 1 a 5 estrellas en cualquier tarjeta.",
        })
      );
    } else {
      ratedItems.forEach((it) => {
        historyGrid.appendChild(
          el("div", { className: "history-card" }, [
            el("div", {}, [
              el("strong", { textContent: it.nombre }),
              el("span", { textContent: domainName(it.categoria) }),
            ]),
            el("span", { className: "history-badge", textContent: `${it.calificacion_usuario}/5` }),
          ])
        );
      });
    }
  }

  if (recsGrid) {
    recsGrid.innerHTML = "";
    let queryTokens = [];
    ratedItems.filter(r => r.calificacion_usuario >= 4).forEach(r => {
      queryTokens.push(r.nombre, (r.etiquetas || []).join(" "), r.categoria);
    });

    const profileRecs = motorLocal.recomendarPorCuestionario({
      categoria: "todo",
      referencias: queryTokens,
      cantidad: 4,
    });

    profileRecs.forEach((it) => {
      recsGrid.appendChild(createItemCard(it));
    });
  }
}

// ============================================================================
// ARRANQUE
// ============================================================================
function arrancar() {
  renderStage();

  const prevBtn = document.getElementById("btn-prev");
  const nextBtn = document.getElementById("btn-next");
  if (prevBtn) prevBtn.addEventListener("click", handlePrevStep);
  if (nextBtn) nextBtn.addEventListener("click", handleNextStep);

  const editBtn = document.getElementById("btn-edit-quiz");
  const restartBtn = document.getElementById("btn-restart-quiz");
  if (editBtn) {
    editBtn.addEventListener("click", () => {
      document.getElementById("wizard-results").hidden = true;
      state.wizard.step = 1;
      renderStage();
    });
  }
  if (restartBtn) {
    restartBtn.addEventListener("click", () => {
      document.getElementById("wizard-results").hidden = true;
      state.wizard.step = 1;
      state.wizard.maxReached = 1;
      state.wizard.referencias = [];
      state.wizard.atributos = [];
      state.wizard.evitar = [];
      state.wizard.texto_libre = "";
      renderStage();
    });
  }

  document.querySelectorAll(".nav-link").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      switchView(e.currentTarget.dataset.view);
    });
  });

  const searchInput = document.getElementById("search-input");
  const searchBtn = document.getElementById("btn-search-exec");
  if (searchInput) {
    searchInput.addEventListener("input", (e) => {
      state.explore.q = e.target.value;
      ejecutarBusquedaExplorar();
    });
  }
  if (searchBtn) {
    searchBtn.addEventListener("click", () => {
      ejecutarBusquedaExplorar();
    });
  }

  const refreshProfileBtn = document.getElementById("btn-refresh-profile");
  if (refreshProfileBtn) {
    refreshProfileBtn.addEventListener("click", () => {
      cargarVistaPerfil();
      showToast("Recomendaciones de perfil actualizadas");
    });
  }

  const closeBtn = document.getElementById("btn-close-modal");
  if (closeBtn) closeBtn.addEventListener("click", cerrarModal);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", arrancar);
} else {
  arrancar();
}
"""

final_js = js_template.replace("__ITEMS_JSON__", items_json).replace("__QUESTIONS_CODE__", questions_clean)

# Ensamblar el documento HTML completo
standalone_html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta name="color-scheme" content="dark" />
  <title>Criterio — Recomendador Inteligente</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link
    href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap"
    rel="stylesheet"
  />
  <style>
{css_content}
  </style>
</head>
<body>
{body_content}
  <script>
{final_js}
  </script>
</body>
</html>
"""

# Guardar en raíz como index.html y como recomendador.html
(ROOT / "index.html").write_text(standalone_html, encoding="utf-8")
(ROOT / "recomendador.html").write_text(standalone_html, encoding="utf-8")
print("Archivos generados exitosamente:")
print(" -", ROOT / "index.html")
print(" -", ROOT / "recomendador.html")

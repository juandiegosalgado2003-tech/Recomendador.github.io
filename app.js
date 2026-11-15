import {
  REFERENCIAS_POPULARES,
  ATRIBUTOS_CALIDAD,
  FORMATO_DURACION,
  FORMATO_COMPANIA,
  FORMATO_PRESUPUESTO,
  CRITERIOS_EXCLUSION,
  EJEMPLOS_TEXTO_LIBRE,
} from "/static/questions.js";

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

const DOMINIOS_FALLBACK = [
  { id: "peliculas", nombre: "Cine y Películas", icono: "film", total: 16 },
  { id: "series", nombre: "Series de Televisión", icono: "tv", total: 12 },
  { id: "libros", nombre: "Literatura y Libros", icono: "book", total: 12 },
  { id: "musica", nombre: "Música y Álbumes", icono: "music", total: 12 },
  { id: "videojuegos", nombre: "Videojuegos", icono: "gamepad", total: 12 },
  { id: "comida", nombre: "Gastronomía y Cocina", icono: "utensils", total: 12 },
  { id: "viajes", nombre: "Viajes y Rutas", icono: "compass", total: 10 },
  { id: "hobbies_regalos", nombre: "Actividades y Aficiones", icono: "gift", total: 12 },
];

const STEP_METAS = [
  { step: 1, label: "Paso 1 de 6", title: "¿Qué te apetece explorar hoy?" },
  { step: 2, label: "Paso 2 de 6", title: "Elige tus obras o referentes de cabecera" },
  { step: 3, label: "Paso 3 de 6", title: "¿Qué cualidades o sensaciones buscas?" },
  { step: 4, label: "Paso 4 de 6", title: "¿En qué plan estás y cuánto tiempo tienes?" },
  { step: 5, label: "Paso 5 de 6", title: "¿Qué prefieres descartar expresamente?" },
  { step: 6, label: "Paso 6 de 6", title: "Detalles finales y tu nombre de perfil" },
];

const state = {
  estado: null,
  activeView: "wizard",
  activeUser: "Lector_Curioso",
  ratingsMap: new Map(),
  allCatalogItems: [],
  clustersData: null,
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
    usuario: "Lector_Curioso",
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

// ============================================================================
// DOM HELPERS
// ============================================================================
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

async function api(url, options = {}) {
  const res = await fetch(url, options);
  if (!res.ok) {
    let detail = "Error de conexión con el servicio";
    try {
      const err = await res.json();
      detail = err.detail || detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json();
}

function showToast(msg) {
  const dock = document.getElementById("toast-dock");
  if (!dock) return;
  const toast = el("div", { className: "toast", textContent: msg });
  dock.appendChild(toast);
  setTimeout(() => {
    toast.remove();
  }, 3000);
}

function domainName(catId) {
  if (catId === "todo") return "Todas las categorías";
  const doms = state.estado?.dominios || DOMINIOS_FALLBACK;
  const found = doms.find((d) => d.id === catId);
  return found ? found.nombre : catId;
}

function switchView(viewName) {
  state.activeView = viewName;
  document.querySelectorAll(".nav-link").forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.view === viewName);
  });
  document.querySelectorAll(".view").forEach((sec) => {
    const active = sec.id === `view-${viewName}`;
    sec.hidden = !active;
  });

  if (viewName === "explorar" && state.allCatalogItems.length === 0) {
    cargarCatalogoExplorador();
  } else if (viewName === "perfil") {
    cargarVistaPerfil();
  }
}

// ============================================================================
// STEPPER & QUESTIONNAIRE FLOW
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

  if (labelEl) {
    labelEl.textContent = STEP_METAS[state.wizard.step - 1].label;
  }
  if (progressFill) {
    const pct = ((state.wizard.step - 1) / 5) * 100;
    progressFill.style.width = `${Math.max(16.66, pct)}%`;
  }

  const prevBtn = document.getElementById("btn-prev");
  const nextBtn = document.getElementById("btn-next");
  if (prevBtn) {
    prevBtn.disabled = state.wizard.step === 1;
  }
  if (nextBtn) {
    if (state.wizard.step === 6) {
      nextBtn.textContent = "Obtener recomendaciones";
    } else {
      nextBtn.textContent = "Siguiente";
    }
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
    case 1:
      renderStep1(stage);
      break;
    case 2:
      renderStep2(stage);
      break;
    case 3:
      renderStep3(stage);
      break;
    case 4:
      renderStep4(stage);
      break;
    case 5:
      renderStep5(stage);
      break;
    case 6:
      renderStep6(stage);
      break;
  }
}

// ETAPA 1: CATEGORÍA
function renderStep1(container) {
  const grid = el("div", { className: "category-grid" });

  const allOption = el(
    "button",
    {
      type: "button",
      className: `category-card ${state.wizard.categoria === "todo" ? "is-selected" : ""}`,
      onClick: (e) => {
        state.wizard.categoria = "todo";
        state.wizard.referencias = [];
        grid.querySelectorAll(".category-card").forEach((c) => c.classList.remove("is-selected"));
        e.currentTarget.classList.add("is-selected");
      },
    },
    [
      el("div", { className: "category-card-icon" }, [createIcon("all")]),
      el("div", { className: "category-card-info" }, [
        el("strong", { textContent: "Todas las categorías" }),
        el("span", { textContent: "Sugerencias transversales" }),
      ]),
    ]
  );
  grid.appendChild(allOption);

  const doms = (state.estado?.dominios || DOMINIOS_FALLBACK).filter((d) => d.id !== "todo");
  doms.forEach((dom) => {
    const isSel = state.wizard.categoria === dom.id;
    const card = el(
      "button",
      {
        type: "button",
        className: `category-card ${isSel ? "is-selected" : ""}`,
        onClick: (e) => {
          state.wizard.categoria = dom.id;
          state.wizard.referencias = [];
          grid.querySelectorAll(".category-card").forEach((c) => c.classList.remove("is-selected"));
          e.currentTarget.classList.add("is-selected");
        },
      },
      [
        el("div", { className: "category-card-icon" }, [createIcon(dom.icono || dom.id)]),
        el("div", { className: "category-card-info" }, [
          el("strong", { textContent: dom.nombre }),
          el("span", { textContent: `${dom.total || dom.total_items || 12} obras curadas` }),
        ]),
      ]
    );
    grid.appendChild(card);
  });

  container.appendChild(grid);
}

// ETAPA 2: REFERENCIAS CULTURALES
function renderStep2(container) {
  const cat = state.wizard.categoria;
  const refs = REFERENCIAS_POPULARES[cat] || REFERENCIAS_POPULARES.todo;
  const grid = el("div", { className: "references-grid" });

  refs.forEach((ref) => {
    const isSel = state.wizard.referencias.includes(ref.label);
    const card = el(
      "div",
      {
        className: `reference-card ${isSel ? "is-selected" : ""}`,
        onClick: () => {
          const pos = state.wizard.referencias.indexOf(ref.label);
          if (pos >= 0) {
            state.wizard.referencias.splice(pos, 1);
            card.classList.remove("is-selected");
          } else {
            state.wizard.referencias.push(ref.label);
            card.classList.add("is-selected");
          }
        },
      },
      [
        el("strong", { textContent: ref.label }),
        el("span", { className: "status-dot", style: isSel ? "background: var(--accent-gold);" : "opacity: 0.3;" }),
      ]
    );
    grid.appendChild(card);
  });
  container.appendChild(grid);

  // Cuadro para escribir referente propio
  const customBox = el("div", { className: "custom-input-box" }, [
    el("label", { for: "custom-ref-input", textContent: "¿Tienes otra obra favorita en mente? Escríbela aquí:" }),
    el("input", {
      id: "custom-ref-input",
      type: "text",
      className: "input-text",
      placeholder: "Ej: Blade Runner, Haruki Murakami, pasta fresca, Radiohead...",
      value: state.wizard.referencia_libre,
      onInput: (e) => {
        state.wizard.referencia_libre = e.target.value;
      },
    }),
  ]);
  container.appendChild(customBox);
}

// ETAPA 3: ATRIBUTOS Y TONO
function renderStep3(container) {
  const grid = el("div", { className: "attributes-grid" });

  ATRIBUTOS_CALIDAD.forEach((attr) => {
    const isSel = state.wizard.atributos.includes(attr.id);
    const card = el(
      "div",
      {
        className: `attribute-card ${isSel ? "is-selected" : ""}`,
        onClick: () => {
          const pos = state.wizard.atributos.indexOf(attr.id);
          if (pos >= 0) {
            state.wizard.atributos.splice(pos, 1);
            card.classList.remove("is-selected");
          } else {
            state.wizard.atributos.push(attr.id);
            card.classList.add("is-selected");
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

// ETAPA 4: FORMATO Y COMPAÑÍA
function renderStep4(container) {
  const stack = el("div", { className: "format-stack" });

  // Duración
  const durGroup = el("div", { className: "format-group" }, [
    el("label", { textContent: "Duración o extensión prevista" }),
    el(
      "div",
      { className: "chips-row" },
      FORMATO_DURACION.map((opt) =>
        el("button", {
          type: "button",
          className: `choice-chip ${state.wizard.duracion === opt.id ? "is-selected" : ""}`,
          textContent: opt.label,
          onClick: (e) => {
            state.wizard.duracion = opt.id;
            e.currentTarget.parentElement.querySelectorAll(".choice-chip").forEach((c) => c.classList.remove("is-selected"));
            e.currentTarget.classList.add("is-selected");
          },
        })
      )
    ),
  ]);
  stack.appendChild(durGroup);

  // Compañía
  const compGroup = el("div", { className: "format-group" }, [
    el("label", { textContent: "¿Con quién lo vas a disfrutar?" }),
    el(
      "div",
      { className: "chips-row" },
      FORMATO_COMPANIA.map((opt) =>
        el("button", {
          type: "button",
          className: `choice-chip ${state.wizard.compania === opt.id ? "is-selected" : ""}`,
          textContent: opt.label,
          onClick: (e) => {
            state.wizard.compania = opt.id;
            e.currentTarget.parentElement.querySelectorAll(".choice-chip").forEach((c) => c.classList.remove("is-selected"));
            e.currentTarget.classList.add("is-selected");
          },
        })
      )
    ),
  ]);
  stack.appendChild(compGroup);

  container.appendChild(stack);
}

// ETAPA 5: FILTROS NEGATIVOS / DESCARTE
function renderStep5(container) {
  const grid = el("div", { className: "exclusions-grid" });

  CRITERIOS_EXCLUSION.forEach((exc) => {
    const isSel = state.wizard.evitar.includes(exc.id);
    const card = el(
      "div",
      {
        className: `exclusion-card ${isSel ? "is-selected" : ""}`,
        onClick: () => {
          const pos = state.wizard.evitar.indexOf(exc.id);
          const ind = card.querySelector(".exclusion-indicator");
          if (pos >= 0) {
            state.wizard.evitar.splice(pos, 1);
            card.classList.remove("is-selected");
            if (ind) ind.textContent = "Permitido";
          } else {
            state.wizard.evitar.push(exc.id);
            card.classList.add("is-selected");
            if (ind) ind.textContent = "Descartado";
          }
        },
      },
      [
        el("span", { textContent: exc.label }),
        el("span", { className: "exclusion-indicator", textContent: isSel ? "Descartado" : "Permitido" }),
      ]
    );
    grid.appendChild(card);
  });

  container.appendChild(grid);
}

// ETAPA 6: NOTAS FINALES Y PERFIL
function renderStep6(container) {
  const layout = el("div", { className: "step-final-layout" });

  const leftCol = el("div", {}, [
    el("label", {
      for: "wizard-free-text",
      style: "display: block; font-size: 0.88rem; font-weight: 600; color: var(--text-secondary); margin-bottom: 0.5rem;",
      textContent: "Describe libremente qué buscas hoy en tus propias palabras:",
    }),
    el("textarea", {
      id: "wizard-free-text",
      className: "textarea-input",
      placeholder: "Ej: Busco algo reflexivo y tranquilo para desconectar por las noches...",
      value: state.wizard.texto_libre,
      onInput: (e) => {
        state.wizard.texto_libre = e.target.value;
      },
    }),
    el("div", { className: "prompt-suggestions" }, [
      ...EJEMPLOS_TEXTO_LIBRE.map((promptText) =>
        el("button", {
          type: "button",
          className: "prompt-btn",
          textContent: promptText,
          onClick: () => {
            state.wizard.texto_libre = promptText;
            const ta = document.getElementById("wizard-free-text");
            if (ta) ta.value = promptText;
          },
        })
      ),
    ]),
  ]);
  layout.appendChild(leftCol);

  const rightCol = el("div", { className: "profile-box" }, [
    el("div", {}, [
      el("label", {
        for: "wizard-user-input",
        style: "display: block; font-size: 0.82rem; font-weight: 600; color: var(--accent-gold); margin-bottom: 0.4rem;",
        textContent: "Tu nombre de usuario o perfil",
      }),
      el("input", {
        id: "wizard-user-input",
        type: "text",
        className: "input-text",
        value: state.wizard.usuario || "Lector_Curioso",
        onInput: (e) => {
          state.wizard.usuario = e.target.value.trim() || "Invitado";
        },
      }),
    ]),
    el("p", {
      style: "font-size: 0.84rem; color: var(--text-muted); line-height: 1.45; margin: 0;",
      textContent: `Ámbito: ${domainName(state.wizard.categoria)} • ${state.wizard.referencias.length} referentes • ${state.wizard.atributos.length} cualidades seleccionadas`,
    }),
  ]);
  layout.appendChild(rightCol);

  container.appendChild(layout);
}

function handleNext() {
  if (state.wizard.step < 6) {
    state.wizard.step += 1;
    if (state.wizard.step > state.wizard.maxReached) {
      state.wizard.maxReached = state.wizard.step;
    }
    renderStage();
  } else {
    solicitarRecomendacion();
  }
}

function handlePrev() {
  if (state.wizard.step > 1) {
    state.wizard.step -= 1;
    renderStage();
  }
}

// ============================================================================
// SOLICITUD DE RECOMENDACIONES
// ============================================================================
async function solicitarRecomendacion() {
  const nextBtn = document.getElementById("btn-next");
  if (nextBtn) {
    nextBtn.disabled = true;
    nextBtn.textContent = "Analizando afinidades...";
  }

  const todasReferencias = [...state.wizard.referencias];
  if (state.wizard.referencia_libre.trim()) {
    todasReferencias.push(state.wizard.referencia_libre.trim());
  }

  const payload = {
    categoria: state.wizard.categoria,
    referencias: todasReferencias,
    atributos: state.wizard.atributos,
    duracion: state.wizard.duracion,
    compania: state.wizard.compania,
    presupuesto: state.wizard.presupuesto,
    evitar: state.wizard.evitar,
    texto_libre: state.wizard.texto_libre,
    usuario: state.wizard.usuario || "Lector_Curioso",
    guardar_perfil: true,
    cantidad: state.wizard.cantidad,
  };

  try {
    const data = await api("/api/cuestionario/recomendar", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    state.wizard.resultados = data.recomendaciones || [];
    state.wizard.explicacion = `Selección adaptada a tu perfil en ${domainName(state.wizard.categoria)}.`;

    renderResultados();
    showToast(`Se han generado ${state.wizard.resultados.length} recomendaciones para ti`);
  } catch (err) {
    showToast(`No se pudo calcular: ${err.message}`);
  } finally {
    if (nextBtn) {
      nextBtn.disabled = false;
      nextBtn.textContent = "Obtener recomendaciones";
    }
  }
}

function renderResultados() {
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
        textContent: "No se encontraron obras con los criterios seleccionados. Prueba a relajar algún filtro.",
      })
    );
    return;
  }

  state.wizard.resultados.forEach((item) => {
    grid.appendChild(createItemCard(item));
  });

  resultsContainer.scrollIntoView({ behavior: "smooth", block: "start" });
}

// CREACIÓN DE TARJETA ELEGANTE Y LIMPIA
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

// ============================================================================
// VALORACIÓN Y CALIFICACIONES
// ============================================================================
async function registrarCalificacion(itemId, score, cardElement) {
  state.ratingsMap.set(itemId, score);
  if (cardElement) {
    const btns = cardElement.querySelectorAll(".rate-btn");
    btns.forEach((btn, idx) => {
      btn.classList.toggle("is-active", idx + 1 <= score);
    });
  }

  try {
    await api("/api/interacciones", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        usuario: state.wizard.usuario || "Lector_Curioso",
        item_id: itemId,
        calificacion: score,
      }),
    });
    showToast(`Calificación ${score}/5 guardada en tu perfil`);
  } catch (err) {
    showToast(`Error al guardar: ${err.message}`);
  }
}

// ============================================================================
// MODAL DE DETALLE
// ============================================================================
async function abrirModal(item) {
  const modal = document.getElementById("item-modal");
  if (!modal) return;

  document.getElementById("modal-category").textContent = domainName(item.categoria);
  document.getElementById("modal-item-title").textContent = item.nombre;
  document.getElementById("modal-desc").textContent = item.descripcion || item.texto || "";

  const tagsContainer = document.getElementById("modal-tags");
  tagsContainer.innerHTML = "";
  (item.etiquetas || item.tags || []).forEach((t) => {
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

  // Tira de valoración
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

  // Obras similares
  const similarsGrid = document.getElementById("modal-similars-grid");
  similarsGrid.innerHTML = el("span", { style: "font-size: 0.8rem; color: var(--text-muted);", textContent: "Buscando similares..." }).outerHTML;

  try {
    const data = await api(`/api/items/${item.id}/similares?cantidad=4`);
    similarsGrid.innerHTML = "";
    if (data.similares && data.similares.length) {
      data.similares.forEach((sim) => {
        similarsGrid.appendChild(
          el(
            "div",
            {
              className: "similar-card",
              onClick: () => abrirModal(sim),
            },
            [
              el("strong", { textContent: sim.nombre }),
              el("span", { textContent: domainName(sim.categoria) }),
            ]
          )
        );
      });
    } else {
      similarsGrid.innerHTML = el("span", { style: "font-size: 0.8rem; color: var(--text-muted);", textContent: "No hay similares directos." }).outerHTML;
    }
  } catch {
    similarsGrid.innerHTML = "";
  }

  modal.showModal();
}

function cerrarModal() {
  const modal = document.getElementById("item-modal");
  if (modal) modal.close();
}

// ============================================================================
// EXPLORADOR Y ORGANIZACIÓN POR CLUSTERS TEMÁTICOS
// ============================================================================
async function cargarCatalogoExplorador() {
  try {
    const [catRes, clustRes] = await Promise.all([
      api("/api/catalogo?limit=180"),
      api("/api/clustering?k_items=8"),
    ]);

    state.allCatalogItems = catRes.items || [];
    state.clustersData = clustRes.clusters_items || [];

    renderColeccionesTematicas();
    renderCategoriasExplorador();
    filtrarCatalogo();
  } catch (err) {
    showToast(`Error al cargar catálogo: ${err.message}`);
  }
}

// Renderiza las colecciones temáticas (el clustering organizando contenidos por temas)
function renderColeccionesTematicas() {
  const container = document.getElementById("explore-cluster-chips");
  if (!container || !state.clustersData) return;

  container.innerHTML = "";

  const allChip = el("button", {
    type: "button",
    className: `cluster-chip ${state.explore.clusterId === null ? "is-active" : ""}`,
    textContent: "Todas las temáticas",
    onClick: () => {
      state.explore.clusterId = null;
      document.querySelectorAll(".cluster-chip").forEach((c) => c.classList.remove("is-active"));
      allChip.classList.add("is-active");
      filtrarCatalogo();
    },
  });
  container.appendChild(allChip);

  state.clustersData.forEach((c) => {
    const cleanName = c.nombre ? c.nombre.replace(/^Clúster \d+ · /, "") : `Tema ${c.id + 1}`;
    const capitalized = cleanName.charAt(0).toUpperCase() + cleanName.slice(1);

    const chip = el("button", {
      type: "button",
      className: `cluster-chip ${state.explore.clusterId === c.id ? "is-active" : ""}`,
      textContent: `${capitalized} (${c.total_items})`,
      onClick: () => {
        state.explore.clusterId = state.explore.clusterId === c.id ? null : c.id;
        document.querySelectorAll(".cluster-chip").forEach((ch) => ch.classList.remove("is-active"));
        if (state.explore.clusterId === c.id) {
          chip.classList.add("is-active");
        } else {
          allChip.classList.add("is-active");
        }
        filtrarCatalogo();
      },
    });
    container.appendChild(chip);
  });
}

function renderCategoriasExplorador() {
  const container = document.getElementById("explore-category-chips");
  if (!container) return;

  container.innerHTML = "";

  const allBtn = el("button", {
    type: "button",
    className: `category-chip ${state.explore.categoria === "todo" ? "is-active" : ""}`,
    textContent: "Todos los ámbitos",
    onClick: () => {
      state.explore.categoria = "todo";
      document.querySelectorAll(".category-chip").forEach((c) => c.classList.remove("is-active"));
      allBtn.classList.add("is-active");
      filtrarCatalogo();
    },
  });
  container.appendChild(allBtn);

  const doms = (state.estado?.dominios || DOMINIOS_FALLBACK).filter((d) => d.id !== "todo");
  doms.forEach((dom) => {
    const chip = el("button", {
      type: "button",
      className: `category-chip ${state.explore.categoria === dom.id ? "is-active" : ""}`,
      textContent: dom.nombre,
      onClick: () => {
        state.explore.categoria = dom.id;
        document.querySelectorAll(".category-chip").forEach((c) => c.classList.remove("is-active"));
        chip.classList.add("is-active");
        filtrarCatalogo();
      },
    });
    container.appendChild(chip);
  });
}

function filtrarCatalogo() {
  const grid = document.getElementById("explore-grid");
  if (!grid) return;

  const query = state.explore.q.toLowerCase().trim();
  const cat = state.explore.categoria;
  const clustId = state.explore.clusterId;

  const filtrados = state.allCatalogItems.filter((item) => {
    if (cat !== "todo" && item.categoria !== cat) return false;
    if (clustId !== null && item.cluster_id !== clustId) return false;
    if (query) {
      const corpus = `${item.nombre} ${item.descripcion || ""} ${item.texto || ""} ${(item.etiquetas || []).join(" ")}`.toLowerCase();
      return query.split(" ").every((w) => corpus.includes(w));
    }
    return true;
  });

  grid.innerHTML = "";
  if (filtrados.length === 0) {
    grid.appendChild(
      el("p", {
        style: "color: var(--text-muted); font-size: 0.92rem; padding: 1.5rem 0;",
        textContent: "No se encontraron obras con los filtros aplicados.",
      })
    );
    return;
  }

  filtrados.slice(0, 36).forEach((item) => {
    grid.appendChild(createItemCard(item));
  });
}

// ============================================================================
// PERFIL Y VALORACIONES
// ============================================================================
async function cargarVistaPerfil() {
  const userInput = document.getElementById("profile-user-input");
  const methodSelect = document.getElementById("profile-method-select");
  const recsGrid = document.getElementById("profile-recs-grid");
  const histGrid = document.getElementById("profile-history-grid");
  const countEl = document.getElementById("profile-ratings-count");

  const usuario = userInput?.value || state.wizard.usuario || "Lector_Curioso";
  const metodo = methodSelect?.value || "combinado";

  if (recsGrid) {
    recsGrid.innerHTML = el("span", { style: "color: var(--text-muted); font-size: 0.86rem;", textContent: "Cargando sugerencias..." }).outerHTML;
  }

  try {
    const data = await api(`/api/usuarios/${usuario}/recomendar?metodo=${metodo}&cantidad=6`);

    if (recsGrid) {
      recsGrid.innerHTML = "";
      if (data.recomendaciones && data.recomendaciones.length) {
        data.recomendaciones.forEach((item) => recsGrid.appendChild(createItemCard(item)));
      } else {
        recsGrid.innerHTML = el("p", { style: "color: var(--text-muted); font-size: 0.88rem;", textContent: "No hay recomendaciones aún para este perfil." }).outerHTML;
      }
    }

    const hist = data.historial || [];
    if (countEl) countEl.textContent = String(hist.length);
    if (histGrid) {
      histGrid.innerHTML = "";
      if (hist.length) {
        hist.forEach((h) => {
          histGrid.appendChild(
            el("div", { className: "history-card" }, [
              el("div", {}, [
                el("strong", { textContent: h.item_nombre || h.item }),
                el("span", { textContent: domainName(h.categoria || "general") }),
              ]),
              el("span", { className: "history-badge", textContent: `${h.calificacion} / 5` }),
            ])
          );
        });
      } else {
        histGrid.innerHTML = el("p", { style: "color: var(--text-muted); font-size: 0.88rem;", textContent: "Aún no has calificado obras en este perfil." }).outerHTML;
      }
    }
  } catch (err) {
    if (recsGrid) recsGrid.innerHTML = el("p", { style: "color: var(--text-muted); font-size: 0.88rem;", textContent: "No se pudieron cargar recomendaciones." }).outerHTML;
  }
}

// ============================================================================
// INICIALIZACIÓN
// ============================================================================
async function inicializarEstado() {
  try {
    const estado = await api("/api/estado");
    state.estado = estado;

    const countBadge = document.getElementById("catalog-count-badge");
    if (countBadge) {
      const tot = estado.estadisticas?.total_items || estado.total_items || 98;
      countBadge.textContent = `${tot} obras en catálogo`;
    }
  } catch {
    const countBadge = document.getElementById("catalog-count-badge");
    if (countBadge) countBadge.textContent = "Modo local";
  }
}

function registrarEventos() {
  // Navegación
  document.querySelectorAll(".nav-link").forEach((btn) => {
    btn.addEventListener("click", () => switchView(btn.dataset.view));
  });

  // Stepper
  const prevBtn = document.getElementById("btn-prev");
  const nextBtn = document.getElementById("btn-next");
  if (prevBtn) prevBtn.addEventListener("click", handlePrev);
  if (nextBtn) nextBtn.addEventListener("click", handleNext);

  // Resultados
  const editBtn = document.getElementById("btn-edit-quiz");
  if (editBtn) {
    editBtn.addEventListener("click", () => {
      document.getElementById("wizard-results").hidden = true;
      document.getElementById("wizard-shell")?.scrollIntoView({ behavior: "smooth" });
    });
  }

  const restartBtn = document.getElementById("btn-restart-quiz");
  if (restartBtn) {
    restartBtn.addEventListener("click", () => {
      state.wizard.step = 1;
      state.wizard.referencias = [];
      state.wizard.atributos = [];
      state.wizard.evitar = [];
      state.wizard.texto_libre = "";
      document.getElementById("wizard-results").hidden = true;
      renderStage();
      document.getElementById("wizard-shell")?.scrollIntoView({ behavior: "smooth" });
    });
  }

  // Explorador
  const searchInput = document.getElementById("search-input");
  const searchExec = document.getElementById("btn-search-exec");

  if (searchInput) {
    searchInput.addEventListener("input", (e) => {
      state.explore.q = e.target.value;
      filtrarCatalogo();
    });
    searchInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        filtrarCatalogo();
      }
    });
  }
  if (searchExec) {
    searchExec.addEventListener("click", () => {
      if (searchInput) state.explore.q = searchInput.value;
      filtrarCatalogo();
    });
  }

  // Perfil
  const refreshProfileBtn = document.getElementById("btn-refresh-profile");
  if (refreshProfileBtn) {
    refreshProfileBtn.addEventListener("click", cargarVistaPerfil);
  }

  // Modal
  const closeBtn = document.getElementById("btn-close-modal");
  if (closeBtn) closeBtn.addEventListener("click", cerrarModal);
}

function arrancar() {
  registrarEventos();
  renderStage();
  inicializarEstado();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", arrancar);
} else {
  arrancar();
}

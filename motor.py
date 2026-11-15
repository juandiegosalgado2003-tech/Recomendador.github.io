from __future__ import annotations

import os
import re
import unicodedata
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

SEMILLA = 42
UMBRAL_GUSTO = 4
N_RECOMENDACIONES = 6
N_COMPONENTES = 6
PESO_COLABORATIVO = 0.5
PROPORCION_PRUEBA = 0.2

METODOS = ["popularidad", "contenido", "colaborativo", "clustering", "combinado"]

STOPWORDS_ES = {
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
    "va", "vamos", "van", "ya", "yo",
}

COLUMNAS_OPCIONALES = {
    "categoria": "general",
    "subcategoria": "",
    "descripcion": "",
    "etiquetas": "",
    "tono": "",
    "presupuesto": "",
    "duracion": "",
    "compania": "",
    "icono": "sparkles",
}


def normalizar_texto(texto: str) -> str:
    """Convierte texto a minúsculas, elimina tildes y conserva tokens útiles."""
    limpio = unicodedata.normalize("NFKD", str(texto).lower())
    sin_tildes = "".join(c for c in limpio if not unicodedata.combining(c))
    tokens = re.findall(r"[a-z0-9_]+", sin_tildes)
    filtrados = [t for t in tokens if len(t) > 1 and t not in STOPWORDS_ES]
    return " ".join(filtrados)


def _split_tags(valor: Any) -> list[str]:
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return []
    partes = [p.strip() for p in str(valor).split(",")]
    return [p for p in partes if p]


def preparar(items: pd.DataFrame | str, interacciones: pd.DataFrame | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Limpia y estandariza tablas de items e interacciones manteniendo compatibilidad con el Colab."""
    if isinstance(items, str):
        items = pd.read_csv(items)
    if isinstance(interacciones, str):
        interacciones = pd.read_csv(interacciones)

    items = items.copy()
    interacciones = interacciones.copy()

    if "id" not in items.columns and "item" in items.columns:
        items = items.rename(columns={"item": "id"})
    if "nombre" not in items.columns:
        items["nombre"] = items["id"].astype(str)
    if "texto" not in items.columns:
        items["texto"] = items["nombre"].astype(str)

    items["texto"] = items["texto"].fillna("").astype(str)
    items["nombre"] = items["nombre"].fillna("").astype(str)

    for col, default in COLUMNAS_OPCIONALES.items():
        if col not in items.columns:
            items[col] = default
        else:
            items[col] = items[col].fillna(default).astype(str)

    # Asegura texto enriquecido para TF-IDF si hay metadatos pero texto corto
    texto_enriquecido = (
        items["nombre"] + " " + items["texto"] + " " + items["categoria"] + " "
        + items["subcategoria"] + " " + items["etiquetas"] + " " + items["tono"]
    )
    items["_texto_motor"] = texto_enriquecido.str.strip()

    if len(interacciones) == 0:
        interacciones = pd.DataFrame(columns=["usuario", "item", "calificacion"])
    else:
        if "calificacion" not in interacciones.columns:
            if "clic" in interacciones.columns:
                interacciones = interacciones[interacciones["clic"] > 0]
            interacciones["calificacion"] = 5.0
        interacciones = interacciones[["usuario", "item", "calificacion"]].copy()
        interacciones["calificacion"] = pd.to_numeric(interacciones["calificacion"], errors="coerce").fillna(4.0)

    return items, interacciones


def normalizar(m: np.ndarray) -> np.ndarray:
    """Normaliza filas de una matriz al rango [0, 1]."""
    if m.size == 0:
        return m
    mn = m.min(axis=1, keepdims=True)
    mx = m.max(axis=1, keepdims=True)
    return (m - mn) / np.where(mx - mn == 0, 1.0, mx - mn)


def _normalizar_vector(v: np.ndarray) -> np.ndarray:
    if v.size == 0:
        return v
    mn = float(v.min())
    mx = float(v.max())
    if mx - mn <= 1e-9:
        return np.zeros_like(v, dtype=float)
    return (v - mn) / (mx - mn)


def _extraer_palabras_clave(centroid: np.ndarray, vocab: np.ndarray, top_n: int = 4) -> list[str]:
    if centroid.size == 0 or vocab.size == 0:
        return ["general"]
    orden = np.argsort(-centroid)
    palabras: list[str] = []
    vistas_unigram = set()
    for idx in orden:
        if centroid[idx] <= 0:
            break
        termino = str(vocab[idx]).strip()
        if not termino:
            continue
        partes = set(termino.split())
        if len(partes) == 1 and termino in vistas_unigram:
            continue
        if len(partes) > 1 and partes.issubset(vistas_unigram):
            continue
        palabras.append(termino)
        vistas_unigram.update(partes)
        if len(palabras) >= top_n:
            break
    return palabras or ["general"]


def _proyectar_2d(matriz: np.ndarray) -> np.ndarray:
    n = matriz.shape[0]
    if n == 0:
        return np.zeros((0, 2), dtype=float)
    if n == 1 or matriz.shape[1] < 2:
        xs = np.linspace(15.0, 85.0, n) if n > 1 else np.array([50.0])
        ys = np.full(n, 50.0)
        return np.column_stack([xs, ys])

    svd = TruncatedSVD(n_components=2, random_state=SEMILLA)
    coords = svd.fit_transform(matriz)
    salida = np.zeros((n, 2), dtype=float)
    for col in range(2):
        v = coords[:, col]
        mn, mx = float(v.min()), float(v.max())
        if mx - mn <= 1e-9:
            salida[:, col] = 50.0
        else:
            salida[:, col] = 7.0 + ((v - mn) / (mx - mn)) * 86.0
    return salida


class Recomendador:
    """Motor híbrido de recomendación (TF-IDF + TruncatedSVD + Clustering K-Means + Popularidad)."""

    def __init__(
        self,
        items: pd.DataFrame | str,
        interacciones: pd.DataFrame | str,
        peso: float = PESO_COLABORATIVO,
        k_clusters_items: int = 8,
        k_clusters_users: int = 5,
    ):
        items_df, inter_df = preparar(items, interacciones)
        self.items = items_df.reset_index(drop=True)
        self.ids = self.items["id"].tolist()
        self.pos_item = {i: k for k, i in enumerate(self.ids)}
        self.peso = float(peso)

        inter_df = inter_df[inter_df["item"].isin(self.pos_item)].copy()
        self.interacciones = inter_df.reset_index(drop=True)
        self.usuarios = sorted(inter_df["usuario"].unique().tolist(), key=lambda x: str(x))
        self.pos_usuario = {u: k for k, u in enumerate(self.usuarios)}

        self.tfidf = TfidfVectorizer(
            preprocessor=normalizar_texto,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        textos_fit = self.items["_texto_motor"].tolist()
        try:
            self.X = self.tfidf.fit_transform(textos_fit)
        except ValueError:
            # Fallback si todos los textos quedan vacíos tras stopwords
            self.tfidf = TfidfVectorizer()
            self.X = self.tfidf.fit_transform(self.items["nombre"].fillna("item"))
        Xd = self.X.toarray()
        self._Xd = Xd
        self._Xd_norm = normalize(Xd) if len(Xd) else Xd

        n_u = len(self.usuarios)
        n_i = len(self.ids)
        R = np.zeros((n_u, n_i), dtype=float)
        if len(inter_df):
            filas = inter_df["usuario"].map(self.pos_usuario).to_numpy()
            cols = inter_df["item"].map(self.pos_item).to_numpy()
            R[filas, cols] = inter_df["calificacion"].to_numpy()
        self.R = R

        gustados = (R >= UMBRAL_GUSTO).astype(float)
        self._gustados = gustados
        self._M = gustados - 0.5 * ((R > 0) & (R < UMBRAL_GUSTO))
        self.S_cont = cosine_similarity(gustados @ Xd, Xd) if n_u and n_i else np.zeros((n_u, n_i))

        n_comp = min(N_COMPONENTES, min(R.shape) - 1) if n_u and n_i else 0
        if n_comp >= 1:
            svd = TruncatedSVD(n_components=n_comp, random_state=SEMILLA)
            self.S_colab = svd.inverse_transform(svd.fit_transform(self._M))
        else:
            self.S_colab = np.zeros_like(R)

        n_val = (R > 0).sum(axis=0)
        media = np.divide(R.sum(axis=0), n_val, out=np.zeros(n_i), where=n_val > 0)
        self.media_item = media
        self.n_valoraciones = n_val
        self.popularidad = gustados.sum(axis=0) + media / 10.0

        # Pre-normaliza texto de metadatos para comparaciones rápidas en cuestionario
        self._tokens_items = [set(normalizar_texto(t).split()) for t in self.items["_texto_motor"]]

        # Clustering K-Means sobre ítems (TF-IDF) y usuarios (perfiles de gusto)
        self.k_items = int(k_clusters_items)
        self.k_users = int(k_clusters_users)
        self.recalcular_clustering(k_items=self.k_items, k_users=self.k_users)

    def recalcular_clustering(self, k_items: int = 8, k_users: int = 5) -> None:
        """Agrupa ítems y usuarios mediante K-Means y calcula coordenadas 2D para visualización."""
        n_i = len(self.ids)
        n_u = len(self.usuarios)
        try:
            vocab = self.tfidf.get_feature_names_out()
        except Exception:
            vocab = np.array([])

        # 1. Clustering de ítems en espacio TF-IDF normalizado (K-Means esférico)
        if n_i == 0:
            self.k_items = 0
            self.item_clusters = np.array([], dtype=int)
            self.item_centroids = np.zeros((0, 0))
            self.item_cluster_keywords = {}
            self.item_cluster_names = {}
            self.item_coords_2d = np.zeros((0, 2))
        else:
            k_i_real = max(1, min(int(k_items), n_i))
            self.k_items = k_i_real
            if k_i_real == 1:
                self.item_clusters = np.zeros(n_i, dtype=int)
                self.item_centroids = normalize(self._Xd_norm.mean(axis=0, keepdims=True))
            else:
                km_items = KMeans(n_clusters=k_i_real, random_state=SEMILLA, n_init=10)
                self.item_clusters = km_items.fit_predict(self._Xd_norm)
                self.item_centroids = normalize(km_items.cluster_centers_)

            self.item_cluster_keywords = {}
            self.item_cluster_names = {}
            for c in range(k_i_real):
                palabras = _extraer_palabras_clave(self.item_centroids[c], vocab, top_n=3)
                self.item_cluster_keywords[c] = palabras
                self.item_cluster_names[c] = f"Clúster {c + 1} · {', '.join(palabras[:3])}"
            self.item_coords_2d = _proyectar_2d(self._Xd_norm)

        # 2. Clustering de usuarios en comunidades de gusto (perfil TF-IDF ponderado)
        if n_u == 0 or n_i == 0:
            self.k_users = 0
            self.user_clusters = np.array([], dtype=int)
            self.user_centroids = np.zeros((0, self._Xd.shape[1] if n_i else 0))
            self.user_cluster_keywords = {}
            self.user_cluster_names = {}
            self.user_coords_2d = np.zeros((0, 2))
            self.S_cluster = np.zeros((n_u, n_i), dtype=float)
            return

        u_tfidf = self._gustados @ self._Xd
        filas_vacias = np.linalg.norm(u_tfidf, axis=1) == 0
        if np.any(filas_vacias):
            u_tfidf[filas_vacias] = np.maximum(self.R[filas_vacias], 0.0) @ self._Xd
        self._U_pref_norm = normalize(u_tfidf)

        k_u_real = max(1, min(int(k_users), n_u))
        self.k_users = k_u_real
        if k_u_real == 1:
            self.user_clusters = np.zeros(n_u, dtype=int)
            self.user_centroids = normalize(self._U_pref_norm.mean(axis=0, keepdims=True))
        else:
            km_users = KMeans(n_clusters=k_u_real, random_state=SEMILLA, n_init=10)
            self.user_clusters = km_users.fit_predict(self._U_pref_norm)
            self.user_centroids = normalize(km_users.cluster_centers_)

        self.user_cluster_keywords = {}
        self.user_cluster_names = {}
        self.S_cluster = np.zeros((n_u, n_i), dtype=float)
        sim_centroid_items = cosine_similarity(self.user_centroids, self._Xd_norm)

        for c in range(k_u_real):
            palabras = _extraer_palabras_clave(self.user_centroids[c], vocab, top_n=3)
            self.user_cluster_keywords[c] = palabras
            self.user_cluster_names[c] = f"Comunidad {c + 1} · {', '.join(palabras[:3])}"

            miembros = np.where(self.user_clusters == c)[0]
            if len(miembros):
                colab_cluster = self._M[miembros].mean(axis=0)
                score_c = 0.65 * _normalizar_vector(colab_cluster) + 0.35 * _normalizar_vector(sim_centroid_items[c])
                self.S_cluster[miembros] = score_c

        self.user_coords_2d = _proyectar_2d(self._U_pref_norm)

    def puntajes(self, usuario: Any, metodo: str = "combinado") -> np.ndarray:
        k = self.pos_usuario[usuario]
        if metodo == "popularidad":
            return self.popularidad.copy()
        if metodo == "contenido":
            return self.S_cont[k].copy()
        if metodo == "colaborativo":
            return self.S_colab[k].copy()
        if metodo == "clustering":
            return self.S_cluster[k].copy()
        c = normalizar(self.S_colab[[k]])[0]
        t = normalizar(self.S_cont[[k]])[0]
        return self.peso * c + (1.0 - self.peso) * t

    def tabla(self, orden: np.ndarray) -> pd.DataFrame:
        return self.items.iloc[orden][["id", "nombre"]].reset_index(drop=True)

    def _item_a_dict(
        self,
        idx: int,
        puntaje: float | None = None,
        afinidad: int | None = None,
        razones: list[str] | None = None,
    ) -> dict[str, Any]:
        fila = self.items.iloc[idx]
        item_id = fila["id"]
        if isinstance(item_id, (np.integer,)):
            item_id = int(item_id)
        cid = int(self.item_clusters[idx]) if idx < len(self.item_clusters) else 0
        data = {
            "id": item_id,
            "nombre": str(fila["nombre"]),
            "categoria": str(fila["categoria"]),
            "subcategoria": str(fila["subcategoria"]),
            "descripcion": str(fila["descripcion"]) or str(fila["texto"]),
            "etiquetas": _split_tags(fila["etiquetas"]),
            "tono": str(fila["tono"]),
            "presupuesto": str(fila["presupuesto"]),
            "duracion": str(fila["duracion"]),
            "compania": _split_tags(fila["compania"]),
            "icono": str(fila["icono"]),
            "cluster_id": cid,
            "cluster_nombre": self.item_cluster_names.get(cid, "Clúster general"),
            "media_calificacion": round(float(self.media_item[idx]), 2) if self.n_valoraciones[idx] > 0 else None,
            "num_valoraciones": int(self.n_valoraciones[idx]),
        }
        if puntaje is not None:
            data["puntaje"] = round(float(puntaje), 4)
        if afinidad is not None:
            data["afinidad"] = int(afinidad)
        if razones is not None:
            data["razones"] = razones
        return data

    def recomendar(self, usuario: Any, n: int = N_RECOMENDACIONES, metodo: str = "combinado") -> pd.DataFrame:
        if usuario not in self.pos_usuario:
            return self.populares(n)
        p = self.puntajes(usuario, metodo)
        p[self.R[self.pos_usuario[usuario]] > 0] = -np.inf
        return self.tabla(np.argsort(-p)[:n])

    def recomendar_detallado(
        self,
        usuario: Any,
        n: int = N_RECOMENDACIONES,
        metodo: str = "combinado",
        categoria: str | None = None,
    ) -> list[dict[str, Any]]:
        if usuario not in self.pos_usuario:
            return self.populares_detallado(n=n, categoria=categoria)

        p = self.puntajes(usuario, metodo)
        p_norm = _normalizar_vector(np.where(np.isfinite(p), p, 0.0))
        vistos = self.R[self.pos_usuario[usuario]] > 0
        p_norm[vistos] = -np.inf

        if categoria and categoria != "todo":
            mask_cat = (self.items["categoria"] == categoria).to_numpy()
            p_norm[~mask_cat] = -np.inf

        orden = np.argsort(-p_norm)
        validos = [int(i) for i in orden if np.isfinite(p_norm[i])][:n]
        resultados = []
        for idx in validos:
            score = float(p_norm[idx])
            afinidad = max(55, min(98, int(round(58 + score * 40))))
            razones = [
                f"Ajustado a tu historial ({metodo})",
            ]
            if self.S_colab.shape[0] > 0 and self.S_colab[self.pos_usuario[usuario], idx] > 0.15:
                razones.append("Muy valorado por personas con gustos afines")
            if self.media_item[idx] >= 4.0:
                razones.append(f"Valoración media de {self.media_item[idx]:.1f}/5")
            resultados.append(self._item_a_dict(idx, puntaje=score, afinidad=afinidad, razones=razones))
        return resultados

    def populares(self, n: int = N_RECOMENDACIONES) -> pd.DataFrame:
        return self.tabla(np.argsort(-self.popularidad)[:n])

    def populares_detallado(self, n: int = N_RECOMENDACIONES, categoria: str | None = None) -> list[dict[str, Any]]:
        pop = _normalizar_vector(self.popularidad)
        if categoria and categoria != "todo":
            pop = np.where(self.items["categoria"] == categoria, pop, -np.inf)
        orden = np.argsort(-pop)
        validos = [int(i) for i in orden if np.isfinite(pop[i])][:n]
        return [
            self._item_a_dict(
                idx,
                puntaje=float(pop[idx]),
                afinidad=max(60, min(96, int(round(62 + pop[idx] * 34)))),
                razones=["Entre los favoritos mejor valorados de la comunidad"],
            )
            for idx in validos
        ]

    def por_texto(self, texto: str, n: int = N_RECOMENDACIONES) -> pd.DataFrame:
        s = cosine_similarity(self.tfidf.transform([texto]), self.X)[0]
        return self.tabla(np.argsort(-s)[:n])

    def por_texto_detallado(self, texto: str, n: int = N_RECOMENDACIONES, categoria: str | None = None) -> list[dict[str, Any]]:
        s = cosine_similarity(self.tfidf.transform([texto]), self.X)[0]
        pop_norm = _normalizar_vector(self.popularidad)
        combinado = 0.9 * _normalizar_vector(s) + 0.1 * pop_norm
        if categoria and categoria != "todo":
            combinado = np.where(self.items["categoria"] == categoria, combinado, -np.inf)

        tokens_query = set(normalizar_texto(texto).split())
        orden = np.argsort(-combinado)
        validos = [int(i) for i in orden if np.isfinite(combinado[i])][:n]
        resultados = []
        for idx in validos:
            coincidencias = sorted(tokens_query & self._tokens_items[idx])
            razones = []
            if coincidencias:
                razones.append("Coincide con: " + ", ".join(coincidencias[:4]))
            else:
                razones.append("Similitud semántica con tu búsqueda")
            afinidad = max(52, min(98, int(round(52 + combinado[idx] * 46))))
            resultados.append(self._item_a_dict(idx, puntaje=float(combinado[idx]), afinidad=afinidad, razones=razones))
        return resultados

    def similares(self, id_item: Any, n: int = N_RECOMENDACIONES, devolver_detalles: bool = False) -> pd.DataFrame | list[dict[str, Any]]:
        if id_item not in self.pos_item:
            return [] if devolver_detalles else pd.DataFrame(columns=["id", "nombre"])
        k = self.pos_item[id_item]
        s = cosine_similarity(self.X[k], self.X)[0]
        s[k] = -np.inf
        orden = np.argsort(-s)[:n]
        if not devolver_detalles:
            return self.tabla(orden)

        tokens_base = self._tokens_items[k]
        cat_base = self.items.iloc[k]["categoria"]
        nombre_base = self.items.iloc[k]["nombre"]
        resultados = []
        for idx in orden:
            idx = int(idx)
            sim = float(max(0.0, s[idx]))
            # Bonus ligero si es de la misma categoría para similitud coherente
            bonus_cat = 0.08 if self.items.iloc[idx]["categoria"] == cat_base else 0.0
            score = min(1.0, sim + bonus_cat)
            comunes = sorted(tokens_base & self._tokens_items[idx])
            razones = [f"Afín a «{nombre_base}»"]
            if comunes:
                razones.append("Comparten rasgos: " + ", ".join(comunes[:4]))
            afinidad = max(55, min(97, int(round(56 + score * 41))))
            resultados.append(self._item_a_dict(idx, puntaje=score, afinidad=afinidad, razones=razones))
        return resultados

    def recomendar_por_cuestionario(
        self,
        categoria: str = "todo",
        referencias: list[str] | None = None,
        atributos: list[str] | None = None,
        animos: list[str] | None = None,
        gustos: list[str] | None = None,
        compania: str | None = None,
        duracion: str | None = None,
        presupuesto: str | None = None,
        evitar: list[str] | None = None,
        texto_libre: str = "",
        usuario: Any | None = None,
        metodo: str = "combinado",
        peso_colaborativo: float | None = None,
        n: int = N_RECOMENDACIONES,
    ) -> list[dict[str, Any]]:
        """Motor de afinidad: combina referencias, atributos de calidad, filtros y representaciones SVD/Clustering."""
        referencias = referencias or []
        atributos = atributos or []
        animos = animos or []
        gustos = gustos or []
        evitar = evitar or []
        n_items = len(self.items)
        if n_items == 0:
            return []

        if peso_colaborativo is not None:
            self.peso = float(peso_colaborativo)

        # Integra referencias y atributos tanto nuevos como heredados
        todos_gustos = gustos + atributos
        todos_animos = animos + referencias

        bloques_query = []
        if texto_libre.strip():
            bloques_query.extend([texto_libre.strip()] * 3)
        if referencias:
            bloques_query.extend([" ".join(referencias)] * 3)
        if todos_gustos:
            bloques_query.extend([" ".join(todos_gustos)] * 2)
        if todos_animos:
            bloques_query.extend([" ".join(todos_animos)] * 2)
        if compania and compania != "indiferente":
            bloques_query.append(compania)
        if duracion and duracion != "indiferente":
            bloques_query.append(duracion)
        if presupuesto and presupuesto != "indiferente":
            bloques_query.append(presupuesto)
        query_completa = " ".join(bloques_query).strip()

        if query_completa:
            q_vec = self.tfidf.transform([query_completa])
            sim_tfidf = cosine_similarity(q_vec, self.X)[0]
            sim_norm = _normalizar_vector(sim_tfidf)
            q_dense = normalize(q_vec.toarray())
            if self.item_centroids.size > 0:
                sim_clusters = cosine_similarity(q_dense, self.item_centroids)[0]
                cluster_pref = np.array([sim_clusters[c] for c in self.item_clusters], dtype=float)
                cluster_norm = _normalizar_vector(cluster_pref)
                cluster_top_id = int(np.argmax(sim_clusters))
            else:
                cluster_norm = np.zeros(n_items, dtype=float)
                cluster_top_id = None
        else:
            sim_tfidf = np.zeros(n_items, dtype=float)
            sim_norm = np.zeros(n_items, dtype=float)
            cluster_norm = np.zeros(n_items, dtype=float)
            cluster_top_id = None

        # Puntaje estructurado de metadatos + explicación transparente
        tokens_referencias = set(normalizar_texto(" ".join(referencias)).split())
        tokens_atributos = set(normalizar_texto(" ".join(atributos)).split())
        tokens_animos = set(normalizar_texto(" ".join(animos)).split())
        tokens_gustos = set(normalizar_texto(" ".join(gustos)).split())
        tokens_libre = set(normalizar_texto(texto_libre).split())
        tokens_evitar = set(normalizar_texto(" ".join(evitar)).split())

        meta_scores = np.zeros(n_items, dtype=float)
        penalizaciones = np.zeros(n_items, dtype=float)
        razones_por_idx: dict[int, list[str]] = {}

        for idx in range(n_items):
            fila = self.items.iloc[idx]
            tokens_item = self._tokens_items[idx]
            razones: list[str] = []
            bonus = 0.0

            if coinc_refs := sorted(tokens_referencias & tokens_item):
                bonus += min(0.40, 0.15 * len(coinc_refs))
                razones.append("Afinidad con tus referencias: " + ", ".join(coinc_refs[:3]))

            coinc_atribs = sorted((tokens_atributos | tokens_gustos) & tokens_item)
            if coinc_atribs:
                bonus += min(0.35, 0.12 * len(coinc_atribs))
                razones.append("Coincide con tus atributos clave: " + ", ".join(coinc_atribs[:3]))

            coinc_animos = sorted(tokens_animos & tokens_item)
            tono_tokens = set(normalizar_texto(fila["tono"]).split())
            if coinc_animos or (tokens_animos & tono_tokens):
                bonus += 0.20
                etiqueta_tono = fila["tono"] or ", ".join(coinc_animos[:2])
                razones.append(f"Tono y atmósfera afín ({etiqueta_tono})")

            if coinc_libre := sorted(tokens_libre & tokens_item):
                bonus += min(0.35, 0.14 * len(coinc_libre))
                razones.append("Conecta con tu descripción: " + ", ".join(coinc_libre[:3]))

            if compania and compania != "indiferente":
                companias_item = set(normalizar_texto(fila["compania"]).split())
                if normalizar_texto(compania) in companias_item:
                    bonus += 0.10
                    razones.append(f"Adaptado para {compania.replace('_', ' ')}")

            if duracion and duracion != "indiferente":
                if normalizar_texto(duracion) == normalizar_texto(fila["duracion"]):
                    bonus += 0.08

            if presupuesto and presupuesto != "indiferente":
                if normalizar_texto(presupuesto) == normalizar_texto(fila["presupuesto"]):
                    bonus += 0.08

            if tokens_evitar:
                coinc_evitar = tokens_evitar & tokens_item
                if coinc_evitar:
                    penalizaciones[idx] = 0.50 + 0.18 * len(coinc_evitar)

            meta_scores[idx] = bonus
            razones_por_idx[idx] = razones

        pop_norm = _normalizar_vector(self.popularidad)

        # Mezcla personalizada con SVD si el usuario existe en el historial
        user_norm = np.zeros(n_items, dtype=float)
        tiene_usuario = usuario is not None and usuario in self.pos_usuario
        if tiene_usuario:
            u_raw = self.puntajes(usuario, metodo)
            u_idx = self.pos_usuario[usuario]
            user_norm = _normalizar_vector(u_raw)
            # Penaliza ítems ya calificados negativamente, pero permite destacar nuevos
            ya_vistos = self.R[u_idx] > 0
            score_total = (
                0.44 * sim_norm
                + 0.23 * _normalizar_vector(meta_scores)
                + 0.18 * user_norm
                + 0.10 * cluster_norm
                + 0.05 * pop_norm
                - penalizaciones
            )
            score_total[ya_vistos] -= 0.65
        else:
            score_total = (
                0.46 * sim_norm
                + 0.32 * _normalizar_vector(meta_scores)
                + 0.12 * cluster_norm
                + 0.10 * pop_norm
                - penalizaciones
            )

        # Filtro por categoría si eligió una específica
        if categoria and categoria != "todo":
            mask = (self.items["categoria"] == categoria).to_numpy()
            score_total = np.where(mask, score_total, -np.inf)

        orden = np.argsort(-score_total)
        validos = [int(i) for i in orden if np.isfinite(score_total[i])]

        if categoria == "todo" and len(validos) > n:
            # Diversidad controlada entre categorías para recomendar "todo tipo de cosas"
            seleccionados: list[int] = []
            conteo_cat: dict[str, int] = {}
            max_por_cat = max(1, int(np.ceil(n / 4)))
            for idx in validos:
                cat = str(self.items.iloc[idx]["categoria"])
                if conteo_cat.get(cat, 0) < max_por_cat:
                    seleccionados.append(idx)
                    conteo_cat[cat] = conteo_cat.get(cat, 0) + 1
                    if len(seleccionados) == n:
                        break
            if len(seleccionados) < n:
                for idx in validos:
                    if idx not in seleccionados:
                        seleccionados.append(idx)
                        if len(seleccionados) == n:
                            break
        else:
            seleccionados = validos[:n]

        max_s = max([score_total[i] for i in seleccionados], default=1.0)
        min_s = min([score_total[i] for i in seleccionados], default=0.0)
        span = max(max_s - min_s, 1e-6)

        resultados = []
        for rank, idx in enumerate(seleccionados):
            raw_s = float(score_total[idx])
            rel = (raw_s - min_s) / span if len(seleccionados) > 1 else 0.92
            afinidad = max(62, min(98, int(round(72 + rel * 24 - rank * 0.8))))
            razones = razones_por_idx.get(idx, [])
            if tiene_usuario and self.S_colab.shape[0] > 0 and self.S_colab[self.pos_usuario[usuario], idx] > 0.15:
                razones.append("Recomendado también por el filtro colaborativo (SVD)")
            if cluster_top_id is not None and len(self.item_clusters) > idx and int(self.item_clusters[idx]) == cluster_top_id:
                razones.append(f"Pertenece a tu clúster afín: {self.item_cluster_names.get(cluster_top_id, '')}")
            if not razones:
                if self.media_item[idx] >= 4.0:
                    razones.append(f"Muy bien valorado ({self.media_item[idx]:.1f}/5)")
                else:
                    razones.append("Alta afinidad general con tu perfil de búsqueda")
            resultados.append(
                self._item_a_dict(idx, puntaje=raw_s, afinidad=afinidad, razones=razones[:3])
            )
        return resultados

    def analizar_clustering_consulta(self, texto_consulta: str, usuario: Any | None = None) -> dict[str, Any]:
        """Identifica a qué clúster temático y comunidad de usuarios corresponde una búsqueda."""
        resultado: dict[str, Any] = {"cluster_items": None, "comunidad_usuarios": None}
        limpio = texto_consulta.strip()
        if not limpio and (usuario is None or usuario not in self.pos_usuario):
            return resultado

        if limpio:
            q_dense = normalize(self.tfidf.transform([limpio]).toarray())
        else:
            u_idx = self.pos_usuario[usuario]
            q_dense = self._U_pref_norm[[u_idx]]

        if self.item_centroids.size > 0:
            sims_i = cosine_similarity(q_dense, self.item_centroids)[0]
            cid = int(np.argmax(sims_i))
            resultado["cluster_items"] = {
                "id": cid,
                "nombre": self.item_cluster_names.get(cid, f"Clúster {cid + 1}"),
                "palabras_clave": self.item_cluster_keywords.get(cid, []),
                "afinidad": max(55, min(99, int(round(62 + float(sims_i[cid]) * 37)))),
            }

        if self.user_centroids.size > 0:
            if usuario is not None and usuario in self.pos_usuario and not limpio:
                uid_c = int(self.user_clusters[self.pos_usuario[usuario]])
                afinidad_u = 95
            else:
                sims_u = cosine_similarity(q_dense, self.user_centroids)[0]
                uid_c = int(np.argmax(sims_u))
                afinidad_u = max(55, min(98, int(round(62 + float(sims_u[uid_c]) * 36))))

            miembros = [str(self.usuarios[i]) for i in np.where(self.user_clusters == uid_c)[0]]
            nombrados = [m for m in miembros if not m.startswith("Explorador_")]
            resultado["comunidad_usuarios"] = {
                "id": uid_c,
                "nombre": self.user_cluster_names.get(uid_c, f"Comunidad {uid_c + 1}"),
                "palabras_clave": self.user_cluster_keywords.get(uid_c, []),
                "afinidad": afinidad_u,
                "miembros_ejemplo": (nombrados or miembros)[:4],
                "total_miembros": len(miembros),
            }
        return resultado

    def comunidad_usuario(self, usuario: Any) -> dict[str, Any] | None:
        if usuario not in self.pos_usuario or self.k_users == 0:
            return None
        u_idx = self.pos_usuario[usuario]
        cid = int(self.user_clusters[u_idx])
        miembros = [str(self.usuarios[i]) for i in np.where(self.user_clusters == cid)[0] if str(self.usuarios[i]) != str(usuario)]
        nombrados = [m for m in miembros if not m.startswith("Explorador_")]
        return {
            "id": cid,
            "nombre": self.user_cluster_names.get(cid, f"Comunidad {cid + 1}"),
            "palabras_clave": self.user_cluster_keywords.get(cid, []),
            "total_miembros": len(miembros) + 1,
            "pares_afines": (nombrados or miembros)[:6],
        }

    def resumen_clustering(self, k_items: int | None = None, k_users: int | None = None) -> dict[str, Any]:
        """Devuelve la estructura completa de clústeres de ítems, comunidades de usuarios y puntos 2D."""
        ki = int(k_items) if k_items is not None else self.k_items
        ku = int(k_users) if k_users is not None else self.k_users
        if ki != self.k_items or ku != self.k_users:
            self.recalcular_clustering(k_items=ki, k_users=ku)

        clusters_items = []
        for c in range(self.k_items):
            idxs = np.where(self.item_clusters == c)[0]
            # Ordena ejemplos por popularidad dentro del clúster
            orden_pop = idxs[np.argsort(-self.popularidad[idxs])]
            ejemplos = [
                {
                    "id": self._item_a_dict(int(i))["id"],
                    "nombre": str(self.items.iloc[i]["nombre"]),
                    "categoria": str(self.items.iloc[i]["categoria"]),
                }
                for i in orden_pop[:6]
            ]
            clusters_items.append({
                "id": c,
                "nombre": self.item_cluster_names.get(c, f"Clúster {c + 1}"),
                "palabras_clave": self.item_cluster_keywords.get(c, []),
                "total_items": int(len(idxs)),
                "ejemplos": ejemplos,
            })

        puntos_items = []
        for i in range(len(self.items)):
            cid = int(self.item_clusters[i])
            item_id = self.ids[i]
            if isinstance(item_id, (np.integer,)):
                item_id = int(item_id)
            puntos_items.append({
                "id": item_id,
                "nombre": str(self.items.iloc[i]["nombre"]),
                "categoria": str(self.items.iloc[i]["categoria"]),
                "subcategoria": str(self.items.iloc[i]["subcategoria"]),
                "cluster_id": cid,
                "cluster_nombre": self.item_cluster_names.get(cid, ""),
                "x": round(float(self.item_coords_2d[i, 0]), 2),
                "y": round(float(self.item_coords_2d[i, 1]), 2),
            })

        clusters_usuarios = []
        for c in range(self.k_users):
            miembros_idx = np.where(self.user_clusters == c)[0]
            miembros = [str(self.usuarios[i]) for i in miembros_idx]
            nombrados = [m for m in miembros if not m.startswith("Explorador_")]
            top_items = []
            if len(miembros_idx) and len(self.ids):
                medias_c = self._M[miembros_idx].mean(axis=0)
                top_idx = np.argsort(-medias_c)[:4]
                top_items = [
                    {
                        "id": self._item_a_dict(int(ti))["id"],
                        "nombre": str(self.items.iloc[ti]["nombre"]),
                        "categoria": str(self.items.iloc[ti]["categoria"]),
                    }
                    for ti in top_idx
                ]
            clusters_usuarios.append({
                "id": c,
                "nombre": self.user_cluster_names.get(c, f"Comunidad {c + 1}"),
                "palabras_clave": self.user_cluster_keywords.get(c, []),
                "total_usuarios": int(len(miembros)),
                "usuarios": (nombrados + [m for m in miembros if m.startswith("Explorador_")])[:8],
                "top_items": top_items,
            })

        puntos_usuarios = []
        for u_idx, usuario in enumerate(self.usuarios):
            cid = int(self.user_clusters[u_idx])
            puntos_usuarios.append({
                "usuario": str(usuario),
                "cluster_id": cid,
                "cluster_nombre": self.user_cluster_names.get(cid, ""),
                "num_valoraciones": int((self.R[u_idx] > 0).sum()),
                "x": round(float(self.user_coords_2d[u_idx, 0]), 2),
                "y": round(float(self.user_coords_2d[u_idx, 1]), 2),
            })

        return {
            "k_items": self.k_items,
            "k_users": self.k_users,
            "clusters_items": clusters_items,
            "puntos_items": puntos_items,
            "clusters_usuarios": clusters_usuarios,
            "puntos_usuarios": puntos_usuarios,
        }


def dividir(interacciones: pd.DataFrame, proporcion: float = PROPORCION_PRUEBA, semilla: int = SEMILLA) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(semilla)
    mascara = rng.random(len(interacciones)) < proporcion
    return interacciones[~mascara].reset_index(drop=True), interacciones[mascara].reset_index(drop=True)


def precision_en_k(
    items: pd.DataFrame | str,
    interacciones: pd.DataFrame | str,
    k: int = 5,
    n_particiones: int = 10,
) -> dict[str, dict[str, float]]:
    """Evalúa Precisión@K en múltiples particiones aleatorias manteniendo compatibilidad con el Colab."""
    items_df, inter_df = preparar(items, interacciones)
    if len(inter_df) < 5 or len(items_df) < k:
        return {m: {"media": 0.0, "std": 0.0} for m in METODOS}

    resultados: dict[str, list[float]] = {m: [] for m in METODOS}
    for s in range(n_particiones):
        entrenamiento, prueba = dividir(inter_df, semilla=s)
        if len(entrenamiento) == 0 or len(prueba) == 0:
            continue
        modelo = Recomendador(items_df, entrenamiento)
        relevantes = (
            prueba[prueba["calificacion"] >= UMBRAL_GUSTO]
            .groupby("usuario")["item"]
            .apply(set)
            .to_dict()
        )
        for metodo in METODOS:
            aciertos = []
            for u, rel in relevantes.items():
                if u in modelo.pos_usuario and len(rel) > 0:
                    recs = set(modelo.recomendar(u, k, metodo)["id"])
                    aciertos.append(len(recs & rel) / k)
            if aciertos:
                resultados[metodo].append(float(np.mean(aciertos)) * 100.0)

    resumen: dict[str, dict[str, float]] = {}
    for m, vals in resultados.items():
        if vals:
            resumen[m] = {"media": round(float(np.mean(vals)), 2), "std": round(float(np.std(vals)), 2)}
        else:
            resumen[m] = {"media": 0.0, "std": 0.0}
    return resumen


def crear_dataset(
    carpeta: str,
    nombre_dominio: str,
    categorias: list[str],
    palabras: dict[str, list[str]],
    n_usuarios: int,
    n_items_por_cat: int,
    semilla: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(semilla)
    os.makedirs(carpeta, exist_ok=True)
    filas_items = []
    cat_item = {}
    for c, cat in enumerate(categorias):
        for j in range(n_items_por_cat):
            idx = c * n_items_por_cat + j + 1
            propias = rng.choice(palabras[cat], size=min(4, len(palabras[cat])), replace=False).tolist()
            ajenas = rng.choice(palabras[categorias[(c + 1) % len(categorias)]], size=1).tolist()
            filas_items.append({"id": idx, "nombre": f"{nombre_dominio} {idx} ({cat})", "texto": " ".join(propias + ajenas)})
            cat_item[idx] = c
    items = pd.DataFrame(filas_items)

    filas = []
    for u in range(1, n_usuarios + 1):
        favorita = rng.integers(len(categorias))
        vistos = rng.choice(items["id"], size=max(1, int(len(items) * 0.45)), replace=False)
        for i in vistos:
            base = 4.4 if cat_item[i] == favorita else 2.0
            v = int(np.clip(round(base + rng.normal(0, 0.9)), 1, 5))
            filas.append({"usuario": u, "item": int(i), "calificacion": v})
    interacciones = pd.DataFrame(filas)

    items.to_csv(f"{carpeta}/items.csv", index=False)
    interacciones.to_csv(f"{carpeta}/interacciones.csv", index=False)
    return items, interacciones

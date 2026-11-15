from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.catalogos import DOMINIOS_META, inicializar_datos
from backend.motor import METODOS, Recomendador, precision_en_k
from backend.storage import AlmacenDatos

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CUSTOM_DIR = DATA_DIR / "custom"
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title="Criterio — Sistema de Recomendación por Afinidad",
    description="Motor de recomendación cultural y de ocio basado en modelado de preferencias y afinidades",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CuestionarioRequest(BaseModel):
    categoria: str = "todo"
    referencias: list[str] = Field(default_factory=list)
    atributos: list[str] = Field(default_factory=list)
    animos: list[str] = Field(default_factory=list)
    gustos: list[str] = Field(default_factory=list)
    compania: str = "indiferente"
    duracion: str = "indiferente"
    presupuesto: str = "indiferente"
    evitar: list[str] = Field(default_factory=list)
    texto_libre: str = ""
    usuario: str | None = None
    guardar_perfil: bool = False
    metodo: str = "combinado"
    peso_colaborativo: float = Field(default=0.5, ge=0.0, le=1.0)
    cantidad: int = Field(default=6, ge=1, le=24)


class InteraccionRequest(BaseModel):
    usuario: str = Field(min_length=1, max_length=80)
    item_id: int | str
    calificacion: float = Field(ge=1.0, le=5.0)


class EvaluacionRequest(BaseModel):
    dominio: str = "todo"
    k: int = Field(default=5, ge=1, le=15)
    n_particiones: int = Field(default=8, ge=2, le=30)


class EstadoMotor:
    def __init__(self) -> None:
        self.almacen = AlmacenDatos(str(DATA_DIR))
        self.modo = "maestro"
        self.items_maestro, self.inter_maestro = inicializar_datos(str(DATA_DIR))
        self.modelo_maestro = Recomendador(self.items_maestro, self.inter_maestro)
        self.items_custom: pd.DataFrame | None = None
        self.inter_custom: pd.DataFrame | None = None
        self.modelo_custom: Recomendador | None = None

    @property
    def modelo_activo(self) -> Recomendador:
        if self.modo == "custom" and self.modelo_custom is not None:
            return self.modelo_custom
        return self.modelo_maestro

    @property
    def ruta_inter_activa(self) -> str:
        if self.modo == "custom":
            os.makedirs(CUSTOM_DIR, exist_ok=True)
            return str(CUSTOM_DIR / "interacciones.csv")
        return str(DATA_DIR / "interacciones.csv")

    def recargar_activo(self, items_df: pd.DataFrame, inter_df: pd.DataFrame) -> None:
        if self.modo == "custom":
            self.items_custom = items_df
            self.inter_custom = inter_df
            self.modelo_custom = Recomendador(items_df, inter_df)
        else:
            self.items_maestro = items_df
            self.inter_maestro = inter_df
            self.modelo_maestro = Recomendador(items_df, inter_df)


estado = EstadoMotor()


@app.get("/api/estado")
def obtener_estado() -> dict[str, Any]:
    modelo = estado.modelo_activo
    conteos = modelo.items["categoria"].value_counts().to_dict()
    dominios = []
    if estado.modo == "maestro":
        for clave, meta in DOMINIOS_META.items():
            total = len(modelo.items) if clave == "todo" else int(conteos.get(clave, 0))
            dominios.append({
                "id": clave,
                "nombre": meta["nombre"],
                "subtitulo": meta["subtitulo"],
                "icono": meta["icono"],
                "total": total,
            })
    else:
        dominios.append({
            "id": "todo",
            "nombre": "Catálogo personalizado (CSV)",
            "subtitulo": "Archivo cargado por el usuario",
            "icono": "sparkles",
            "total": len(modelo.items),
        })
        for cat, cnt in conteos.items():
            if cat != "general":
                dominios.append({
                    "id": str(cat),
                    "nombre": str(cat).capitalize(),
                    "subtitulo": "Categoría detectada en tu CSV",
                    "icono": "sparkles",
                    "total": int(cnt),
                })

    perfiles_guardados = estado.almacen.cargar_perfiles()
    usuarios_ordenados = sorted(
        set([str(u) for u in modelo.usuarios] + list(perfiles_guardados.keys())),
        key=lambda x: (x.startswith("Explorador_"), x.lower()),
    )

    return {
        "modo": estado.modo,
        "dominios": dominios,
        "usuarios": usuarios_ordenados,
        "perfiles_guardados": list(perfiles_guardados.keys()),
        "metodos": METODOS,
        "estadisticas": {
            "total_items": len(modelo.items),
            "total_usuarios": len(modelo.usuarios),
            "total_interacciones": len(modelo.interacciones),
            "total_categorias": int(modelo.items["categoria"].nunique()),
            "k_items": modelo.k_items,
            "k_users": modelo.k_users,
        },
    }


@app.get("/api/clustering")
def obtener_clustering(
    k_items: int = Query(default=8, ge=1, le=16),
    k_users: int = Query(default=5, ge=1, le=12),
) -> dict[str, Any]:
    modelo = estado.modelo_activo
    return modelo.resumen_clustering(k_items=k_items, k_users=k_users)


@app.get("/api/catalogo")
def explorar_catalogo(
    categoria: str = "todo",
    q: str = "",
    limit: int = Query(default=24, ge=1, le=200),
) -> dict[str, Any]:
    modelo = estado.modelo_activo
    if q.strip():
        resultados = modelo.por_texto_detallado(
            q.strip(),
            n=limit,
            categoria=None if categoria == "todo" else categoria,
        )
        return {"items": resultados, "total": len(resultados)}

    df = modelo.items
    if categoria != "todo":
        df = df[df["categoria"] == categoria]
    indices = df.index.tolist()[:limit]
    items = [modelo._item_a_dict(int(idx)) for idx in indices]
    return {"items": items, "total": len(df)}


@app.post("/api/cuestionario/recomendar")
def recomendar_cuestionario(req: CuestionarioRequest) -> dict[str, Any]:
    modelo = estado.modelo_activo
    usuario_id = req.usuario.strip() if req.usuario and req.usuario.strip() else None

    if usuario_id and req.guardar_perfil:
        estado.almacen.guardar_perfil(
            usuario_id,
            {
                "categoria": req.categoria,
                "referencias": req.referencias,
                "atributos": req.atributos,
                "animos": req.animos,
                "gustos": req.gustos,
                "compania": req.compania,
                "duracion": req.duracion,
                "presupuesto": req.presupuesto,
                "evitar": req.evitar,
                "texto_libre": req.texto_libre,
            },
        )

    resultados = modelo.recomendar_por_cuestionario(
        categoria=req.categoria,
        referencias=req.referencias,
        atributos=req.atributos,
        animos=req.animos,
        gustos=req.gustos,
        compania=req.compania,
        duracion=req.duracion,
        presupuesto=req.presupuesto,
        evitar=req.evitar,
        texto_libre=req.texto_libre,
        usuario=usuario_id,
        metodo=req.metodo if req.metodo in METODOS else "combinado",
        peso_colaborativo=req.peso_colaborativo,
        n=req.cantidad,
    )
    partes_consulta = [req.texto_libre.strip()]
    if req.referencias:
        partes_consulta.append(" ".join(req.referencias))
    if req.atributos:
        partes_consulta.append(" ".join(req.atributos))
    if req.gustos:
        partes_consulta.append(" ".join(req.gustos))
    if req.animos:
        partes_consulta.append(" ".join(req.animos))
    consulta_clustering = " ".join([p for p in partes_consulta if p]).strip()
    clustering_match = modelo.analizar_clustering_consulta(consulta_clustering, usuario=usuario_id)
    return {
        "recomendaciones": resultados,
        "clustering_match": clustering_match,
        "usuario": usuario_id,
        "categoria": req.categoria,
    }


@app.get("/api/usuarios/{usuario}/recomendar")
def recomendar_por_usuario(
    usuario: str,
    metodo: str = "combinado",
    peso_colaborativo: float = Query(default=0.5, ge=0.0, le=1.0),
    cantidad: int = Query(default=6, ge=1, le=24),
    categoria: str = "todo",
) -> dict[str, Any]:
    modelo = estado.modelo_activo
    modelo.peso = float(peso_colaborativo)
    m = metodo if metodo in METODOS else "combinado"
    resultados = modelo.recomendar_detallado(
        usuario=usuario,
        n=cantidad,
        metodo=m,
        categoria=None if categoria == "todo" else categoria,
    )
    return {"usuario": usuario, "metodo": m, "recomendaciones": resultados}


@app.get("/api/usuarios/{usuario}/perfil")
def obtener_perfil_usuario(usuario: str) -> dict[str, Any]:
    modelo = estado.modelo_activo
    perfil = estado.almacen.obtener_perfil(usuario)
    inter = modelo.interacciones[modelo.interacciones["usuario"].astype(str) == str(usuario)]
    historial = []
    for _, row in inter.iterrows():
        item_id = row["item"]
        if item_id in modelo.pos_item:
            item_dict = modelo._item_a_dict(modelo.pos_item[item_id])
            item_dict["calificacion_usuario"] = float(row["calificacion"])
            historial.append(item_dict)
    historial.sort(key=lambda x: x["calificacion_usuario"], reverse=True)
    return {
        "usuario": usuario,
        "perfil": perfil,
        "comunidad": modelo.comunidad_usuario(usuario),
        "historial": historial,
        "total_valoraciones": len(historial),
    }


@app.post("/api/interacciones")
def registrar_interaccion(req: InteraccionRequest) -> dict[str, Any]:
    modelo = estado.modelo_activo
    item_id: int | str = req.item_id
    if item_id not in modelo.pos_item:
        try:
            as_int = int(item_id)
            if as_int in modelo.pos_item:
                item_id = as_int
        except (ValueError, TypeError):
            pass

    if item_id not in modelo.pos_item:
        raise HTTPException(status_code=404, detail="Ítem no encontrado en el catálogo activo.")

    usuario = req.usuario.strip()
    inter_actual = estado.inter_custom if estado.modo == "custom" and estado.inter_custom is not None else estado.inter_maestro
    items_actual = estado.items_custom if estado.modo == "custom" and estado.items_custom is not None else estado.items_maestro

    nuevo_inter = estado.almacen.registrar_interaccion_csv(
        estado.ruta_inter_activa,
        inter_actual,
        usuario=usuario,
        item_id=item_id,
        calificacion=req.calificacion,
    )
    estado.recargar_activo(items_actual, nuevo_inter)
    return {
        "ok": True,
        "usuario": usuario,
        "item_id": item_id,
        "calificacion": req.calificacion,
        "total_interacciones": len(nuevo_inter),
    }


@app.get("/api/items/{item_id}/similares")
def obtener_similares(item_id: str, cantidad: int = Query(default=6, ge=1, le=20)) -> dict[str, Any]:
    modelo = estado.modelo_activo
    clave: int | str = item_id
    if clave not in modelo.pos_item:
        try:
            as_int = int(item_id)
            if as_int in modelo.pos_item:
                clave = as_int
        except (ValueError, TypeError):
            pass
    if clave not in modelo.pos_item:
        raise HTTPException(status_code=404, detail="Ítem no encontrado.")

    base = modelo._item_a_dict(modelo.pos_item[clave])
    similares = modelo.similares(clave, n=cantidad, devolver_detalles=True)
    return {"item": base, "similares": similares}


@app.post("/api/evaluacion")
def evaluar_motor(req: EvaluacionRequest) -> dict[str, Any]:
    modelo = estado.modelo_activo
    items_df = modelo.items
    inter_df = modelo.interacciones

    if req.dominio != "todo" and estado.modo == "maestro":
        items_df = items_df[items_df["categoria"] == req.dominio].copy()
        inter_df = inter_df[inter_df["item"].isin(items_df["id"])].copy()

    metricas = precision_en_k(items_df, inter_df, k=req.k, n_particiones=req.n_particiones)
    return {
        "dominio": req.dominio,
        "k": req.k,
        "n_particiones": req.n_particiones,
        "total_items": len(items_df),
        "total_interacciones": len(inter_df),
        "metricas": metricas,
    }


@app.post("/api/csv/subir")
async def subir_csv(
    items_csv: UploadFile = File(...),
    interacciones_csv: UploadFile | None = File(default=None),
) -> dict[str, Any]:
    try:
        items_bytes = await items_csv.read()
        items_df = pd.read_csv(io.BytesIO(items_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error leyendo items.csv: {exc}") from exc

    if "id" not in items_df.columns and "item" not in items_df.columns:
        raise HTTPException(status_code=400, detail="El archivo items.csv debe tener una columna 'id' o 'item'.")
    if "nombre" not in items_df.columns or "texto" not in items_df.columns:
        raise HTTPException(status_code=400, detail="El archivo items.csv debe tener las columnas 'nombre' y 'texto'.")

    if interacciones_csv is not None:
        try:
            inter_bytes = await interacciones_csv.read()
            inter_df = pd.read_csv(io.BytesIO(inter_bytes))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Error leyendo interacciones.csv: {exc}") from exc
    else:
        inter_df = pd.DataFrame(columns=["usuario", "item", "calificacion"])

    os.makedirs(CUSTOM_DIR, exist_ok=True)
    items_df.to_csv(CUSTOM_DIR / "items.csv", index=False)
    inter_df.to_csv(CUSTOM_DIR / "interacciones.csv", index=False)

    estado.modo = "custom"
    estado.recargar_activo(items_df, inter_df)

    return {
        "ok": True,
        "total_items": len(estado.modelo_activo.items),
        "total_usuarios": len(estado.modelo_activo.usuarios),
        "total_interacciones": len(estado.modelo_activo.interacciones),
    }


@app.post("/api/csv/restaurar")
def restaurar_catalogo() -> dict[str, Any]:
    estado.modo = "maestro"
    return {"ok": True, "modo": "maestro"}


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def servir_inicio() -> FileResponse:
    response = FileResponse(str(FRONTEND_DIR / "index.html"))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

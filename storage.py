from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd


class AlmacenDatos:
    """Gestiona la persistencia de interacciones y perfiles de cuestionario en disco."""

    def __init__(self, carpeta_datos: str):
        self.carpeta_datos = carpeta_datos
        os.makedirs(self.carpeta_datos, exist_ok=True)
        self.ruta_perfiles = os.path.join(self.carpeta_datos, "perfiles.json")

    def cargar_perfiles(self) -> dict[str, dict[str, Any]]:
        if not os.path.exists(self.ruta_perfiles):
            return {}
        try:
            with open(self.ruta_perfiles, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def guardar_perfil(self, nombre: str, preferencias: dict[str, Any]) -> dict[str, Any]:
        nombre_limpio = nombre.strip()
        if not nombre_limpio:
            return {}
        perfiles = self.cargar_perfiles()
        ahora = datetime.now(timezone.utc).isoformat()
        existente = perfiles.get(nombre_limpio, {})
        perfiles[nombre_limpio] = {
            "nombre": nombre_limpio,
            "creado_en": existente.get("creado_en", ahora),
            "actualizado_en": ahora,
            "preferencias": preferencias,
        }
        with open(self.ruta_perfiles, "w", encoding="utf-8") as f:
            json.dump(perfiles, f, ensure_ascii=False, indent=2)
        return perfiles[nombre_limpio]

    def obtener_perfil(self, nombre: str) -> dict[str, Any] | None:
        return self.cargar_perfiles().get(nombre)

    def registrar_interaccion_csv(
        self,
        ruta_csv: str,
        interacciones_df: pd.DataFrame,
        usuario: str,
        item_id: Any,
        calificacion: float,
    ) -> pd.DataFrame:
        df = interacciones_df.copy()
        mascara = (df["usuario"].astype(str) == str(usuario)) & (df["item"].astype(str) == str(item_id))
        if mascara.any():
            df.loc[mascara, "calificacion"] = float(calificacion)
        else:
            nueva = pd.DataFrame([{"usuario": usuario, "item": item_id, "calificacion": float(calificacion)}])
            df = pd.concat([df, nueva], ignore_index=True)
        df.to_csv(ruta_csv, index=False)
        return df

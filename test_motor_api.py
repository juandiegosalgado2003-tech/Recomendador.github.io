from __future__ import annotations

import io
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend import main as main_mod
from backend.catalogos import inicializar_datos, obtener_catalogo_base
from backend.motor import METODOS, Recomendador, crear_dataset, precision_en_k
from backend.storage import AlmacenDatos


@pytest.fixture()
def entorno_aislado(tmp_path, monkeypatch):
    """Aísla DATA_DIR y EstadoMotor en un directorio temporal por prueba."""
    data_dir = tmp_path / "data"
    custom_dir = data_dir / "custom"
    monkeypatch.setattr(main_mod, "DATA_DIR", data_dir)
    monkeypatch.setattr(main_mod, "CUSTOM_DIR", custom_dir)

    nuevo_estado = main_mod.EstadoMotor()
    monkeypatch.setattr(main_mod, "estado", nuevo_estado)
    return TestClient(main_mod.app)


def test_compatibilidad_colab_motor(tmp_path):
    items, inter = crear_dataset(
        str(tmp_path / "ds"),
        "Libro",
        ["ciencia", "cocina"],
        {
            "ciencia": ["fisica", "universo", "evolucion", "quimica", "energia"],
            "cocina": ["receta", "horno", "postres", "especias", "sabor"],
        },
        n_usuarios=12,
        n_items_por_cat=6,
        semilla=42,
    )
    modelo = Recomendador(items, inter)
    assert len(modelo.recomendar(1, n=4)) == 4
    assert len(modelo.por_texto("receta horno postres", n=4)) == 4
    assert len(modelo.similares(1, n=4)) == 4

    metricas = precision_en_k(items, inter, k=4, n_particiones=4)
    assert set(metricas.keys()) == set(METODOS)


def test_cuestionario_multidominio_y_filtro_negativo():
    items_df = pd.DataFrame(obtener_catalogo_base())
    modelo = Recomendador(items_df, pd.DataFrame(columns=["usuario", "item", "calificacion"]))

    # Modo "todo": debe diversificar categorías
    mix = modelo.recomendar_por_cuestionario(
        categoria="todo",
        animos=["relajado calma cafe"],
        gustos=["ciencia ficcion espacio naturaleza"],
        n=8,
    )
    assert len(mix) == 8
    categorias_devueltas = {item["categoria"] for item in mix}
    assert len(categorias_devueltas) >= 3

    # Filtro negativo en películas: evitar terror
    pelis_sin_terror = modelo.recomendar_por_cuestionario(
        categoria="peliculas",
        animos=["oscuro misterioso"],
        evitar=["terror miedo sustos horror maldicion"],
        n=5,
    )
    assert len(pelis_sin_terror) == 5
    nombres_sin_terror = [p["nombre"] for p in pelis_sin_terror]
    assert "El legado del diablo (Hereditary)" not in nombres_sin_terror


def test_api_flujo_completo(entorno_aislado):
    client = entorno_aislado

    r_home = client.get("/")
    assert r_home.status_code == 200
    assert "Criterio" in r_home.text

    r_estado = client.get("/api/estado")
    assert r_estado.status_code == 200
    estado = r_estado.json()
    assert estado["modo"] == "maestro"
    assert estado["estadisticas"]["total_items"] >= 80
    assert estado["estadisticas"]["total_categorias"] == 8

    # Recomendación por cuestionario + guardar perfil
    r_quiz = client.post(
        "/api/cuestionario/recomendar",
        json={
            "categoria": "todo",
            "animos": ["curioso aprender ingenio"],
            "gustos": ["espacio ciencia ficcion cafe"],
            "compania": "solo",
            "duracion": "medio",
            "presupuesto": "economico",
            "evitar": ["terror sustos"],
            "texto_libre": "Me gusta Interestelar y el café artesanal",
            "usuario": "ValeriaTest",
            "guardar_perfil": True,
            "cantidad": 6,
        },
    )
    assert r_quiz.status_code == 200
    quiz_json = r_quiz.json()
    recs = quiz_json["recomendaciones"]
    assert len(recs) == 6
    assert all(60 <= item["afinidad"] <= 99 for item in recs)
    assert all(len(item["razones"]) >= 1 for item in recs)
    assert quiz_json["clustering_match"]["cluster_items"] is not None
    assert quiz_json["clustering_match"]["comunidad_usuarios"] is not None

    primer_id = recs[0]["id"]

    # Registrar valoración y comprobar aprendizaje en perfil
    r_rate = client.post(
        "/api/interacciones",
        json={"usuario": "ValeriaTest", "item_id": primer_id, "calificacion": 5},
    )
    assert r_rate.status_code == 200

    r_perfil = client.get("/api/usuarios/ValeriaTest/perfil")
    assert r_perfil.status_code == 200
    perfil_data = r_perfil.json()
    assert perfil_data["total_valoraciones"] == 1
    assert perfil_data["perfil"]["nombre"] == "ValeriaTest"
    assert perfil_data["comunidad"] is not None

    # Recomendación usando método clustering
    r_rec_cluster = client.get("/api/usuarios/ValeriaTest/recomendar?metodo=clustering&cantidad=4")
    assert r_rec_cluster.status_code == 200
    assert len(r_rec_cluster.json()["recomendaciones"]) == 4

    # Endpoint de Clustering K-Means
    r_clust = client.get("/api/clustering?k_items=6&k_users=4")
    assert r_clust.status_code == 200
    clust_data = r_clust.json()
    assert len(clust_data["clusters_items"]) == 6
    assert len(clust_data["clusters_usuarios"]) == 4
    assert len(clust_data["puntos_items"]) == estado["estadisticas"]["total_items"]
    assert all(0.0 <= p["x"] <= 100.0 and 0.0 <= p["y"] <= 100.0 for p in clust_data["puntos_items"])

    # Similares
    r_sim = client.get(f"/api/items/{primer_id}/similares?cantidad=4")
    assert r_sim.status_code == 200
    assert len(r_sim.json()["similares"]) == 4

    # Evaluación Precisión@K
    r_eval = client.post(
        "/api/evaluacion",
        json={"dominio": "peliculas", "k": 4, "n_particiones": 3},
    )
    assert r_eval.status_code == 200
    assert set(r_eval.json()["metricas"].keys()) == set(METODOS)


def test_subida_y_restauracion_csv(entorno_aislado):
    client = entorno_aislado

    items_csv = io.BytesIO(
        b"id,nombre,texto\n"
        b"1,Cafe Espresso,cafe intenso amargo manana energia\n"
        b"2,Te Matcha,te verde relajado calma japon\n"
        b"3,Chocolate Caliente,dulce cacao caliente invierno\n"
    )
    inter_csv = io.BytesIO(
        b"usuario,item,calificacion\n"
        b"Ana,1,5\n"
        b"Ana,2,2\n"
    )

    r_up = client.post(
        "/api/csv/subir",
        files={
            "items_csv": ("items.csv", items_csv, "text/csv"),
            "interacciones_csv": ("interacciones.csv", inter_csv, "text/csv"),
        },
    )
    assert r_up.status_code == 200
    assert r_up.json()["total_items"] == 3

    r_cat = client.get("/api/catalogo?q=te%20verde%20calma")
    assert r_cat.status_code == 200
    assert r_cat.json()["items"][0]["nombre"] == "Te Matcha"

    r_rest = client.post("/api/csv/restaurar")
    assert r_rest.status_code == 200
    assert client.get("/api/estado").json()["modo"] == "maestro"

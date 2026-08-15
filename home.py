import geopandas as gpd
import numpy as np
import pandas as pd
import pydeck as pdk
import shapely
import streamlit as st

from joblib import load
from shapely.geometry.polygon import orient

from notebooks.src.config import DADOS_GEO_MEDIAN, DADOS_LIMPOS, MODELO_FINAL


@st.cache_data
def carregar_dados_limpos():
    return pd.read_parquet(DADOS_LIMPOS)


@st.cache_data
def carregar_dados_geo():
    gdf_geo = gpd.read_parquet(DADOS_GEO_MEDIAN)

    # Explode MultiPolygons into individual polygons
    gdf_geo = gdf_geo.explode(ignore_index=True)

    # Function to check and fix invalid geometries
    def fix_and_orient_geometry(geometry):
        if not geometry.is_valid:
            geometry = geometry.buffer(0)

        # Orient the polygon to be counter-clockwise
        if isinstance(
            geometry,
            (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)
        ):
            geometry = orient(geometry, sign=1.0)

        return geometry

    # Apply the fix and orientation function to geometries
    gdf_geo["geometry"] = gdf_geo["geometry"].apply(
        fix_and_orient_geometry
    )

    # Extract polygon coordinates
    def get_polygon_coordinates(geometry):
        if isinstance(geometry, shapely.geometry.Polygon):
            return [
                [x, y]
                for x, y in geometry.exterior.coords
            ]

        else:
            return [
                [
                    [x, y]
                    for x, y in polygon.exterior.coords
                ]
                for polygon in geometry.geoms
            ]

    # Create a new column with polygon coordinates
    gdf_geo["geometry_coords"] = gdf_geo["geometry"].apply(
        get_polygon_coordinates
    )

    return gdf_geo


@st.cache_resource
def carregar_modelo():
    return load(MODELO_FINAL)


# Carregar dados e modelo
df = carregar_dados_limpos()
gdf_geo = carregar_dados_geo()
modelo = carregar_modelo()


st.title("Previsão de Preços de Imóveis")


condados = sorted(gdf_geo["name"].unique())


coluna1, coluna2 = st.columns(2)


with coluna1:

    with st.form (key="formulario"):

        selecionar_condado = st.selectbox(
            "Condado",
            condados
        )

        # Selecionar dados do condado
        dados_condado = gdf_geo.query(
            "name == @selecionar_condado"
        ).iloc[0]

        longitude = dados_condado["longitude"]
        latitude = dados_condado["latitude"]

        total_rooms = dados_condado["total_rooms"]
        total_bedrooms = dados_condado["total_bedrooms"]
        population = dados_condado["population"]
        households = dados_condado["households"]

        ocean_proximity = dados_condado["ocean_proximity"]

        rooms_per_household = dados_condado["rooms_per_household"]
        bedrooms_per_room = dados_condado["bedrooms_per_room"]
        population_per_household = dados_condado[
            "population_per_household"
        ]


        housing_median_age = st.number_input(
            "Idade do Imóvel",
            value=10,
            min_value=1,
            max_value=50
        )


        median_income = st.slider(
            "Renda Média (milhares de US$)",
            5.0,
            100.0,
            45.0,
            5.0
        )


        bins_income = [0, 1.5, 3, 4.5, 6, np.inf]

        median_income_cat = np.digitize(
            median_income / 10,
            bins=bins_income
        )


        entrada_modelo = {
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms,
            "population": population,
            "households": households,
            "median_income": median_income / 10,
            "ocean_proximity": ocean_proximity,
            "median_income_cat": median_income_cat,
            "rooms_per_household": rooms_per_household,
            "bedrooms_per_room": bedrooms_per_room,
            "population_per_household": population_per_household,
        }


        df_entrada_modelo = pd.DataFrame(
            [entrada_modelo]
        )


        botao_previsao = st.form_submit_button("Prever Preço")


    if botao_previsao:

        preco = modelo.predict(
            df_entrada_modelo
        )

        st.metric(label="Preço Previsto: (US$)", value=f"{preco[0][0]:.2f}")

with coluna2:

    view_state = pdk.ViewState(
        latitude=float(latitude),
        longitude=float(longitude),
        zoom=5,
        min_zoom=5,
        max_zoom=15,
    )


    polygon_layer = pdk.Layer(
        "PolygonLayer",
        data=gdf_geo[
            ["name", "geometry_coords"]
        ],
        get_polygon="geometry_coords",
        get_fill_color=[0, 0, 255, 100],
        get_line_color=[255, 255, 255],
        get_line_width=50,
        pickable=True,
        auto_highlight=True,
    )


    condado_selecionado = gdf_geo.query(
        "name == @selecionar_condado"
    )


    highlight_layer = pdk.Layer(
        "PolygonLayer",
        data=condado_selecionado[
            ["name", "geometry_coords"]
        ],
        get_polygon="geometry_coords",
        get_fill_color=[255, 0, 0, 180],
        get_line_color=[0, 0, 0],
        get_line_width=500,
        pickable=True,
        auto_highlight=True,
    )

    tooltip = {
        "html": "<b>Condado: </b> {name}",
        "style": {"backgroundColor": "steelblue", "color": "white", "fontsize": "10px"}
    }

    mapa = pdk.Deck(
        initial_view_state=view_state,
        map_style="light",
        layers=[
            polygon_layer,
            highlight_layer,
        ],
        tooltip=tooltip,
    )


    st.pydeck_chart(mapa)
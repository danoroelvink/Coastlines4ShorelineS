import dask
import geopandas as gpd
import pandas as pd

dask.config.set({"dataframe.query-planning": False})

import dask_geopandas
import pystac
from coastmonitor.io.utils import read_items_extent
from dask.dataframe.utils import make_meta
from shapely.geometry import LineString


def filter_sp(sp_raw):
    """
    Function to filter shorelines with specified filtering indicator.
    """

    # set up the indicators
    # `sp_clean` will include only shoreline positions that satisfy the indicators
    sp_clean = sp_raw[
        (sp_raw.sh_sinuosity < 10)
        & (~sp_raw.obs_on_shoal)
        & (sp_raw.obs_is_primary)
        & (sp_raw.tr_is_qa)
        & (sp_raw.mdn_offset < 3 * sp_raw.tr_stdev)
        & (sp_raw.obs_count >= 5)
        & (~sp_raw.obs_is_outlier)
    ].copy()

    # set up the `sp_clean` table
    sp_clean = (
        sp_clean[
            [
                "time",
                "transect_id",
                "lon",
                "lat",
                "shoreline_position_trans",
                "geometry",
            ]
        ]  # columns to be included in the clean tables
        .rename(columns=({"shoreline_position_trans": "shoreline_position"}))
        .reset_index(drop=True)
    )

    return sp_clean


def shoreline_intersections_to_coastline(df):
    # Ensure df is sorted if not already
    df = df.sort_values(by=["transect_id", "segment_id", "transect_dist"])

    # Identify partitions by checking where the difference in transect_dist is not 100
    # diff() is NaN for the first row, so we use fillna() to set it to a value that does not equal 100 (e.g., 0)
    df["partition"] = (df["transect_dist"].diff().fillna(0) != 100).cumsum()

    lines = []
    for _, partition_df in df.groupby("partition"):
        if len(partition_df) > 1:
            coords = gpd.GeoSeries.from_xy(
                partition_df["lon"], partition_df["lat"]
            ).to_list()

            lines.append(LineString(coords))
        # Else case can be added if needed to handle single-point partitions

    return pd.Series(lines)


def transect_origins_to_coastline(df):
    # Ensure df is sorted if not already
    df = df.sort_values(by=["transect_id", "segment_id", "transect_dist"])

    # Identify partitions by checking where the difference in transect_dist is not 100
    # diff() is NaN for the first row, so we use fillna() to set it to a value that does not equal 100 (e.g., 0)
    df["partition"] = (df["transect_dist"].diff().fillna(0) != 100).cumsum()

    lines = []
    for _, partition_df in df.groupby("partition"):
        if len(partition_df) > 1:
            coords = gpd.GeoSeries.from_xy(
                partition_df["lon"], partition_df["lat"]
            ).to_list()

            # Check if the coastline is closed and this is the only partition
            if (
                partition_df.osm_coastline_is_closed.iloc[0]
                and len(df["partition"].unique()) == 1
            ):
                coords.append(
                    coords[0]
                )  # Add the first point at the end to close the loop

            lines.append(LineString(coords))
        # Else case can be added if needed to handle single-point partitions

    return pd.Series(lines)


def retrieve_transects_by_roi(roi, storage_options=None):
    coclico_catalog = pystac.Catalog.from_file(
        "https://coclico.blob.core.windows.net/stac/v1/catalog.json"
    )
    gcts_collection = coclico_catalog.get_child("gctr")
    gcts_extents = read_items_extent(
        gcts_collection, columns=["geometry", "assets"], storage_options=storage_options
    )
    hrefs = gpd.sjoin(gcts_extents, roi).drop(columns=["index_right"]).href.to_list()

    TRANSECT_COLUMNS = [
        "transect_id",
        "lon",
        "lat",
        "bearing",
        "geometry",
        "osm_coastline_is_closed",
        "osm_coastline_length",
        "utm_epsg",
        "bbox",
        "quadkey",
        "country",
        "common_country_name",
        "dist_b0",
        "dist_b30",
        "dist_b330",
    ]
    transects = dask_geopandas.read_parquet(
        hrefs, storage_options=storage_options, columns=TRANSECT_COLUMNS
    )
    transects_roi = (
        transects.sjoin(roi.to_crs(transects.crs))
        .drop(columns=["index_right"])
        .compute()
    )

    unique_coastline_names = list(
        map(str, transects_roi.transect_id.str.extract(r"(cl\d+s\d+)")[0].unique())
    )

    def add_coastline_name(df):
        df["coastline_name"] = df.transect_id.str.extract(r"(cl\d+s\d+)")
        return df

    meta = make_meta(transects)
    new_col_meta = pd.DataFrame({"coastline_name": pd.Series([], dtype=str)})
    meta = pd.concat([meta, new_col_meta])

    transects = transects.map_partitions(add_coastline_name, meta=meta)
    transects_roi = transects.loc[
        transects["coastline_name"].isin(unique_coastline_names)
    ].compute()

    transects_roi = transects_roi.sort_values("transect_id")
    transects_roi[["coastline_id", "segment_id", "transect_dist"]] = (
        transects_roi.transect_id.str.extract(r"cl(\d+)s(\d+)tr(\d+)")
    )
    transects_roi = transects_roi.astype(
        {"coastline_id": int, "segment_id": int, "transect_dist": int}
    )
    return transects_roi

import dask
import geopandas as gpd
import pandas as pd

dask.config.set({"dataframe.query-planning": False})

import dask_geopandas
import pystac
from coastmonitor.io.utils import read_items_extent
from dask.dataframe.utils import make_meta
from shapely.geometry import LineString


def transect_origins_to_coastline(df):
    # Ensure df is sorted if not already
    df = df.sort_values(by=["tr_name", "segment_id", "transect_id"])

    # Identify partitions by checking where the difference in transect_id is not 100
    # diff() is NaN for the first row, so we use fillna() to set it to a value that does not equal 100 (e.g., 0)
    df["partition"] = (df["transect_id"].diff().fillna(0) != 100).cumsum()

    lines = []
    for _, partition_df in df.groupby("partition"):
        if len(partition_df) > 1:
            coords = gpd.GeoSeries.from_xy(
                partition_df["lon"], partition_df["lat"]
            ).to_list()

            # Check if the coastline is closed and this is the only partition
            if (
                partition_df.coastline_is_closed.iloc[0]
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
    gcts_collection = coclico_catalog.get_child("gcts")
    gcts_extents = read_items_extent(
        gcts_collection, columns=["geometry", "assets"], storage_options=storage_options
    )
    hrefs = gpd.sjoin(gcts_extents, roi).drop(columns=["index_right"]).href.to_list()

    transects = dask_geopandas.read_parquet(hrefs, storage_options=storage_options)
    transects_roi = (
        transects.sjoin(roi.to_crs(transects.crs))
        .drop(columns=["index_right"])
        .compute()
    )

    unique_coastline_names = list(
        map(str, transects_roi.tr_name.str.extract(r"(cl\d+s\d+)")[0].unique())
    )

    def add_coastline_name(df):
        df["coastline_name"] = df.tr_name.str.extract(r"(cl\d+s\d+)")
        return df

    meta = make_meta(transects)
    new_col_meta = pd.DataFrame({"coastline_name": pd.Series([], dtype=str)})
    meta = pd.concat([meta, new_col_meta])

    transects = transects.map_partitions(add_coastline_name, meta=meta)
    transects_roi = transects.loc[
        transects["coastline_name"].isin(unique_coastline_names)
    ].compute()

    transects_roi = transects_roi.sort_values("tr_name")
    transects_roi[["coastline_id", "segment_id", "transect_id"]] = (
        transects_roi.tr_name.str.extract(r"cl(\d+)s(\d+)tr(\d+)")
    )
    transects_roi = transects_roi.astype(
        {"coastline_id": int, "segment_id": int, "transect_id": int}
    )
    return transects_roi


#  [markdown]
# # Examples how to work with Global Coastal Transect System 
# 
# Run the first few cells to load required functions and jump to the section you're interested in afterwards. 

# 
import sys

sys.path.insert(0, "../src")

from coastmonitor.io.drive_config import configure_instance

configure_instance(branch="dev")

import logging
import os
import pathlib

import dask

dask.config.set({"dataframe.query-planning": False})

import dask_geopandas
import duckdb
import geopandas as gpd
import hvplot.pandas
import pandas as pd
import pystac
import shapely
from dotenv import load_dotenv
from ipyleaflet import Map, basemaps

from coastmonitor.geo.geometries import geo_bbox

load_dotenv(override=True)

sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN")
account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
storage_options = {"account_name": account_name, "credential": sas_token}

logging.getLogger("azure").setLevel(logging.WARNING)

# 
storage_options

#  [markdown]
# ## Load from STAC catalog
# 
# Load the transects from our CoCliCo STAC catalog. 

# 
coclico_catalog = pystac.Catalog.from_file(
    "https://coclico.blob.core.windows.net/stac/v1/catalog.json"
)

# 
coclico_catalog

# 
list(coclico_catalog.get_all_collections())

# 
gcts = coclico_catalog.get_child("gcts")
gcts

#  [markdown]
# ## The dataset is partitioned into geospatial chunks
# 
# The dataset is divided into different chunks, that each span a different region of the world. In the next cell
# we read the spatial extends of each chunk and compose that into a GeoDataFrame

# 
# makes a list of all items (data partitions) in the GCTS STAC catalog
items = list(gcts.get_all_items())
bboxes = pd.concat([geo_bbox(*i.to_dict()["bbox"]) for i in items])
bboxes = bboxes.reset_index(drop=True)
bboxes.explore()

# 
# ## Compose the transect origins into coastlines

# 
import geopandas as gpd
import pandas as pd
import shapely
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


# ]
# ## Plus we can handle region of interests that do not span all coastlines

# 
import fiona
fiona.drvsupport.supported_drivers["KML"] = "rw"
import os   
from pathlib import Path                                                                
import glob  
maindir = r"d:\FHICS\ShorelineS\ROI"  
outdir = r"d:\FHICS\ShorelineS\shapefiles"                                                             
os.chdir(maindir)       
for file in glob.iglob('*.kml'):                           
    print(file)
    kml_fp = file
    
    root = Path(kml_fp).stem
    
    
    roi = gpd.read_file(kml_fp, driver="KML")

    bboxes_roi = gpd.sjoin(bboxes, roi)[bboxes.columns]
    items_roi = [items[i] for i in bboxes_roi.index]
    hrefs = [i.assets["data"].href for i in items_roi]

 
    transects = dask_geopandas.read_parquet(hrefs, storage_options=storage_options)
    transects
    # 
    transects = dask_geopandas.read_parquet(hrefs, storage_options=storage_options)
    transects_roi = (
        transects.sjoin(roi.to_crs(transects.crs))
        .drop(columns=["index_right", "Name"])
        .compute()
    )
    # to ensure that all transects are sorted along the coastline
    transects_roi = transects_roi.sort_values("tr_name")
    # add the id's
    transects_roi[
        ["coastline_id", "segment_id", "transect_id"]
    ] = transects_roi.tr_name.str.extract(r"cl(\d+)s(\d+)tr(\d+)")
    transects_roi = transects_roi.astype(
        {"coastline_id": int, "segment_id": int, "transect_id": int}
    )

    # 
    # Apply function and explode to get one LineString per row
    coastline = (
        transects_roi.groupby("coastline_id")
        .apply(transect_origins_to_coastline)
        .dropna()
        .reset_index()
        .rename(columns={0: "geometry"})
)
    coastline = gpd.GeoDataFrame(coastline, crs=4326)
    coastline = gpd.overlay(coastline, roi[["geometry"]]).explode(index_parts=False)

    # 
    m = roi.explore()
    #gpd.GeoDataFrame(coastline, crs=4326).explore(color="red", m=m)
    gpd.GeoDataFrame(coastline).explore().save(os.path.join(maindir,root+".html"))
    # 
    coastline_UTM=coastline.to_crs(26918)
    coastline_UTM.head
    # 
    coastline_UTM.to_file(os.path.join(outdir,root+".shp"))

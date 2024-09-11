#!/usr/bin/env python
# coding: utf-8

# # How to extract the Primary Shoreline-Change Signal per Transect?
# 
# Many satellite-derived shoreline raw series include multiple shoreline-intersections due to transects spanning narrow coastal features, such as spits or lagoons. This results in noisy time series at trensect level because the series often include observations from multiple shorelines. Additionally, challenging conditions like cloud coverage can lead to incomplete detection of shorelines, causing significant step changes and further noise in the data. To address these issues, a filtering process is applied to extract the primary shoreline-change series from the raw data. 
# 
# ## Processing Steps
# The current workflow extracts the primary shoreline-change signal for a transect by following these steps:
# 
# 1. Determine Maximum Valid Distance:
#     - Set a default transect length of 2000 meters.
#     - Reduce the distance in areas where the transect intersects a second time with the OSM coastline (e.g., bays) to exclude observations from secondary shorelines.
# 2. Select Offshore Observations:
#     - Select one observation per year, that the furthest offshore within the determined maximum distance, as the primary observation.
# 3. Iteratively update the Primary Observations by filtering out observations with large step changes AND mad outlier detection:
#     - Group observations at the transect level within a maximum step change range of 150 meters.
#     - Remove observations with large step changes and replace with alternative observations if those exist for the same year.
#     - Recursively detect and remove outliers using the MAD method while replacing with alternative observations if those exist for the same year. 
# 4. Compute Statistics:
#     - Calculate statistical measures for the primary observations.
# 
# By following these steps, the primary shoreline-change signal is accurately identified for each transect, ensuring the use of the most relevant and representative data for analysis. This process also provides detailed statistics at the transect level for the primary observations.

# ## Instructions
# 
# 1. Use the DynamicMap: By default, the map is set to Namibia. Select your area of interest on the DynamicMap. After selecting the area, proceed to the next step to store the spatial extent in variables and retrieve the data from cloud storage.
# 
# 2. Create Visualization Panels: Run the subsequent cells to create dashboard app that shows the clean series with respect to the raw data. By default, these panels open in a new tab in the browser for stability. If the plot does not display correctly, refresh the tab several times (up to 10 times may be needed).
# 
# In practice, once you have become familiar with the different visualiation tools, you probably want to explore several areas, so you go back to the DynamicMap and zoom to another region, extract the bounds, retrieve, process and show the data (step 1-2). 

# In[1]:


import sys

import dask

dask.config.set({"dataframe.query-planning": False})

import logging
import os
import pathlib

import coastpy
import colorcet as cc
import dask_geopandas
import duckdb
import geopandas as gpd
import geoviews as gv
import holoviews as hv
import hvplot.pandas
import numpy as np
import pandas as pd
import panel as pn
import pystac
import shapely
from dotenv import load_dotenv

from coastmonitor.shorelines.intersection import (
    add_transect_statistics,
    find_primary_signal_per_transect_group,
)
sys.path.insert(0, "../src")
from coastlines4shorelines.utils import retrieve_transects_by_roi

load_dotenv(override=True)

# NOTE: access tokens to the data are available upon request.
sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN")
account_name = "coclico"
storage_options = {"account_name": account_name, "credential": sas_token}

# These are the URL's to the STAC catalog that we can use to efficiently index the data
COCLICO_STAC_URL = "https://coclico.blob.core.windows.net/stac/v1/catalog.json"

# Global Coastal Transect System (publicly available and in review)
GCTS_COLLECTION_NAME = "gcts"

# Global Coastal Transect Repository (unreleased; access keys provided upon request). This dataset consists
# of GCTS + several other characteristics, such as intersection distance to nearest coastline.
GCTR_COLLECTION_NAME = "gctr"

# ShorelineMonitor Raw Series (unreleased; access keys provided upon request). This dataset consists
# ShorelineMonitor Shorlines that are mapped onto the Global Coastal Transect System (Raw Series) that
# have a wide range of additional statistics used to filter out the primary, high-quality observations.
SM_COLLECTION_NAME = "shorelinemonitor-raw-series"

# These are the transect columns required for the analysis
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

hv.extension("bokeh")
pn.extension()


# ## Read the STAC collections

# In[2]:


coclico_catalog = pystac.Catalog.from_file(COCLICO_STAC_URL)
sm_collection = coclico_catalog.get_child(SM_COLLECTION_NAME)
gcts_collection = coclico_catalog.get_child(GCTR_COLLECTION_NAME)


# ## Define a region of interest (ROI) based on a kml file

# In[3]:


import pathlib
import fiona
 
fiona.drvsupport.supported_drivers["KML"] = "rw"
roi_dir = pathlib.Path.cwd().parent / "data" / "ROIs"
roi = gpd.read_file(roi_dir / "SanibelCaptiva.kml", driver="KML")


# In[4]:


# Get the total bounds
bounds = roi.total_bounds

minx, miny, maxx, maxy = bounds
print(minx,miny,maxx, maxy)


# ## Create a DuckDB query engine to retrieve data from cloud storage

# In[5]:


from coastmonitor.shorelines.intersection import (
    clean_raw_series,
    compute_diffs,
    compute_ols_trend,
)

sds_ts_engine = coastpy.io.STACQueryEngine(
    stac_collection=sm_collection,
    storage_backend="azure",
)
sds_ts = sds_ts_engine.get_data_within_bbox(minx, miny, maxx, maxy)
transects_engine = coastpy.io.STACQueryEngine(
    stac_collection=gcts_collection, storage_backend="azure", columns=TRANSECT_COLUMNS
)


# In[ ]:


transects = transects_engine.get_data_within_bbox(minx, miny, maxx, maxy)

sds_ts_clean = clean_raw_series(
    sds_ts,
    transects,
    method="offshore",
    multi_obs_threshold=17.5,
    max_step_change=150,
    relative_importance_threshold=0.6,
)
# filter out the primary observations to get cleaner data


# In[ ]:


sds_ts_clean_prim = sds_ts_clean[sds_ts_clean["obs_is_primary"]].copy()


# In[ ]:


transects_roi = retrieve_transects_by_roi(roi, storage_options=storage_options)


# In[ ]:


transects_roi[["geometry"]].explore()


# In[ ]:


merge=pd.merge(transects_roi[["transect_id","lon","lat"]],sds_ts_clean_prim,how="left",on="transect_id")


# In[ ]:


merge.groupby("transect_id").time.value_counts().unique()


# In[ ]:


merge.head()


# In[ ]:


import netCDF4
import xarray as xr

gdf = gpd.GeoDataFrame(merge)
gdf['time'] = pd.to_datetime(gdf['time'])

# Pivot the data
lon_pivot = gdf.pivot(index='time', columns='transect_id', values='lon_y')
lat_pivot = gdf.pivot(index='time', columns='transect_id', values='lat_y')

# Convert the pivot tables to xarray DataArray
lon_xr = xr.DataArray(lon_pivot)
lat_xr = xr.DataArray(lat_pivot)

# Create a Dataset from the DataArrays
ds = xr.Dataset({'lon': lon_xr, 'lat': lat_xr})
ds.to_netcdf('test.nc')
print(ds)


# In[ ]:


print(sds_ts_clean.time.unique())


# In[ ]:


gdf


# In[ ]:


print(ds["time"])


# ### Visualize in a small app

# In[ ]:


from coastmonitor.visualization.apptools import SpatialDataFrameApp

sds_ts_clean_ac = compute_ols_trend(
    sds_ts_clean[sds_ts_clean["obs_is_primary"]],
    transects,
    x="time",
    y="shoreline_position",
)

app = SpatialDataFrameApp(sds_ts_clean_ac, transects, sds_ts_clean)
app.create_view()
app.view.show()


# In[ ]:





# In[ ]:





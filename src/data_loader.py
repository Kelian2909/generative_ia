import xarray as xr
import numpy as np


def load_haines(path: str):
    ds = xr.open_dataset(path)

    t850 = ds.t.sel(pressure_level=850) - 273.15
    t700 = ds.t.sel(pressure_level=700) - 273.15
    rh850 = ds.r.sel(pressure_level=850)

    dewpoint_depression_850 = (100 - rh850) / 5
    haines_index = (t850 - t700) + dewpoint_depression_850

    print("Dimensions de l'indice :", haines_index.dims)
    print("Aperçu des valeurs :", haines_index.values.flatten()[:5])

    return ds, haines_index

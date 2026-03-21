"""Ionospheric profile utilities based on IRI2020."""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import logging

# Optional import for iri2020
try:
    import iri2020
    IRI2020_AVAILABLE = True
except ImportError:
    IRI2020_AVAILABLE = False

_log = logging.getLogger(__name__)


class Ionosphere1D:
    """One-dimensional ionosphere profile computed from IRI2020.

    Parameters
    ----------
    lat : float
        Latitude in degrees.
    lon : float
        Longitude in degrees.
    alt_km : array-like
        Altitude grid in km.
    time : datetime
        UTC time used for model evaluation.
    model : str, optional
        Ionospheric model name. Only ``"iri2020"`` is supported.

    Attributes
    ----------
    ionosphere : xr.Dataset
        Dataset with variable ``electron_density`` in ``m^-3`` and coordinate
        ``altitude`` in km.
    """
    
    def __init__(self, lat, lon, alt_km, time, model="iri2020"):
        """Initialize and compute an IRI2020 profile."""
        if model != "iri2020":
            raise ValueError("Invalid model. Only 'iri2020' is supported.")
            
        if model == "iri2020" and not IRI2020_AVAILABLE:
            raise ImportError("IRI2020 model requested but iri2020 package is not installed. "
                            "Please install it with: pip install iri2020")
            
        self.model = model
        self.lat = float(lat)
        self.lon = float(lon)
        self.alt_km = np.array(alt_km)
        self.time = time
        self.ionosphere = None
        
        if model == "iri2020":
            self.compute_iri2020_model()
        
    def __str__(self):
        """Return a concise description of this ionosphere profile."""
        return (f"Ionosphere1D Model: {self.model}\n"
                f"Latitude (deg): {self.lat}\n"
                f"Longitude (deg): {self.lon}\n"
                f"Altitude range (km): {self.alt_km.min():.1f} - {self.alt_km.max():.1f}\n"
                f"Time: {self.time}\n"
                f"Number of altitude points: {len(self.alt_km)}")

    def print(self):
        """Log the object representation at INFO level."""
        _log.info("%s", self.__str__())
        
    def compute_iri2020_model(self):
        """Compute and store an IRI2020 vertical profile.

        Returns
        -------
        xr.Dataset
            Ionosphere dataset with ``electron_density`` in ``m^-3``.
        """
        # Prepare altitude range for IRI2020: [min, max, step]
        alt_min = float(self.alt_km.min())
        alt_max = float(self.alt_km.max())
        
        # Determine step size based on the altitude grid
        if len(self.alt_km) > 1:
            alt_diffs = np.diff(self.alt_km)
            # Use the most common step size, or minimum if all different
            unique_diffs = np.unique(alt_diffs)
            if len(unique_diffs) == 1:
                alt_step = float(unique_diffs[0])
            else:
                # Use minimum step to ensure all requested altitudes are covered
                alt_step = float(alt_diffs.min())
        else:
            alt_step = 1.0  # Default 1 km step for single altitude
            
        # Create altitude range for IRI2020: [min, max, step]
        altkmrange = [alt_min, alt_max, alt_step]
        
        # Run IRI2020 model
        iri_data = iri2020.IRI(self.time, altkmrange, self.lat, self.lon)
        
        # Extract IRI2020 altitudes and electron density
        iri_altitudes = iri_data.alt_km.values
        iri_electron_density = iri_data.ne.values  # m⁻³
        
        # Interpolate to requested altitude grid if different
        if not np.array_equal(iri_altitudes, self.alt_km):
            # Interpolate electron density to requested altitudes
            electron_density = np.interp(self.alt_km, iri_altitudes, iri_electron_density)
        else:
            electron_density = iri_electron_density
            
        # Create xarray Dataset for the ionospheric model
        self.ionosphere = xr.Dataset(
            {
                "electron_density": (["altitude"], electron_density),
            },
            coords={
                "altitude": self.alt_km,
                "latitude": self.lat,
                "longitude": self.lon,
                "time": self.time
            },
            attrs={
                "model": self.model,
                "description": "Ionospheric parameters from IRI2020 model",
                "electron_density_units": "m^-3",
                "altitude_units": "km",
            }
        )
        self.ionosphere["electron_density"].attrs["units"] = "m^-3"
        
        # Store additional IRI2020 parameters if available
        try:
            self.ionosphere.attrs['TEC'] = float(iri_data.TEC.values)
        except (AttributeError, TypeError, ValueError) as exc:
            _log.debug("IRI2020 TEC attribute unavailable: %s", exc)
            
        try:
            self.ionosphere.attrs['hmF2'] = float(iri_data.hmF2.values)
        except (AttributeError, TypeError, ValueError) as exc:
            _log.debug("IRI2020 hmF2 attribute unavailable: %s", exc)
        return self.ionosphere
            
    def plot(self):
        """
        Plot the electron density profile vs altitude.
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 10))
        
        # Plot electron density
        ax.semilogx(self.ionosphere["electron_density"], self.alt_km)
        ax.set_xlabel('Electron Density (m⁻³)')
        ax.set_ylabel('Altitude (km)')
        ax.set_title(f'Ionospheric Electron Density Profile\n'
                    f'Lat: {self.lat:.2f}°, Lon: {self.lon:.2f}°, Time: {self.time}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class IonosphereModel:
    """Factory for ionosphere profile classes."""
    
    def __init__(self):
        self.supported_models = ["iri2020"]
        
    def create_ionosphere_1d(self, lat, lon, alt_km, time, model="iri2020"):
        """Create a configured ``Ionosphere1D`` instance."""
        if model not in self.supported_models:
            raise ValueError(f"Model '{model}' not supported. Available models: {self.supported_models}")
            
        if model == "iri2020" and not IRI2020_AVAILABLE:
            raise ImportError("IRI2020 model requested but iri2020 package is not installed. "
                            "Please install it with: pip install iri2020")
            
        return Ionosphere1D(lat, lon, alt_km, time, model)

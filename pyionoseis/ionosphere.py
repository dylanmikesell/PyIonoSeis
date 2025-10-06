'''
A module for ionospheric modeling classes and functions.
This module provides classes for computing ionospheric parameters, particularly
electron density, using various ionospheric models.
'''

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Optional import for iri2020
try:
    import iri2020
    IRI2020_AVAILABLE = True
except ImportError:
    IRI2020_AVAILABLE = False


class Ionosphere1D:
    '''
    A class to represent a one-dimensional ionospheric model using the IRI2020 model.
    
    Parameters:
    -----------
    lat : float
        Latitude in degrees.
    lon : float
        Longitude in degrees.
    alt_km : array-like
        Altitude array in kilometers.
    time : datetime
        The time for which the ionospheric parameters are calculated.
    model : str
        The ionospheric model used (default is "iri2020").
        
    Attributes:
    -----------
    lat : float
        Latitude in degrees.
    lon : float
        Longitude in degrees.
    alt_km : array-like
        Altitude array in kilometers.
    time : datetime
        The time for calculation.
    model : str
        The ionospheric model used.
    ionosphere : xarray.Dataset
        The ionospheric parameters including electron density.
        
    Methods:
    --------
    __init__(self, lat, lon, alt_km, time, model="iri2020"):
        Initializes the Ionosphere1D object with the given parameters.
    __str__(self):
        Returns a string representation of the Ionosphere1D object.
    print(self):
        Prints the string representation of the Ionosphere1D object.
    compute_iri2020_model(self):
        Computes the ionospheric model using the IRI2020 model and calculates
        electron density and other ionospheric parameters.
    plot(self):
        Plot the electron density profile vs altitude.
    '''
    
    def __init__(self, lat, lon, alt_km, time, model="iri2020"):
        """
        Initialize the Ionosphere1D object.
        
        Parameters:
        -----------
        lat : float
            Latitude in degrees.
        lon : float
            Longitude in degrees.
        alt_km : array-like
            Altitude array in kilometers.
        time : datetime
            The time for which the ionospheric parameters are calculated.
        model : str
            The ionospheric model used (default is "iri2020").
        """
        if model != "iri2020":
            raise ValueError("Invalid model. Only 'iri2020' is supported.")
            
        if model == "iri2020" and not IRI2020_AVAILABLE:
            raise ImportError("IRI2020 model requested but iri2020 package is not installed. "
                            "Please install it with: pip install iri2020")
            
        self.model = model
        self.lat = lat
        self.lon = lon
        self.alt_km = np.array(alt_km)
        self.time = time
        
        if model == "iri2020":
            self.compute_iri2020_model()
        
    def __str__(self):
        """
        Returns a string representation of the Ionosphere1D object.
        """
        return (f"Ionosphere1D Model: {self.model}\n"
                f"Latitude (deg): {self.lat}\n"
                f"Longitude (deg): {self.lon}\n"
                f"Altitude range (km): {self.alt_km.min():.1f} - {self.alt_km.max():.1f}\n"
                f"Time: {self.time}\n"
                f"Number of altitude points: {len(self.alt_km)}")

    def print(self):
        """
        Prints the string representation of the Ionosphere1D object.
        """
        print(self.__str__())
        
    def compute_iri2020_model(self):
        """
        Compute the ionospheric model using the IRI2020 model.
        
        This function calculates ionospheric parameters, particularly electron density,
        using the IRI2020 model based on the provided time, altitude range, latitude,
        and longitude.
        
        The IRI2020 model provides electron density in m⁻³ and other ionospheric parameters.
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
                "electron_density": (["altitude"], electron_density),  # m⁻³
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
                "electron_density_units": "m⁻³",
                "altitude_units": "km"
            }
        )
        
        # Store additional IRI2020 parameters if available
        try:
            self.ionosphere.attrs['TEC'] = float(iri_data.TEC.values)
        except Exception:
            pass
            
        try:
            self.ionosphere.attrs['hmF2'] = float(iri_data.hmF2.values)
        except Exception:
            pass
            
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
    """
    Base class for ionospheric models. This can be extended to support multiple models.
    
    Currently supports:
    - IRI2020: International Reference Ionosphere 2020 model
    """
    
    def __init__(self):
        self.supported_models = ["iri2020"]
        
    def create_ionosphere_1d(self, lat, lon, alt_km, time, model="iri2020"):
        """
        Factory method to create a 1D ionospheric model.
        
        Parameters:
        -----------
        lat : float
            Latitude in degrees.
        lon : float
            Longitude in degrees.
        alt_km : array-like
            Altitude array in kilometers.
        time : datetime
            The time for calculation.
        model : str
            The ionospheric model to use.
            
        Returns:
        --------
        Ionosphere1D
            A 1D ionospheric model object.
        """
        if model not in self.supported_models:
            raise ValueError(f"Model '{model}' not supported. Available models: {self.supported_models}")
            
        if model == "iri2020" and not IRI2020_AVAILABLE:
            raise ImportError("IRI2020 model requested but iri2020 package is not installed. "
                            "Please install it with: pip install iri2020")
            
        return Ionosphere1D(lat, lon, alt_km, time, model)

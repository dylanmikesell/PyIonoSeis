"""
IGRF (International Geomagnetic Reference Field) module for PyIonoSeis.

This module provides classes and functions for computing magnetic field parameters
using the IGRF model via the ppigrf package.

Classes:
--------
MagneticField1D : Class for computing 1D magnetic field profiles
MagneticFieldModel : Factory class for magnetic field model selection

Functions:
----------
Various utility functions for magnetic field calculations

Author: PyIonoSeis Development Team
License: MIT
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Try to import ppigrf with graceful fallback
try:
    import ppigrf
    PPIGRF_AVAILABLE = True
except ImportError:
    ppigrf = None
    PPIGRF_AVAILABLE = False


class MagneticField1D:
    """
    Class for computing 1D magnetic field profiles using IGRF models.
    
    This class calculates magnetic field components and derived parameters
    (inclination, declination) for a specific location and time using the
    IGRF model via the ppigrf package.
    
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
        The magnetic field model used.
    magnetic_field : xarray.Dataset
        The magnetic field parameters including field components and derived quantities.
        
    Methods:
    --------
    __init__(self, lat, lon, alt_km, time, model="igrf"):
        Initializes the MagneticField1D object with the given parameters.
    __str__(self):
        Returns a string representation of the MagneticField1D object.
    print(self):
        Prints the string representation of the MagneticField1D object.
    compute_igrf_model(self):
        Computes the magnetic field model using the IGRF model and calculates
        field components, inclination, and declination.
    plot(self, variable='total_field'):
        Plot magnetic field parameters vs altitude.
    """

    def __init__(self, lat, lon, alt_km, time, model="igrf"):
        """
        Initialize MagneticField1D object.
        
        Parameters:
        -----------
        lat : float
            Latitude in degrees.
        lon : float
            Longitude in degrees.
        alt_km : array-like
            Altitude array in kilometers.
        time : datetime
            Time for the calculation.
        model : str, optional
            Magnetic field model to use (default: "igrf").
        """
        self.lat = lat
        self.lon = lon
        self.alt_km = np.array(alt_km)
        self.time = time
        self.model = model
        self.magnetic_field = None
        
        # Compute the magnetic field model
        if model.lower() == "igrf":
            self.compute_igrf_model()
        else:
            raise ValueError(f"Unsupported magnetic field model: {model}")

    def __str__(self):
        """Return string representation of MagneticField1D object."""
        if self.magnetic_field is not None:
            field_info = f"\n Magnetic field computed: {list(self.magnetic_field.data_vars)}"
        else:
            field_info = "\n No magnetic field computed"
            
        return (f"MagneticField1D:\n"
                f" Location: {self.lat:.2f}°N, {self.lon:.2f}°E\n"
                f" Time: {self.time}\n"
                f" Altitude range: {self.alt_km.min():.0f} - {self.alt_km.max():.0f} km\n"
                f" Model: {self.model}{field_info}")

    def print(self):
        """Print the string representation of the object."""
        print(self.__str__())

    def compute_igrf_model(self):
        """
        Compute the magnetic field model using the IGRF model.
        
        This function calculates magnetic field components (Be, Bn, Bu),
        derived parameters (inclination, declination, total field intensity),
        using the IGRF model via ppigrf.
        
        The IGRF model provides magnetic field components in nT.
        """
        if not PPIGRF_AVAILABLE:
            raise ImportError("ppigrf package is not available. Please install it with: pip install ppigrf")
        
        # Calculate magnetic field components using ppigrf
        # ppigrf.igrf returns Be (east), Bn (north), Bu (up) in geodetic coordinates
        Be, Bn, Bu = ppigrf.igrf(self.lon, self.lat, self.alt_km, self.time)
        
        # Calculate derived parameters
        inclination, declination = ppigrf.get_inclination_declination(Be, Bn, Bu, degrees=True)
        
        # Calculate horizontal field intensity
        horizontal_intensity = np.sqrt(Be**2 + Bn**2)
        
        # Calculate total field intensity
        total_field = np.sqrt(Be**2 + Bn**2 + Bu**2)
        
        # Create xarray Dataset for the magnetic field model
        self.magnetic_field = xr.Dataset(
            {
                "Be": (["altitude"], Be.flatten()),  # East component (nT)
                "Bn": (["altitude"], Bn.flatten()),  # North component (nT)
                "Bu": (["altitude"], Bu.flatten()),  # Up component (nT)
                "inclination": (["altitude"], inclination.flatten()),  # Inclination (degrees)
                "declination": (["altitude"], declination.flatten()),  # Declination (degrees)
                "horizontal_intensity": (["altitude"], horizontal_intensity.flatten()),  # Horizontal intensity (nT)
                "total_field": (["altitude"], total_field.flatten()),  # Total field intensity (nT)
            },
            coords={
                "altitude": self.alt_km,
                "latitude": self.lat,
                "longitude": self.lon,
                "time": self.time
            },
            attrs={
                "model": self.model,
                "description": "Magnetic field parameters from IGRF model",
                "field_units": "nT",
                "angle_units": "degrees",
                "altitude_units": "km"
            }
        )
            
    def plot(self, variable='total_field'):
        """
        Plot magnetic field parameters vs altitude.
        
        Parameters:
        -----------
        variable : str, optional
            Variable to plot. Options:
            - 'total_field': Total field intensity (default)
            - 'components': All field components (Be, Bn, Bu)
            - 'inclination': Magnetic inclination
            - 'declination': Magnetic declination
            - 'horizontal_intensity': Horizontal field intensity
        """
        if self.magnetic_field is None:
            raise ValueError("Magnetic field not computed. Call compute_igrf_model() first.")
        
        if variable == 'total_field':
            fig, ax = plt.subplots(1, 1, figsize=(8, 10))
            ax.plot(self.magnetic_field["total_field"], self.alt_km)
            ax.set_xlabel('Total Field Intensity (nT)')
            ax.set_ylabel('Altitude (km)')
            ax.set_title(f'IGRF Total Magnetic Field Intensity\n'
                        f'Lat: {self.lat:.2f}°, Lon: {self.lon:.2f}°, Time: {self.time}')
            ax.grid(True, alpha=0.3)
            
        elif variable == 'components':
            fig, axes = plt.subplots(1, 3, figsize=(15, 10))
            
            # East component
            axes[0].plot(self.magnetic_field["Be"], self.alt_km, 'r-', label='Be (East)')
            axes[0].set_xlabel('Be - East Component (nT)')
            axes[0].set_ylabel('Altitude (km)')
            axes[0].grid(True, alpha=0.3)
            
            # North component
            axes[1].plot(self.magnetic_field["Bn"], self.alt_km, 'g-', label='Bn (North)')
            axes[1].set_xlabel('Bn - North Component (nT)')
            axes[1].set_ylabel('Altitude (km)')
            axes[1].grid(True, alpha=0.3)
            
            # Up component
            axes[2].plot(self.magnetic_field["Bu"], self.alt_km, 'b-', label='Bu (Up)')
            axes[2].set_xlabel('Bu - Up Component (nT)')
            axes[2].set_ylabel('Altitude (km)')
            axes[2].grid(True, alpha=0.3)
            
            fig.suptitle(f'IGRF Magnetic Field Components\n'
                        f'Lat: {self.lat:.2f}°, Lon: {self.lon:.2f}°, Time: {self.time}')
            
        elif variable == 'inclination':
            fig, ax = plt.subplots(1, 1, figsize=(8, 10))
            ax.plot(self.magnetic_field["inclination"], self.alt_km)
            ax.set_xlabel('Magnetic Inclination (degrees)')
            ax.set_ylabel('Altitude (km)')
            ax.set_title(f'IGRF Magnetic Inclination\n'
                        f'Lat: {self.lat:.2f}°, Lon: {self.lon:.2f}°, Time: {self.time}')
            ax.grid(True, alpha=0.3)
            
        elif variable == 'declination':
            fig, ax = plt.subplots(1, 1, figsize=(8, 10))
            ax.plot(self.magnetic_field["declination"], self.alt_km)
            ax.set_xlabel('Magnetic Declination (degrees)')
            ax.set_ylabel('Altitude (km)')
            ax.set_title(f'IGRF Magnetic Declination\n'
                        f'Lat: {self.lat:.2f}°, Lon: {self.lon:.2f}°, Time: {self.time}')
            ax.grid(True, alpha=0.3)
            
        elif variable == 'horizontal_intensity':
            fig, ax = plt.subplots(1, 1, figsize=(8, 10))
            ax.plot(self.magnetic_field["horizontal_intensity"], self.alt_km)
            ax.set_xlabel('Horizontal Field Intensity (nT)')
            ax.set_ylabel('Altitude (km)')
            ax.set_title(f'IGRF Horizontal Magnetic Field Intensity\n'
                        f'Lat: {self.lat:.2f}°, Lon: {self.lon:.2f}°, Time: {self.time}')
            ax.grid(True, alpha=0.3)
            
        else:
            available_vars = ['total_field', 'components', 'inclination', 'declination', 'horizontal_intensity']
            raise ValueError(f"Unknown variable '{variable}'. Available variables: {available_vars}")
        
        plt.tight_layout()
        plt.show()


class MagneticFieldModel:
    """
    Factory class for magnetic field model selection.
    
    This class provides a unified interface for creating magnetic field
    models with different implementations.
    """
    
    @staticmethod
    def create(lat, lon, alt_km, time, model="igrf"):
        """
        Create a magnetic field model instance.
        
        Parameters:
        -----------
        lat : float
            Latitude in degrees.
        lon : float
            Longitude in degrees.
        alt_km : array-like
            Altitude array in kilometers.
        time : datetime
            Time for calculation.
        model : str, optional
            Model type (default: "igrf").
            
        Returns:
        --------
        MagneticField1D
            Magnetic field model instance.
        """
        if model.lower() == "igrf":
            return MagneticField1D(lat, lon, alt_km, time, model)
        else:
            raise ValueError(f"Unsupported magnetic field model: {model}")


# Convenience function for quick calculations
def calculate_magnetic_field(lat, lon, alt_km, time, model="igrf"):
    """
    Convenience function to quickly calculate magnetic field parameters.
    
    Parameters:
    -----------
    lat : float
        Latitude in degrees.
    lon : float
        Longitude in degrees.
    alt_km : array-like
        Altitude array in kilometers.
    time : datetime
        Time for calculation.
    model : str, optional
        Model type (default: "igrf").
        
    Returns:
    --------
    MagneticField1D
        Computed magnetic field model.
    """
    return MagneticFieldModel.create(lat, lon, alt_km, time, model)
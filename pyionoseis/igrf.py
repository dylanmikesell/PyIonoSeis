"""IGRF magnetic-field profile utilities using ``ppigrf``."""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import logging

_log = logging.getLogger(__name__)

# Try to import ppigrf with graceful fallback
try:
    import ppigrf
    PPIGRF_AVAILABLE = True
except ImportError:
    ppigrf = None
    PPIGRF_AVAILABLE = False


class MagneticField1D:
    """One-dimensional magnetic field profile computed from IGRF.

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
        Magnetic field model name. Only ``"igrf"`` is supported.

    Attributes
    ----------
    magnetic_field : xr.Dataset
        Dataset with geodetic/geocentric magnetic components and derived
        inclination, declination, and field intensities.
    """

    def __init__(self, lat, lon, alt_km, time, model="igrf"):
        """Initialize and compute an IGRF magnetic profile."""
        self.lat = float(lat)
        self.lon = float(lon)
        self.alt_km = np.array(alt_km)
        self.time = time
        self.model = model
        self.magnetic_field = None
        
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
        """Log the object representation at INFO level."""
        _log.info("%s", self.__str__())

    def compute_igrf_model(self):
        """Compute and store an IGRF magnetic-field profile.

        Returns
        -------
        xr.Dataset
            Dataset containing magnetic components and derived quantities.

        Raises
        ------
        ImportError
            If ``ppigrf`` is unavailable.
        """
        if not PPIGRF_AVAILABLE:
            raise ImportError("ppigrf package is not available. Please install it with: pip install ppigrf")
        
        Be, Bn, Bu = ppigrf.igrf(self.lon, self.lat, self.alt_km, self.time)
        
        RE = 6371.2  # km, standard geophysical radius
        r = RE + self.alt_km  # radius from Earth center
        theta = 90.0 - self.lat  # colatitude (degrees)
        phi = self.lon  # longitude (same as geodetic)
        
        Br, Btheta, Bphi = ppigrf.igrf_gc(r, theta, phi, self.time)
        
        inclination, declination = ppigrf.get_inclination_declination(Be, Bn, Bu, degrees=True)
        
        horizontal_intensity = np.sqrt(Be**2 + Bn**2)
        total_field = np.sqrt(Be**2 + Bn**2 + Bu**2)
        
        self.magnetic_field = xr.Dataset(
            {
                "Be": (["altitude"], Be.flatten()),
                "Bn": (["altitude"], Bn.flatten()),
                "Bu": (["altitude"], Bu.flatten()),
                "Br": (["altitude"], Br.flatten()),
                "Btheta": (["altitude"], Btheta.flatten()),
                "Bphi": (["altitude"], Bphi.flatten()),
                "inclination": (["altitude"], inclination.flatten()),
                "declination": (["altitude"], declination.flatten()),
                "horizontal_intensity": (["altitude"], horizontal_intensity.flatten()),
                "total_field": (["altitude"], total_field.flatten()),
            },
            coords={
                "altitude": self.alt_km,
                "latitude": self.lat,
                "longitude": self.lon,
                "time": self.time
            },
            attrs={
                "model": self.model,
                "description": "Magnetic field parameters from IGRF model in geodetic and geocentric coordinates",
                "field_units": "nT",
                "angle_units": "degrees",
                "altitude_units": "km",
                "geodetic_components": "Be (east), Bn (north), Bu (up)",
                "geocentric_components": "Br (radial), Btheta (south), Bphi (east)",
                "coordinate_system_note": "Geodetic components are tangent to Earth's ellipsoid, geocentric are spherical"
            }
        )
        for var in [
            "Be", "Bn", "Bu", "Br", "Btheta", "Bphi",
            "horizontal_intensity", "total_field",
        ]:
            self.magnetic_field[var].attrs["units"] = "nT"
        self.magnetic_field["inclination"].attrs["units"] = "degrees"
        self.magnetic_field["declination"].attrs["units"] = "degrees"
        return self.magnetic_field
            
    def plot(self, variable='total_field'):
        """
        Plot magnetic field parameters vs altitude.
        
        Parameters:
        -----------
        variable : str, optional
            Variable to plot. Options:
            - 'total_field': Total field intensity (default)
            - 'components': Geodetic field components (Be, Bn, Bu)
            - 'geocentric_components': Geocentric field components (Br, Btheta, Bphi)
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
            
            fig.suptitle(f'IGRF Magnetic Field Components (Geodetic)\n'
                        f'Lat: {self.lat:.2f}°, Lon: {self.lon:.2f}°, Time: {self.time}')
            
        elif variable == 'geocentric_components':
            fig, axes = plt.subplots(1, 3, figsize=(15, 10))
            
            # Radial component
            axes[0].plot(self.magnetic_field["Br"], self.alt_km, 'r-', label='Br (Radial)')
            axes[0].set_xlabel('Br - Radial Component (nT)')
            axes[0].set_ylabel('Altitude (km)')
            axes[0].grid(True, alpha=0.3)
            
            # Colatitude component
            axes[1].plot(self.magnetic_field["Btheta"], self.alt_km, 'g-', label='Btheta (Colatitude)')
            axes[1].set_xlabel('Btheta - Colatitude Component (nT)')
            axes[1].set_ylabel('Altitude (km)')
            axes[1].grid(True, alpha=0.3)
            
            # Azimuth component
            axes[2].plot(self.magnetic_field["Bphi"], self.alt_km, 'b-', label='Bphi (Azimuth)')
            axes[2].set_xlabel('Bphi - Azimuth Component (nT)')
            axes[2].set_ylabel('Altitude (km)')
            axes[2].grid(True, alpha=0.3)
            
            fig.suptitle(f'IGRF Magnetic Field Components (Geocentric)\n'
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
            available_vars = ['total_field', 'components', 'geocentric_components', 'inclination', 'declination', 'horizontal_intensity']
            raise ValueError(f"Unknown variable '{variable}'. Available variables: {available_vars}")
        
        plt.tight_layout()
        plt.show()


class MagneticFieldModel:
    """Factory for magnetic-field profile classes."""
    
    @staticmethod
    def create(lat, lon, alt_km, time, model="igrf"):
        """Create a configured ``MagneticField1D`` instance."""
        if model.lower() == "igrf":
            return MagneticField1D(lat, lon, alt_km, time, model)
        else:
            raise ValueError(f"Unsupported magnetic field model: {model}")


# Convenience function for quick calculations
def calculate_magnetic_field(lat, lon, alt_km, time, model="igrf"):
    """Compute and return a magnetic-field profile object."""
    return MagneticFieldModel.create(lat, lon, alt_km, time, model)
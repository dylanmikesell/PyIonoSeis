"""Atmospheric profile utilities based on the MSISE-00 model."""

import msise00
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import logging

_log = logging.getLogger(__name__)

# Constants
AVO = 6.022e23  # (molecules/kmole) Avogadro's number 
R = 8.314  # J/(mol K) - ideal gas constant

# MSISE-00 OUTPUT VARIABLES: from msis_gtd8d.F90
#       D(1)  He number density (cm-3)
#       D(2)  O number density (cm-3)
#       D(3)  N2 number density (cm-3)
#       D(4)  O2 number density (cm-3)
#       D(5)  Ar number density (cm-3)
#       D(6)  Total mass density (g/cm3)
#       D(7)  H number density (cm-3)
#       D(8)  N number density (cm-3)
#       D(9)  Anomalous oxygen number density (cm-3)
#       T(1)  Exospheric temperature (K)
#       T(2)  Temperature at altitude (K)


class Atmosphere1D:
    """One-dimensional neutral atmosphere profile using MSISE-00.

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
        Atmospheric model name. Only ``"msise00"`` is supported.

    Attributes
    ----------
    atmosphere : xr.Dataset
        Dataset with variables ``velocity`` (km/s), ``density`` (kg m^-3),
        ``pressure`` (Pa), and ``temperature`` (K) on coordinate ``altitude``.
    f107 : float
        Daily F10.7 index from MSISE output attributes.
    Ap : float
        Daily geomagnetic activity index from MSISE output attributes.
    """

    def __init__(self, lat, lon, alt_km, time, model="msise00"):
        if model != "msise00":
            raise ValueError("Invalid model. Only 'msise00' is supported.")
        self.model = model
        self.lat = float(lat)
        self.lon = float(lon)
        self.alt_km = np.asarray(alt_km, dtype=float)
        self.time = time
        self.atmosphere = None
        self.f107 = None
        self.Ap = None

        self.compute_mise00_model()
        
    def __str__(self):
        """Return a concise description of this atmosphere profile."""
        return (f"Atmosphere1D Model: {self.model}\n"
                f"Latitude (deg): {self.lat}\n"
                f"Longitude (deg): {self.lon}\n"
                f"Altitude range (km): {self.alt_km.min():.1f} - "
                f"{self.alt_km.max():.1f}\n"
                f"Time: {self.time}\n"
                f"F107: {self.f107}\n"
                f"Ap: {self.Ap}\n")

    def print(self):
        """Log the object representation at INFO level."""
        _log.info("%s", self.__str__())
        
    def compute_mise00_model(self):
        """Compute and store an MSISE-00 vertical profile.

        Returns
        -------
        xr.Dataset
            Atmosphere dataset with named coordinates and units.
        """
        atmos = msise00.run(self.time, self.alt_km, self.lat, self.lon)

        self.f107 = atmos.attrs.get("f107")
        self.Ap = atmos.attrs.get("Ap")

        density2 = (
            4.0 * atmos["He"].squeeze()
            + 16.0 * atmos["O"].squeeze()
            + 28.0 * atmos["N2"].squeeze()
            + 32.0 * atmos["O2"].squeeze()
            + 40.0 * atmos["Ar"].squeeze()
            + 1.0 * atmos["H"].squeeze()
            + 14.0 * atmos["N"].squeeze()
        ) / AVO
        
        pressure = (
            atmos["He"].squeeze()
            + atmos["O"].squeeze()
            + atmos["N2"].squeeze()
            + atmos["O2"].squeeze()
            + atmos["Ar"].squeeze()
            + atmos["H"].squeeze()
            + atmos["N"].squeeze()
        ) * R * atmos["Texo"].squeeze() * 10.0e5 / AVO

        anm = (
            atmos["He"].squeeze()
            + atmos["O"].squeeze()
            + atmos["N2"].squeeze()
            + atmos["O2"].squeeze()
            + atmos["Ar"].squeeze()
            + atmos["H"].squeeze()
            + atmos["N"].squeeze()
        )
        bi = (atmos["N2"].squeeze() + atmos["O2"].squeeze()) / anm
        mono = (
            atmos["He"].squeeze()
            + atmos["O"].squeeze()
            + atmos["Ar"].squeeze()
            + atmos["H"].squeeze()
            + atmos["N"].squeeze()
        ) / anm
        cv = 1.5 * R * mono + 2.5 * R * bi
        cp = cv + R
        gamma = cp / cv
        
        velocity = np.sqrt(gamma * pressure / density2 / 1000.0) / 1000.0
        density_kg_m3 = atmos["Total"].squeeze()
        temperature_k = atmos["Tn"].squeeze()

        self.atmosphere = xr.Dataset(
            {
                "velocity": (["altitude"], velocity.values),
                "density": (["altitude"], density_kg_m3.values),
                "pressure": (["altitude"], pressure.values),
                "temperature": (["altitude"], temperature_k.values),
            },
            coords={
                "altitude": self.alt_km,
                "latitude": self.lat,
                "longitude": self.lon,
                "time": self.time,
            },
            attrs={"model": self.model, "description": "MSISE-00 atmosphere profile"},
        )

        self.atmosphere["velocity"].attrs["units"] = "km/s"
        self.atmosphere["density"].attrs["units"] = "kg m^-3"
        self.atmosphere["pressure"].attrs["units"] = "Pa"
        self.atmosphere["temperature"].attrs["units"] = "K"
        return self.atmosphere

    def plot(self):
        """Plot density, pressure, velocity, and temperature vs altitude."""
        fig, axs = plt.subplots(1, 4, figsize=(24, 6))
        
        # Plot density
        axs[0].plot(self.atmosphere["density"], self.alt_km)
        axs[0].set_xscale('log')
        axs[0].set_xlabel('Density (kg/m^3)')
        axs[0].set_ylabel('Altitude (km)')
        axs[0].grid(True)
        
        # Plot pressure
        axs[1].plot(self.atmosphere["pressure"], self.alt_km)
        axs[1].set_xscale('log')
        axs[1].set_xlabel('Pressure (Pa)')
        axs[1].set_ylabel('Altitude (km)')
        axs[1].grid(True)
        
        # Plot velocity
        axs[2].plot(self.atmosphere["velocity"], self.alt_km)
        axs[2].set_xlabel('Velocity (km/s)')
        axs[2].set_ylabel('Altitude (km)')
        axs[2].grid(True)
        
        # Plot temperature
        axs[3].plot(self.atmosphere["temperature"], self.alt_km)
        axs[3].set_xlabel('Temperature (K)')
        axs[3].set_ylabel('Altitude (km)')
        axs[3].grid(True)
        
        plt.tight_layout()
        plt.show()
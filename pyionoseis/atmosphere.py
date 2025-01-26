'''
A module for the atmosphere class. This class is used to generate the atmospheric parameters for the model.
'''

import msise00
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

'''
For MSISE00, the units are as follows:
    Temperature: degrees Kelvin [K]
    Density: particles per cubic meter [m^-3]
    Mass density: kilograms per cubic meter [kg m^-3]
'''

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
    '''
    A class to represent a one-dimensional atmospheric model using the MSISE00 model.
    ----------
    model : str
        The atmospheric model used (default is "msise00").
    lat : float
        Latitude in degrees.
    lon : float
        Longitude in degrees.
    alt_km : float
        Altitude in kilometers.
    time : datetime
        The time for which the atmospheric parameters are calculated.
    density_kg_m3 : float
        The total mass density in kg/m^3.
    f107 : float
        The daily F10.7 cm radio flux for the previous day.
    Ap : float
        The geomagnetic activity index.
    pressure_pa : float
        The atmospheric pressure in Pascals.
    velocity : float
        The velocity in km/s.
    Methods:
    -------
    __init__(self, glat, glon, alt_km, time, model="msise00"):
        Initializes the Atmosphere1D object with the given parameters.
    __str__(self):
        Returns a string representation of the Atmosphere1D object.
    print(self):
        Prints the string representation of the Atmosphere1D object.
    compute_mise00_model(self):
        Computes the atmospheric model using the MSISE00 model and calculates 
        various atmospheric parameters such as density, pressure, and velocity.
    '''
    pass
    def __init__(self, lat, lon, alt_km, time, model="msise00"):
        if model != "msise00":
            raise ValueError("Invalid model. Only 'msise00' is supported.")
        self.model = model
        self.lat = lat
        self.lon = lon
        self.alt_km = alt_km
        self.time = time

        if model == "msise00":
            self.compute_mise00_model()  # get the atmospheric parameters
        
    def __str__(self):
        # ToDo: fix the string representation after merging all atmosphere parameters into one xarray Dataset
        return (f"Atmosphere1D Model: {self.model}\n"
                f"Latitude (deg): {self.lat}\n"
                f"Longitude (deg): {self.lon}\n"
                f"Altitude (km): {self.alt_km}\n"
                f"Time: {self.time}\n"
                f"F107: {self.f107}\n"
                f"Ap: {self.Ap}\n"
                # f"Density (kg/m^3): {self.atmosphere["density"]}\n"
                # f"Pressure (Pa): {self.atmosphere["pressure"]}\n"
                # f"Velocity (km/s): {self.atmosphere["velocity"]}"
                )

    def print(self):
        print(self.__str__())
        
    def compute_mise00_model(self):
        """
        Compute the atmospheric model using the MSISE00 model.
        This function calculates various atmospheric parameters such as density, 
        pressure, and velocity using the MSISE00 model based on the provided 
        time, altitude, latitude, and longitude.
        Attributes:
            density_kg_m3 (float): The total mass density in kg/m^3.
            f107 (float): The daily F10.7 cm radio flux for the previous day.
            Ap (float): The geomagnetic activity index.
            pressure_pa (float): The atmospheric pressure in Pascals.
            velocity (float): The velocity in km/s.
        Notes:
            - The MSISE00 model includes species such as He, O, N2, O2, Ar, H, and N.
            - Missing density values are returned as 9.999e-38.
            - The F107 and F107A values are the 10.7 cm radio flux at the Sun-Earth distance, 
              not the radio flux at 1 AU.
            - The pressure calculation uses the universal gas constant R and the Avogadro constant AVO.
            - The heat capacity ratio (gamma) is calculated using the specific heat capacities 
              at constant volume (cv) and constant pressure (cp).
        """
        
        # We are using the default spieces for the MSISE00 model
        # Species included in mass density calculation are set in MSISINIT
        # Missing density values are returned as 9.999e-38
        atmos = msise00.run(self.time, self.alt_km, self.lat, self.lon)

        # F107 and F107A values are the 10.7 cm radio flux at the Sun-Earth distance, not the radio flux at 1 AU.

        # Daily F10.7 for previous day
        self.f107 = atmos.attrs.get("f107")
        # AP: Geomagnetic activity index array:
        #       (1) Daily Ap
        self.Ap = atmos.attrs.get("Ap")

        density2 = (4.0*atmos["He"].squeeze() + 
                16.*atmos["O"].squeeze() + 
                28.*atmos["N2"].squeeze() + 
                32.*atmos["O2"].squeeze() + 
                40.*atmos["Ar"].squeeze() + 
                1.*atmos["H"].squeeze() + 
                14.*atmos["N"].squeeze()) / AVO  # g/cm^3
        
        pressure = (atmos["He"].squeeze() + 
            atmos["O"].squeeze() + 
            atmos["N2"].squeeze() + 
            atmos["O2"].squeeze() + 
            atmos["Ar"].squeeze() + 
            atmos["H"].squeeze() + 
            atmos["N"].squeeze())*R*atmos["Texo"].squeeze()*10.e5/AVO  # Pa  ??

        # capacite calorifique a volume constant
        anm = (atmos["He"].squeeze() + 
                atmos["O"].squeeze() + 
                atmos["N2"].squeeze() + 
                atmos["O2"].squeeze() + 
                atmos["Ar"].squeeze() + 
                atmos["H"].squeeze() + 
                atmos["N"].squeeze())
        bi = (atmos["N2"].squeeze() + 
            atmos["O2"].squeeze()) / anm
        mono = (atmos["He"].squeeze() + 
                atmos["O"].squeeze() + 
                atmos["Ar"].squeeze() + 
                atmos["H"].squeeze() + 
                atmos["N"].squeeze()) / anm
        cv = 1.5*R*mono + 2.5*R*bi
        cp = cv + R
        gamma = cp / cv  # heat capacity ratio
        
        velocity = np.sqrt( gamma*pressure/density2/1000.) / 1000.  # km/s
        density_kg_m3 = atmos["Total"].squeeze()  # kg/m^3
        temperature_k = atmos["Tn"].squeeze()  # K

        # Create a new xarray Dataset for the 1D atmosphere model
        self.atmosphere = xr.Dataset(
            {
            "velocity": (["altitude"], velocity.values),  # km/s
            "density": (["altitude"], density_kg_m3.values),  # kg/m^3,
            "pressure": (["altitude"], pressure.values),  # Pa
            "temperature": (["altitude"], temperature_k.values)  # K
            },
            coords={
            "altitude": self.alt_km,
            "latitude": self.lat,
            "longitude": self.lon,
            "time": self.time
            }
        )

    def plot(self):
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
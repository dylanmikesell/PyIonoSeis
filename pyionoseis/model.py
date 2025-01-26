"""The model module contains common functions and classes used to generate the spatial modes.
"""

import toml
import xarray as xr
import numpy as np
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyionoseis.atmosphere import Atmosphere1D

class Model3D:
    """
    A class to represent a 3D model for ionospheric studies.
    Attributes:
    -----------
    name : str
        The name of the model.
    radius : float
        The radius of the model in kilometers.
    height : float
        The height of the model in kilometers.
    winds : bool
        A flag indicating if winds are considered in the model.
    grid_spacing : float
        The spacing of the grid in degrees.
    height_spacing : float
        The spacing of the height levels in kilometers.
    source : object
        The source object associated with the model.
    grid : xarray.DataArray
        The 3D grid of the model.
    lat_extent : tuple
        The latitude extent of the grid.
    lon_extent : tuple
        The longitude extent of the grid.
    Methods:
    --------
    __init__(self, toml_file=None):
        Initialize the Model3D instance.
        Parameters:
        -----------
        toml_file : str, optional
            The path to a TOML file to load model parameters from.
    load_from_toml(self, toml_file):
        Load model parameters from a TOML file.
        Parameters:
        -----------
        toml_file : str
            The path to the TOML file.
    assign_source(self, source):
        Assign a source to the model.
        Parameters:
        -----------
        source : object
            The source object to be assigned to the model.
    __str__(self):
        Return a string representation of the model.
        Returns:
        --------
        str
            A string describing the model.
    print_info(self):
        Print the model information.
    make_3Dgrid(self):
        Create a 3D grid with the given source and model parameters.
        Raises:
        -------
        AttributeError
            If the source is not assigned to the model.
    plot(self):
        Plot the source location on a map.
        Raises:
        -------
        AttributeError
            If the source is not assigned to the model.
        ValueError
            If the source does not have 'latitude' and 'longitude' attributes.
    plot_grid(self, show_gridlines=False):
        Plot the 2D grid points on a map.
        Parameters:
        -----------
        show_gridlines : bool, optional
            Whether to show gridlines on the plot. Default is False.
        Raises:
        -------
        AttributeError
            If the 3D grid is not created.
    plot_grid_3d(self):
        Plot the 3D grid points.
        Raises:
        -------
        AttributeError
            If the 3D grid is not created.
    """
    pass
    def __init__(self, toml_file=None):
        if toml_file:
            self.load_from_toml(toml_file)
        else:
            self.name = "No-name model"
            self.radius = 100.0
            self.height = 500.0
            self.winds = False
            self.grid_spacing = 1.0
            self.height_spacing = 20.0 
            self.source = None
            self.atmosphere = None

    def load_from_toml(self, toml_file):
        data = toml.load(toml_file)
        model = data.get('model', {})
        self.name = model.get('name', "No-name model")
        self.radius = model.get('radius', 100.0)
        self.height = model.get('height', 500.0)
        self.winds = model.get('winds', False)
        self.atmosphere_model = model.get('atmosphere', "msise00")
        self.grid_spacing = model.get('grid_spacing', 1.0)
        units = model.get('grid_units')
        if units == "km":
            self.grid_spacing = self.grid_spacing / 111.32  # 1 degree is approximately 111.32 km at the equator
        self.height_spacing = model.get('height_spacing', 20.0) 
        

    def assign_source(self, source):
        if hasattr(self, 'source') and self.source is not None:
            print("Source already exists and will be updated.")
        self.source = source

    def assign_atmosphere(self):
        if not hasattr(self, 'source'):
            raise AttributeError("Source not assigned to the model.")
        if not hasattr(self, 'grid'):
            raise AttributeError("3D grid not created. Call make_3Dgrid() first.")
        
        self.atmosphere = Atmosphere1D(self.source.get_latitude(),
                                  self.source.get_longitude(), 
                                  self.grid.coords['altitude'].values, 
                                  self.source.get_time(), 
                                  self.atmosphere_model)


        
    def __str__(self):
        source_info = f"latitude = {self.source.get_latitude():.2f} (deg), longitude={self.source.get_longitude():.2f} (deg)" if hasattr(self, 'source') else "None"
        lat_extent_info = f"lat_extent = ({self.lat_extent[0]:.2f}, {self.lat_extent[1]:.2f})" if hasattr(self, 'lat_extent') else ""
        lon_extent_info = f"lon_extent = ({self.lon_extent[0]:.2f}, {self.lon_extent[1]:.2f})" if hasattr(self, 'lon_extent') else ""
        atmosphere_info = f"atmosphere_model = {self.atmosphere_model}" if hasattr(self, 'atmosphere_model') else ""
        return (f"Model3D: name = {self.name}\n radius = {self.radius} (km)\n height = {self.height} (km)\n"
                f" winds = {self.winds}\n {atmosphere_info}\n grid_spacing = {self.grid_spacing} (deg)\n height_spacing = {self.height_spacing} (km)\n"
                f" source: {source_info}\n {lat_extent_info}\n {lon_extent_info}")

    def print_info(self):
        print(self)

    def make_3Dgrid(self):
        """
        Create a 3D grid with the given source and model parameters.
        """
        if not hasattr(self, 'source'):
            raise AttributeError("Source not assigned to the model.")

        lat0 = self.source.get_latitude()
        lon0 = self.source.get_longitude()
        # Convert radius from kilometers to degrees (approximation)
        radius_in_degrees = self.radius / 111.32  # 1 degree is approximately 111.32 km at the equator

        # Create latitude and longitude arrays
        latitudes = np.arange(lat0 - radius_in_degrees, lat0 + radius_in_degrees, self.grid_spacing)
        longitudes = np.arange(lon0 - radius_in_degrees, lon0 + radius_in_degrees, self.grid_spacing)
        altitudes = np.arange(0, self.height, self.height_spacing)

        self.grid = xr.DataArray(
            np.zeros((len(latitudes), len(longitudes), len(altitudes))),
            coords=[latitudes, longitudes, altitudes],
            dims=["latitude", "longitude", "altitude"]
        )

        self.lat_extent = (latitudes[0], latitudes[-1])
        self.lon_extent = (longitudes[0], longitudes[-1])


    def plot_source(self):
        if not hasattr(self, 'source'):
            raise AttributeError("Source not assigned to the model.")
        
        lat = self.source.get_latitude()
        lon = self.source.get_longitude()
        
        if lat is None or lon is None:
            raise ValueError("Source must have 'latitude' and 'longitude' attributes.")
        
        # Convert radius from kilometers to degrees (approximation)
        radius_in_degrees = self.radius / 111.32  # 1 degree is approximately 111.32 km at the equator
        
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent([lon - radius_in_degrees, lon + radius_in_degrees, lat - radius_in_degrees, lat + radius_in_degrees], crs=ccrs.PlateCarree())
        
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        ax.add_feature(cfeature.RIVERS)
        
        ax.plot(lon, lat, 'k*', markersize=10, transform=ccrs.PlateCarree(), label='Source')
        # ax.text(lon, lat, ' Source', horizontalalignment='left', transform=ccrs.PlateCarree())
        
        plt.title(f"Source Location: ({lat:.2f}, {lon:.2f})")
        
        # Add scale bar
        # scale_bar(ax, (0.1, 0.05), 100)  # 100 km scale bar
        
        # Add legend
        ax.legend(loc='upper right')
        
        plt.show()

    # def scale_bar(ax, location, length):
    #     """
    #     Add a scale bar to the map.
        
    #     Parameters:
    #     ax : matplotlib axes object
    #     The axes to draw the scale bar on.
    #     location : tuple
    #     The location of the scale bar in axes coordinates (0 to 1).
    #     length : float
    #     The length of the scale bar in kilometers.
    #     """
    #     # Get the extent of the map in degrees
    #     extent = ax.get_extent(ccrs.PlateCarree())
    #     # Calculate the length of the scale bar in degrees
    #     length_in_degrees = length / 111.32  # 1 degree is approximately 111.32 km at the equator
        
    #     # Create a line for the scale bar
    #     line = plt.Line2D([location[0], location[0] + length_in_degrees], [location[1], location[1]], 
    #               transform=ax.transAxes, color='black', linewidth=2)
    #     ax.add_line(line)
        
    #     # Add text for the scale bar
    #     ax.text(location[0] + length_in_degrees / 2, location[1] - 0.02, f'{length} km', 
    #         transform=ax.transAxes, horizontalalignment='center', verticalalignment='top')

    def plot_grid(self, show_gridlines=False):
        if not hasattr(self, 'grid'):
            raise AttributeError("3D grid not created. Call make_3Dgrid() first.")
        
        latitudes = self.grid.coords['latitude'].values
        longitudes = self.grid.coords['longitude'].values
        
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent([longitudes[0] - self.grid_spacing, longitudes[-1] + self.grid_spacing, 
                   latitudes[0] - self.grid_spacing, latitudes[-1] + self.grid_spacing], 
                  crs=ccrs.PlateCarree())
        
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        ax.add_feature(cfeature.RIVERS)
        
        for lat in latitudes:
            for lon in longitudes:
                ax.plot(lon, lat, 'ro', markersize=2, transform=ccrs.PlateCarree())

        # Define the scale bar
        fontprops = fm.FontProperties(size=12)
        scalebar = AnchoredSizeBar(ax.transData,
                                1,  # Length of the scale bar in data units
                                '100 km',  # Label for the scale bar
                                'lower right',  # Location of the scale bar
                                pad=0.1,
                                color='black',
                                frameon=False,
                                size_vertical=0.1,
                                fontproperties=fontprops)
        
        # Add the scale bar to the plot
        ax.add_artist(scalebar)
        if show_gridlines:
            ax.gridlines(draw_labels=True)
        
        plt.title("Lat/Lon Grid Points")
        
        plt.show()

    def plot_grid_3d(self):
        if not hasattr(self, 'grid'):
            raise AttributeError("3D grid not created. Call make_3Dgrid() first.")
        
        latitudes = self.grid.coords['latitude'].values
        longitudes = self.grid.coords['longitude'].values
        altitudes = self.grid.coords['altitude'].values
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        lat_grid, lon_grid, alt_grid = np.meshgrid(latitudes, longitudes, altitudes, indexing='ij')
        
        ax.scatter(lon_grid, lat_grid, alt_grid, c='k', marker='o', s=1)
        
        ax.set_xlabel('Longitude (deg)')
        ax.set_ylabel('Latitude (deg)')
        ax.set_zlabel('Altitude (km)')
        
        plt.title("3D Grid Points")
        
        plt.show()
import toml
import xarray as xr
import numpy as np
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

class Model3D:
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

    def load_from_toml(self, toml_file):
        data = toml.load(toml_file)
        model = data.get('model', {})
        self.name = model.get('name', "No-name model")
        self.radius = model.get('radius', 100.0)
        self.height = model.get('height', 500.0)
        self.winds = model.get('winds', False)
        self.grid_spacing = model.get('grid_spacing', 1.0)
        units = model.get('grid_units')
        if units == "km":
            self.grid_spacing = self.grid_spacing / 111.32  # 1 degree is approximately 111.32 km at the equator
        self.height_spacing = model.get('height_spacing', 20.0) 
                        
    def assign_source(self, source):
        if hasattr(self, 'source') and self.source is not None:
            print("Source already exists and will be updated.")
        self.source = source

        
    def __str__(self):
        source_info = f"latitude={self.source.get_latitude()}, longitude={self.source.get_longitude()}" if hasattr(self, 'source') else "None"
        lat_extent_info = f"lat_extent={self.lat_extent}" if hasattr(self, 'lat_extent') else ""
        lon_extent_info = f"lon_extent={self.lon_extent}" if hasattr(self, 'lon_extent') else ""
        return (f"Model3D: name={self.name}, radius={self.radius} (km), height={self.height} (km), "
                f"winds={self.winds}, grid_spacing={self.grid_spacing} (deg), height_spacing={self.height_spacing} (km), "
                f"source={source_info} {lat_extent_info} {lon_extent_info}")

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


    def plot(self):
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
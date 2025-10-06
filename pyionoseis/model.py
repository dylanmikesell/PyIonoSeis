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
from pyionoseis.ionosphere import Ionosphere1D
from pyionoseis.igrf import MagneticField1D, PPIGRF_AVAILABLE

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
    assign_ionosphere(self, ionosphere_model="iri2020"):
        Assign ionospheric electron density to all points in the 3D model domain.
        Parameters:
        -----------
        ionosphere_model : str
            The ionospheric model to use (default: "iri2020")
    plot_variable(self, variable='grid_points', altitude_slice=None):
        Plot different variables on the model grid.
        Parameters:
        -----------
        variable : str
            Variable to plot: 'grid_points', 'electron_density', 'density', 
            'pressure', 'velocity', 'temperature'
        altitude_slice : float, optional
            Altitude level (km) for 2D maps
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
            self.atmosphere_model = "msise00"
            self.ionosphere_model = "iri2020"

    def load_from_toml(self, toml_file):
        data = toml.load(toml_file)
        model = data.get('model', {})
        self.name = model.get('name', "No-name model")
        self.radius = model.get('radius', 100.0)
        self.height = model.get('height', 500.0)
        self.winds = model.get('winds', False)
        self.atmosphere_model = model.get('atmosphere', "msise00")
        self.ionosphere_model = model.get('ionosphere', "iri2020")
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

    def assign_ionosphere(self, ionosphere_model="iri2020"):
        """
        Assign ionospheric electron density to all points in the 3D model domain.
        
        This method computes electron density using the specified ionospheric model
        for each latitude-longitude location in the 3D grid. The computation is done
        efficiently by computing vertical profiles for each lat-lon point.
        
        Parameters:
        -----------
        ionosphere_model : str
            The ionospheric model to use. Currently supports:
            - "iri2020": International Reference Ionosphere 2020 model (default)
            
        Raises:
        -------
        AttributeError
            If source is not assigned to the model or 3D grid is not created.
        ValueError
            If an unsupported ionosphere model is specified.
            
        Notes:
        ------
        The electron density is computed at each grid point and stored in the
        'electron_density' field of the grid DataArray. Units are m⁻³.
        
        The computation strategy:
        1. For each unique lat-lon pair in the 3D grid
        2. Compute a vertical electron density profile using the ionosphere model
        3. Interpolate/assign the profile to all altitude levels at that location
        4. Store results in the 3D grid structure
        """
        if not hasattr(self, 'source'):
            raise AttributeError("Source not assigned to the model.")
        if not hasattr(self, 'grid'):
            raise AttributeError("3D grid not created. Call make_3Dgrid() first.")
            
        # Supported ionosphere models
        supported_models = ["iri2020"]
        if ionosphere_model not in supported_models:
            raise ValueError(f"Unsupported ionosphere model '{ionosphere_model}'. "
                           f"Supported models: {supported_models}")
        
        # Get grid coordinates
        latitudes = self.grid.coords['latitude'].values
        longitudes = self.grid.coords['longitude'].values
        altitudes = self.grid.coords['altitude'].values
        
        # Get source time
        time = self.source.get_time()
        
        print(f"Computing ionospheric electron density using {ionosphere_model}...")
        print(f"Grid dimensions: {len(latitudes)} lat × {len(longitudes)} lon × {len(altitudes)} alt")
        
        # Initialize the electron density array
        electron_density_3d = np.zeros((len(latitudes), len(longitudes), len(altitudes)))
        
        # Compute electron density for each lat-lon location
        total_profiles = len(latitudes) * len(longitudes)
        profile_count = 0
        
        for i, lat in enumerate(latitudes):
            for j, lon in enumerate(longitudes):
                profile_count += 1
                
                # Show progress for large grids
                if total_profiles > 10 and profile_count % max(1, total_profiles // 10) == 0:
                    print(f"  Progress: {profile_count}/{total_profiles} "
                          f"({100*profile_count/total_profiles:.0f}%) profiles computed")
                
                try:
                    # Create 1D ionosphere model for this location
                    iono_1d = Ionosphere1D(lat, lon, altitudes, time, ionosphere_model)
                    
                    # Extract electron density profile
                    ne_profile = iono_1d.ionosphere.electron_density.values
                    
                    # Assign to 3D grid
                    electron_density_3d[i, j, :] = ne_profile
                    
                except Exception as e:
                    print(f"Warning: Failed to compute ionosphere at lat={lat:.2f}, lon={lon:.2f}: {e}")
                    # Fill with NaN for failed computations
                    electron_density_3d[i, j, :] = np.nan
        
        print("Ionospheric computation completed.")
        
        # Convert DataArray to Dataset and add electron density
        if isinstance(self.grid, xr.DataArray):
            # Convert to Dataset with the original data as 'grid_points'
            grid_dataset = self.grid.to_dataset(name='grid_points')
            # Add electron density as a new variable
            grid_dataset['electron_density'] = (["latitude", "longitude", "altitude"], electron_density_3d)
            self.grid = grid_dataset
        else:
            # If already a Dataset, use assign
            self.grid = self.grid.assign(
                electron_density=(["latitude", "longitude", "altitude"], electron_density_3d)
            )
        
        # Update grid attributes
        if not hasattr(self.grid, 'attrs'):
            self.grid.attrs = {}
        self.grid.attrs['ionosphere_model'] = ionosphere_model
        self.grid.attrs['electron_density_units'] = 'm⁻³'
        self.grid.attrs['electron_density_description'] = f'Electron density from {ionosphere_model} model'
        
        # Store ionosphere model info
        self.ionosphere_model = ionosphere_model
        
    def assign_magnetic_field(self, magnetic_field_model="igrf"):
        """
        Assign magnetic field parameters to all points in the 3D model domain.
        
        This method computes magnetic field components (Be, Bn, Bu) and derived
        parameters (inclination, declination) at each grid point using the
        specified magnetic field model.
        
        Parameters:
        -----------
        magnetic_field_model : str
            The magnetic field model to use. Currently supports:
            - "igrf": International Geomagnetic Reference Field model (default)
            
        Raises:
        -------
        AttributeError
            If source is not assigned to the model or 3D grid is not created.
        ValueError
            If an unsupported magnetic field model is specified.
            
        Notes:
        ------
        The magnetic field components are computed at each grid point and stored in the
        grid Dataset. The following variables are added:
        
        Geodetic components (tangent to Earth's ellipsoid):
        - 'Be': East component of magnetic field (nT)
        - 'Bn': North component of magnetic field (nT) 
        - 'Bu': Up component of magnetic field (nT)
        
        Geocentric components (spherical coordinates):
        - 'Br': Radial component of magnetic field (nT)
        - 'Btheta': Colatitude component of magnetic field (nT)
        - 'Bphi': Azimuth component of magnetic field (nT)
        
        Derived parameters:
        - 'inclination': Magnetic inclination angle (degrees)
        - 'declination': Magnetic declination angle (degrees)
        
        The computation strategy:
        1. For each unique lat-lon pair in the 3D grid
        2. Compute magnetic field profile using specified model
        3. Assign computed values to all altitude levels at that location
        """
        if not hasattr(self, 'source'):
            raise AttributeError("Source not assigned to the model.")
        if not hasattr(self, 'grid'):
            raise AttributeError("3D grid not created. Call make_3Dgrid() first.")
        
        if magnetic_field_model.lower() != "igrf":
            raise ValueError(f"Unsupported magnetic field model: {magnetic_field_model}")
            
        if not PPIGRF_AVAILABLE:
            raise ImportError("ppigrf package is not available. Please install it with: pip install ppigrf")
        
        print(f"Computing magnetic field using {magnetic_field_model}...")
        
        # Get grid dimensions
        latitudes = self.grid.coords['latitude'].values
        longitudes = self.grid.coords['longitude'].values
        altitudes = self.grid.coords['altitude'].values
        
        print(f"Grid dimensions: {len(latitudes)} lat × {len(longitudes)} lon × {len(altitudes)} alt")
        
        # Initialize 3D arrays for magnetic field parameters
        Be_3d = np.full((len(latitudes), len(longitudes), len(altitudes)), np.nan)
        Bn_3d = np.full((len(latitudes), len(longitudes), len(altitudes)), np.nan)
        Bu_3d = np.full((len(latitudes), len(longitudes), len(altitudes)), np.nan)
        Br_3d = np.full((len(latitudes), len(longitudes), len(altitudes)), np.nan)
        Btheta_3d = np.full((len(latitudes), len(longitudes), len(altitudes)), np.nan)
        Bphi_3d = np.full((len(latitudes), len(longitudes), len(altitudes)), np.nan)
        inclination_3d = np.full((len(latitudes), len(longitudes), len(altitudes)), np.nan)
        declination_3d = np.full((len(latitudes), len(longitudes), len(altitudes)), np.nan)
        total_field_3d = np.full((len(latitudes), len(longitudes), len(altitudes)), np.nan)
        horizontal_intensity_3d = np.full((len(latitudes), len(longitudes), len(altitudes)), np.nan)
        
        # Compute magnetic field for each lat-lon pair
        total_profiles = len(latitudes) * len(longitudes)
        computed_profiles = 0
        
        for i, lat in enumerate(latitudes):
            for j, lon in enumerate(longitudes):
                try:
                    # Create magnetic field model for this location
                    mag_field = MagneticField1D(lat, lon, altitudes, self.source.get_time(), magnetic_field_model)
                    
                    # Extract computed parameters
                    Be_profile = mag_field.magnetic_field['Be'].values
                    Bn_profile = mag_field.magnetic_field['Bn'].values
                    Bu_profile = mag_field.magnetic_field['Bu'].values
                    Br_profile = mag_field.magnetic_field['Br'].values
                    Btheta_profile = mag_field.magnetic_field['Btheta'].values
                    Bphi_profile = mag_field.magnetic_field['Bphi'].values
                    inc_profile = mag_field.magnetic_field['inclination'].values
                    dec_profile = mag_field.magnetic_field['declination'].values
                    total_field_profile = mag_field.magnetic_field['total_field'].values
                    horizontal_intensity_profile = mag_field.magnetic_field['horizontal_intensity'].values
                    
                    # Assign to 3D grids
                    Be_3d[i, j, :] = Be_profile
                    Bn_3d[i, j, :] = Bn_profile
                    Bu_3d[i, j, :] = Bu_profile
                    Br_3d[i, j, :] = Br_profile
                    Btheta_3d[i, j, :] = Btheta_profile
                    Bphi_3d[i, j, :] = Bphi_profile
                    inclination_3d[i, j, :] = inc_profile
                    declination_3d[i, j, :] = dec_profile
                    total_field_3d[i, j, :] = total_field_profile
                    horizontal_intensity_3d[i, j, :] = horizontal_intensity_profile
                    
                    computed_profiles += 1
                    
                except Exception as e:
                    print(f"Warning: Failed to compute magnetic field at lat={lat:.2f}, lon={lon:.2f}: {e}")
                    # NaN values are already assigned above
                    
                # Progress reporting
                if computed_profiles % max(1, total_profiles // 10) == 0:
                    progress = int(100 * computed_profiles / total_profiles)
                    print(f"  Progress: {computed_profiles}/{total_profiles} ({progress}%) profiles computed")
        
        print("Magnetic field computation completed.")
        
        # Add magnetic field parameters to the grid Dataset
        if isinstance(self.grid, xr.DataArray):
            # Convert to Dataset with the original data as 'grid_points'
            grid_dataset = self.grid.to_dataset(name='grid_points')
            # Add magnetic field variables
            grid_dataset['Be'] = (["latitude", "longitude", "altitude"], Be_3d)
            grid_dataset['Bn'] = (["latitude", "longitude", "altitude"], Bn_3d)
            grid_dataset['Bu'] = (["latitude", "longitude", "altitude"], Bu_3d)
            grid_dataset['Br'] = (["latitude", "longitude", "altitude"], Br_3d)
            grid_dataset['Btheta'] = (["latitude", "longitude", "altitude"], Btheta_3d)
            grid_dataset['Bphi'] = (["latitude", "longitude", "altitude"], Bphi_3d)
            grid_dataset['inclination'] = (["latitude", "longitude", "altitude"], inclination_3d)
            grid_dataset['declination'] = (["latitude", "longitude", "altitude"], declination_3d)
            grid_dataset['total_field'] = (["latitude", "longitude", "altitude"], total_field_3d)
            grid_dataset['horizontal_intensity'] = (["latitude", "longitude", "altitude"], horizontal_intensity_3d)
            self.grid = grid_dataset
        else:
            # If already a Dataset, add new variables
            self.grid['Be'] = (["latitude", "longitude", "altitude"], Be_3d)
            self.grid['Bn'] = (["latitude", "longitude", "altitude"], Bn_3d)
            self.grid['Bu'] = (["latitude", "longitude", "altitude"], Bu_3d)
            self.grid['Br'] = (["latitude", "longitude", "altitude"], Br_3d)
            self.grid['Btheta'] = (["latitude", "longitude", "altitude"], Btheta_3d)
            self.grid['Bphi'] = (["latitude", "longitude", "altitude"], Bphi_3d)
            self.grid['inclination'] = (["latitude", "longitude", "altitude"], inclination_3d)
            self.grid['declination'] = (["latitude", "longitude", "altitude"], declination_3d)
            self.grid['total_field'] = (["latitude", "longitude", "altitude"], total_field_3d)
            self.grid['horizontal_intensity'] = (["latitude", "longitude", "altitude"], horizontal_intensity_3d)
        
        # Update grid attributes
        if not hasattr(self.grid, 'attrs'):
            self.grid.attrs = {}
        self.grid.attrs['magnetic_field_model'] = magnetic_field_model
        self.grid.attrs['magnetic_field_units'] = 'nT'
        self.grid.attrs['magnetic_angle_units'] = 'degrees'
        self.grid.attrs['Be_description'] = f'East component of magnetic field from {magnetic_field_model} model'
        self.grid.attrs['Bn_description'] = f'North component of magnetic field from {magnetic_field_model} model'
        self.grid.attrs['Bu_description'] = f'Up component of magnetic field from {magnetic_field_model} model'
        self.grid.attrs['Br_description'] = f'Radial component of magnetic field from {magnetic_field_model} model (geocentric)'
        self.grid.attrs['Btheta_description'] = f'Colatitude component of magnetic field from {magnetic_field_model} model (geocentric)'
        self.grid.attrs['Bphi_description'] = f'Azimuth component of magnetic field from {magnetic_field_model} model (geocentric)'
        self.grid.attrs['inclination_description'] = f'Magnetic inclination from {magnetic_field_model} model'
        self.grid.attrs['declination_description'] = f'Magnetic declination from {magnetic_field_model} model'
        
        # Store magnetic field model info
        self.magnetic_field_model = magnetic_field_model


        
    def __str__(self):
        source_info = f"latitude = {self.source.get_latitude():.2f} (deg), longitude={self.source.get_longitude():.2f} (deg)" if hasattr(self, 'source') else "None"
        lat_extent_info = f"lat_extent = ({self.lat_extent[0]:.2f}, {self.lat_extent[1]:.2f})" if hasattr(self, 'lat_extent') else ""
        lon_extent_info = f"lon_extent = ({self.lon_extent[0]:.2f}, {self.lon_extent[1]:.2f})" if hasattr(self, 'lon_extent') else ""
        atmosphere_info = f"atmosphere_model = {self.atmosphere_model}" if hasattr(self, 'atmosphere_model') else ""
        ionosphere_info = f"ionosphere_model = {self.ionosphere_model}" if hasattr(self, 'ionosphere_model') else ""
        
        base_info = (f"Model3D: name = {self.name}\n radius = {self.radius} (km)\n height = {self.height} (km)\n"
                     f" winds = {self.winds}\n {atmosphere_info}\n {ionosphere_info}\n grid_spacing = {self.grid_spacing} (deg)\n height_spacing = {self.height_spacing} (km)\n"
                     f" source: {source_info}\n {lat_extent_info}\n {lon_extent_info}")
        
        return base_info

    def print_info(self):
        """
        Print comprehensive information about the Model3D object.
        
        This method provides detailed information about:
        - Basic model parameters
        - Source configuration
        - Grid dimensions and extent
        - Available data variables
        - Model states (atmosphere, ionosphere, magnetic field)
        """
        print("=" * 80)
        print("MODEL3D INFORMATION")
        print("=" * 80)
        
        # Basic Model Information
        print(f"Model Name: {self.name}")
        print(f"Model Radius: {self.radius} km")
        print(f"Model Height: {self.height} km")
        print(f"Include Winds: {self.winds}")
        print(f"Grid Spacing: {self.grid_spacing} degrees")
        print(f"Height Spacing: {self.height_spacing} km")
        
        # Source Information
        print("\nSOURCE INFORMATION:")
        if hasattr(self, 'source') and self.source is not None:
            print(f"  Latitude: {self.source.get_latitude():.4f}°")
            print(f"  Longitude: {self.source.get_longitude():.4f}°")
            print(f"  Time: {self.source.get_time()}")
            if hasattr(self.source, 'get_depth'):
                print(f"  Depth: {self.source.get_depth()} km")
        else:
            print("  No source assigned")
        
        # Grid Information
        print("\nGRID INFORMATION:")
        if hasattr(self, 'grid') and self.grid is not None:
            if hasattr(self, 'lat_extent'):
                print(f"  Latitude extent: {self.lat_extent[0]:.4f}° to {self.lat_extent[1]:.4f}°")
            if hasattr(self, 'lon_extent'):
                print(f"  Longitude extent: {self.lon_extent[0]:.4f}° to {self.lon_extent[1]:.4f}°")
            
            # Grid dimensions
            if hasattr(self.grid, 'sizes'):
                # This is an xarray Dataset/DataArray with sizes
                grid_shape = self.grid.sizes
                print(f"  Grid dimensions: {dict(grid_shape)}")
                total_points = 1
                for dim, size in grid_shape.items():
                    total_points *= size
                print(f"  Total grid points: {total_points:,}")
            elif hasattr(self.grid, 'shape'):
                # This is a numpy array or similar
                grid_shape = self.grid.shape
                print(f"  Grid shape: {grid_shape}")
                total_points = 1
                for dim in grid_shape:
                    total_points *= dim
                print(f"  Total grid points: {total_points:,}")
            else:
                print("  Grid shape: Unknown")
            
            # Coordinate ranges
            if hasattr(self.grid, 'coords'):
                coords = self.grid.coords
                if 'latitude' in coords:
                    lat_vals = coords['latitude'].values
                    print(f"  Latitude range: {lat_vals.min():.4f}° to {lat_vals.max():.4f}° ({len(lat_vals)} points)")
                if 'longitude' in coords:
                    lon_vals = coords['longitude'].values
                    print(f"  Longitude range: {lon_vals.min():.4f}° to {lon_vals.max():.4f}° ({len(lon_vals)} points)")
                if 'altitude' in coords:
                    alt_vals = coords['altitude'].values
                    print(f"  Altitude range: {alt_vals.min():.1f} to {alt_vals.max():.1f} km ({len(alt_vals)} points)")
        else:
            print("  No grid created yet")
        
        # Available Data Variables
        print("\nAVAILABLE DATA VARIABLES:")
        if hasattr(self, 'grid') and self.grid is not None and hasattr(self.grid, 'data_vars'):
            if len(self.grid.data_vars) > 0:
                # Group variables by type
                atmospheric_vars = []
                ionospheric_vars = []
                magnetic_geodetic_vars = []
                magnetic_geocentric_vars = []
                magnetic_derived_vars = []
                other_vars = []
                
                for var in self.grid.data_vars:
                    if var in ['density', 'pressure', 'temperature', 'velocity']:
                        atmospheric_vars.append(var)
                    elif var in ['electron_density']:
                        ionospheric_vars.append(var)
                    elif var in ['Be', 'Bn', 'Bu']:
                        magnetic_geodetic_vars.append(var)
                    elif var in ['Br', 'Btheta', 'Bphi']:
                        magnetic_geocentric_vars.append(var)
                    elif var in ['inclination', 'declination', 'total_field', 'horizontal_intensity']:
                        magnetic_derived_vars.append(var)
                    else:
                        other_vars.append(var)
                
                if atmospheric_vars:
                    print(f"  Atmospheric variables: {', '.join(atmospheric_vars)}")
                if ionospheric_vars:
                    print(f"  Ionospheric variables: {', '.join(ionospheric_vars)}")
                if magnetic_geodetic_vars:
                    print(f"  Magnetic field (geodetic): {', '.join(magnetic_geodetic_vars)}")
                if magnetic_geocentric_vars:
                    print(f"  Magnetic field (geocentric): {', '.join(magnetic_geocentric_vars)}")
                if magnetic_derived_vars:
                    print(f"  Magnetic field (derived): {', '.join(magnetic_derived_vars)}")
                if other_vars:
                    print(f"  Other variables: {', '.join(other_vars)}")
                
                print(f"  Total variables: {len(self.grid.data_vars)}")
                
                # Show variable ranges for numeric data
                print("\n  Variable Ranges:")
                for var in sorted(self.grid.data_vars):
                    if var != 'grid_points':  # Skip the grid_points variable
                        try:
                            data = self.grid[var]
                            if data.dtype.kind in ['f', 'i']:  # float or integer
                                min_val = float(data.min().values)
                                max_val = float(data.max().values)
                                units = ""
                                if 'nT' in str(data.attrs.get('units', '')) or var in ['Be', 'Bn', 'Bu', 'Br', 'Btheta', 'Bphi', 'total_field', 'horizontal_intensity']:
                                    units = " nT"
                                elif var in ['inclination', 'declination']:
                                    units = "°"
                                elif var in ['density']:
                                    units = " kg/m³"
                                elif var in ['pressure']:
                                    units = " Pa"
                                elif var in ['temperature']:
                                    units = " K"
                                elif var in ['electron_density']:
                                    units = " /m³"
                                print(f"    {var}: {min_val:.2f} to {max_val:.2f}{units}")
                        except Exception:
                            print(f"    {var}: [data present]")
            else:
                print("  No data variables computed yet")
        else:
            print("  No data variables (grid not created)")
        
        # Model States
        print("\nMODEL STATES:")
        
        # Atmosphere model
        if hasattr(self, 'atmosphere_model'):
            print(f"  Atmosphere model: {self.atmosphere_model}")
            if hasattr(self, 'grid') and self.grid is not None and hasattr(self.grid, 'data_vars'):
                atm_vars = [v for v in self.grid.data_vars if v in ['density', 'pressure', 'temperature', 'velocity']]
                print(f"    Status: {'Computed' if atm_vars else 'Not computed'}")
                if atm_vars:
                    print(f"    Variables: {', '.join(atm_vars)}")
        
        # Ionosphere model
        if hasattr(self, 'ionosphere_model'):
            print(f"  Ionosphere model: {self.ionosphere_model}")
            if hasattr(self, 'grid') and self.grid is not None and hasattr(self.grid, 'data_vars'):
                iono_computed = 'electron_density' in self.grid.data_vars
                print(f"    Status: {'Computed' if iono_computed else 'Not computed'}")
        
        # Magnetic field model
        if hasattr(self, 'magnetic_field_model'):
            print(f"  Magnetic field model: {self.magnetic_field_model}")
            if hasattr(self, 'grid') and self.grid is not None and hasattr(self.grid, 'data_vars'):
                mag_vars = [v for v in self.grid.data_vars if v in ['Be', 'Bn', 'Bu', 'Br', 'Btheta', 'Bphi', 'inclination', 'declination']]
                print(f"    Status: {'Computed' if mag_vars else 'Not computed'}")
                if mag_vars:
                    geodetic = [v for v in mag_vars if v in ['Be', 'Bn', 'Bu']]
                    geocentric = [v for v in mag_vars if v in ['Br', 'Btheta', 'Bphi']]
                    derived = [v for v in mag_vars if v in ['inclination', 'declination']]
                    if geodetic:
                        print(f"    Geodetic components: {', '.join(geodetic)}")
                    if geocentric:
                        print(f"    Geocentric components: {', '.join(geocentric)}")
                    if derived:
                        print(f"    Derived parameters: {', '.join(derived)}")
        
        # Grid attributes
        if hasattr(self, 'grid') and self.grid is not None and hasattr(self.grid, 'attrs') and self.grid.attrs:
            print("\nGRID ATTRIBUTES:")
            for key, value in self.grid.attrs.items():
                if not key.endswith('_description'):  # Skip long description attributes
                    print(f"  {key}: {value}")
        
        print("=" * 80)

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
        
    def plot_variable(self, variable='grid_points', altitude_slice=None, **kwargs):
        """
        Plot different variables on the model grid.
        
        Parameters:
        -----------
        variable : str
            The variable to plot. Options:
            - 'grid_points': Plot grid points only (default)
            - 'electron_density': Plot electron density (requires assign_ionosphere)
            - 'Be': Plot east magnetic field component (requires assign_magnetic_field)
            - 'Bn': Plot north magnetic field component (requires assign_magnetic_field)
            - 'Bu': Plot up magnetic field component (requires assign_magnetic_field)
            - 'Br': Plot radial magnetic field component (requires assign_magnetic_field)
            - 'Btheta': Plot colatitude magnetic field component (requires assign_magnetic_field)
            - 'Bphi': Plot azimuth magnetic field component (requires assign_magnetic_field)
            - 'inclination': Plot magnetic inclination (requires assign_magnetic_field)
            - 'declination': Plot magnetic declination (requires assign_magnetic_field)
            - 'density': Plot atmospheric density (requires assign_atmosphere)
            - 'pressure': Plot atmospheric pressure (requires assign_atmosphere)
            - 'velocity': Plot atmospheric velocity (requires assign_atmosphere)
            - 'temperature': Plot atmospheric temperature (requires assign_atmosphere)
        altitude_slice : float, optional
            Altitude level (km) to plot for 2D maps. If None, plots all grid points or
            vertical profiles depending on the variable.
        **kwargs : dict
            Additional plotting parameters passed to matplotlib functions.
            
        Examples:
        ---------
        # Plot grid points
        model.plot_variable('grid_points')
        
        # Plot electron density at 300 km altitude
        model.plot_variable('electron_density', altitude_slice=300)
        
        # Plot magnetic inclination at 100 km altitude
        model.plot_variable('inclination', altitude_slice=100)
        
        # Plot vertical profile at source location
        model.plot_variable('electron_density')
        """
        if not hasattr(self, 'grid'):
            raise AttributeError("3D grid not created. Call make_3Dgrid() first.")
            
        if variable == 'grid_points':
            # Plot grid points (existing functionality)
            if altitude_slice is None:
                self.plot_grid_3d()
            else:
                self.plot_grid()
                
        elif variable == 'electron_density':
            self._plot_electron_density(altitude_slice, **kwargs)
            
        elif variable in ['Be', 'Bn', 'Bu', 'Br', 'Btheta', 'Bphi', 'inclination', 'declination']:
            self._plot_magnetic_field_variable(variable, altitude_slice, **kwargs)
            
        elif variable in ['density', 'pressure', 'velocity', 'temperature']:
            self._plot_atmospheric_variable(variable, altitude_slice, **kwargs)
            
        else:
            available_vars = ['grid_points', 'electron_density', 'Be', 'Bn', 'Bu', 'inclination', 'declination', 'density', 'pressure', 'velocity', 'temperature']
            raise ValueError(f"Unknown variable '{variable}'. Available variables: {available_vars}")
    
    def _plot_electron_density(self, altitude_slice=None, **kwargs):
        """Plot electron density."""
        if 'electron_density' not in self.grid.data_vars:
            raise AttributeError("Electron density not computed. Call assign_ionosphere() first.")
        
        if altitude_slice is not None:
            # Plot 2D map at specified altitude
            self._plot_2d_map('electron_density', altitude_slice, 
                            title_prefix='Electron Density', 
                            units='m⁻³', log_scale=True, **kwargs)
        else:
            # Plot vertical profile at source location
            self._plot_vertical_profile('electron_density', 
                                      title_prefix='Electron Density',
                                      units='m⁻³', log_scale=True, **kwargs)
    
    def _plot_atmospheric_variable(self, variable, altitude_slice=None, **kwargs):
        """Plot atmospheric variables."""
        if not hasattr(self, 'atmosphere'):
            raise AttributeError("Atmospheric data not computed. Call assign_atmosphere() first.")
        
        # For atmospheric variables, we only have 1D profiles at source location
        if altitude_slice is not None:
            print("Warning: altitude_slice not supported for atmospheric variables. "
                  "Atmospheric data is only available as 1D profile at source location.")
        
        # Plot 1D atmospheric profile
        fig, ax = plt.subplots(1, 1, figsize=(8, 10))
        
        atmos_data = self.atmosphere.atmosphere[variable]
        altitudes = self.atmosphere.alt_km
        
        if variable in ['density', 'pressure']:
            ax.semilogx(atmos_data, altitudes, **kwargs)
        else:
            ax.plot(atmos_data, altitudes, **kwargs)
            
        var_labels = {
            'density': 'Density (kg/m³)',
            'pressure': 'Pressure (Pa)', 
            'velocity': 'Velocity (km/s)',
            'temperature': 'Temperature (K)'
        }
        
        ax.set_xlabel(var_labels.get(variable, variable))
        ax.set_ylabel('Altitude (km)')
        ax.set_title(f'Atmospheric {variable.title()} Profile\n'
                    f'Lat: {self.source.get_latitude():.2f}°, '
                    f'Lon: {self.source.get_longitude():.2f}°')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_magnetic_field_variable(self, variable, altitude_slice=None, **kwargs):
        """Plot magnetic field variables."""
        if variable not in self.grid.data_vars:
            raise AttributeError(f"Magnetic field variable '{variable}' not computed. Call assign_magnetic_field() first.")
        
        # Define variable properties
        var_properties = {
            'Be': {'title': 'East Magnetic Field Component', 'units': 'nT', 'log_scale': False},
            'Bn': {'title': 'North Magnetic Field Component', 'units': 'nT', 'log_scale': False},
            'Bu': {'title': 'Up Magnetic Field Component', 'units': 'nT', 'log_scale': False},
            'Br': {'title': 'Radial Magnetic Field Component', 'units': 'nT', 'log_scale': False},
            'Btheta': {'title': 'Colatitude Magnetic Field Component', 'units': 'nT', 'log_scale': False},
            'Bphi': {'title': 'Azimuth Magnetic Field Component', 'units': 'nT', 'log_scale': False},
            'inclination': {'title': 'Magnetic Inclination', 'units': 'degrees', 'log_scale': False},
            'declination': {'title': 'Magnetic Declination', 'units': 'degrees', 'log_scale': False}
        }
        
        props = var_properties.get(variable, {'title': variable.title(), 'units': '', 'log_scale': False})
        
        if altitude_slice is not None:
            # Plot 2D map at specified altitude
            self._plot_2d_map(variable, altitude_slice, 
                            title_prefix=props['title'], 
                            units=props['units'], 
                            log_scale=props['log_scale'], **kwargs)
        else:
            # Plot vertical profile at source location
            self._plot_vertical_profile(variable, 
                                      title_prefix=props['title'],
                                      units=props['units'], 
                                      log_scale=props['log_scale'], **kwargs)
    
    def _plot_2d_map(self, variable, altitude, title_prefix, units, log_scale=False, **kwargs):
        """Plot 2D map of a variable at specified altitude."""
        # Find closest altitude level
        altitudes = self.grid.coords['altitude'].values
        alt_idx = np.argmin(np.abs(altitudes - altitude))
        actual_altitude = altitudes[alt_idx]
        
        # Extract 2D slice
        data_2d = self.grid[variable].isel(altitude=alt_idx)
        
        # Create map
        latitudes = self.grid.coords['latitude'].values
        longitudes = self.grid.coords['longitude'].values
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8), 
                              subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Set map extent
        extent = [longitudes.min(), longitudes.max(), 
                 latitudes.min(), latitudes.max()]
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, alpha=0.3)
        ax.add_feature(cfeature.OCEAN, alpha=0.3)
        
        # Create meshgrid for plotting
        lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
        
        # Plot data
        if log_scale:
            im = ax.contourf(lon_grid, lat_grid, data_2d.values, 
                           levels=20, transform=ccrs.PlateCarree(), 
                           norm=plt.cm.colors.LogNorm(), **kwargs)
        else:
            im = ax.contourf(lon_grid, lat_grid, data_2d.values, 
                           levels=20, transform=ccrs.PlateCarree(), **kwargs)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(f'{title_prefix} ({units})')
        
        # Plot source location
        if hasattr(self, 'source'):
            ax.plot(self.source.get_longitude(), self.source.get_latitude(), 
                   'k*', markersize=15, transform=ccrs.PlateCarree(), 
                   label='Source')
            ax.legend()
        
        # Add gridlines
        ax.gridlines(draw_labels=True, alpha=0.5)
        
        plt.title(f'{title_prefix} at {actual_altitude:.1f} km\n'
                 f'Time: {self.source.get_time() if hasattr(self, "source") else "N/A"}')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_vertical_profile(self, variable, title_prefix, units, log_scale=False, **kwargs):
        """Plot vertical profile of a variable at source location."""
        if not hasattr(self, 'source'):
            raise AttributeError("Source not assigned to the model.")
        
        # Find closest grid point to source
        latitudes = self.grid.coords['latitude'].values
        longitudes = self.grid.coords['longitude'].values
        
        lat_idx = np.argmin(np.abs(latitudes - self.source.get_latitude()))
        lon_idx = np.argmin(np.abs(longitudes - self.source.get_longitude()))
        
        # Extract vertical profile
        profile = self.grid[variable].isel(latitude=lat_idx, longitude=lon_idx)
        altitudes = self.grid.coords['altitude'].values
        
        # Plot profile
        fig, ax = plt.subplots(1, 1, figsize=(8, 10))
        
        if log_scale:
            ax.semilogx(profile.values, altitudes, **kwargs)
        else:
            ax.plot(profile.values, altitudes, **kwargs)
            
        ax.set_xlabel(f'{title_prefix} ({units})')
        ax.set_ylabel('Altitude (km)')
        ax.set_title(f'{title_prefix} Vertical Profile\n'
                    f'Lat: {self.source.get_latitude():.2f}°, '
                    f'Lon: {self.source.get_longitude():.2f}°, '
                    f'Time: {self.source.get_time()}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
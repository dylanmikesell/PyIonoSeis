"""Plotting mixin for Model3D.

All visualisation methods live here so that ``model.py`` stays focused on
physics orchestration. ``Model3D`` inherits from ``ModelPlotMixin``; no
public API change is required.
"""

from __future__ import annotations

import matplotlib.colors
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


class ModelPlotMixin:
    """Mixin that provides all plotting methods for ``Model3D``.

    Methods
    -------
    plot_source()
        Plot the earthquake source location on a cartopy map.
    plot_grid(show_gridlines=False)
        Plot 2-D lat/lon grid points on a cartopy map.
    plot_grid_3d()
        Scatter-plot the full 3-D grid in a matplotlib 3D axes.
    plot_variable(variable, altitude_slice, **kwargs)
        Unified entry point for all variable plots.
    """

    # ------------------------------------------------------------------
    # Public plot methods
    # ------------------------------------------------------------------

    def plot_source(self):
        """Plot the earthquake source location on a cartopy map.

        Raises
        ------
        AttributeError
            If no source has been assigned.
        ValueError
            If the source has no latitude / longitude.
        """
        if not hasattr(self, "source"):
            raise AttributeError("Source not assigned to the model.")

        lat = self.source.get_latitude()
        lon = self.source.get_longitude()

        if lat is None or lon is None:
            raise ValueError("Source must have 'latitude' and 'longitude' attributes.")

        radius_in_degrees = self.radius / 111.32

        fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
        ax.set_extent(
            [
                lon - radius_in_degrees,
                lon + radius_in_degrees,
                lat - radius_in_degrees,
                lat + radius_in_degrees,
            ],
            crs=ccrs.PlateCarree(),
        )

        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        ax.add_feature(cfeature.RIVERS)

        ax.plot(lon, lat, "k*", markersize=10, transform=ccrs.PlateCarree(), label="Source")
        ax.legend(loc="upper right")

        plt.title(f"Source Location: ({lat:.2f}, {lon:.2f})")
        plt.show()

    def plot_grid(self, show_gridlines=False):
        """Plot 2-D lat/lon grid points on a cartopy map.

        Parameters
        ----------
        show_gridlines : bool, optional
            Draw labelled gridlines. Default is False.

        Raises
        ------
        AttributeError
            If the 3-D grid has not been created yet.
        """
        if not hasattr(self, "grid"):
            raise AttributeError("3D grid not created. Call make_3Dgrid() first.")

        latitudes = self.grid.coords["latitude"].values
        longitudes = self.grid.coords["longitude"].values

        fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
        ax.set_extent(
            [
                longitudes[0] - self.grid_spacing,
                longitudes[-1] + self.grid_spacing,
                latitudes[0] - self.grid_spacing,
                latitudes[-1] + self.grid_spacing,
            ],
            crs=ccrs.PlateCarree(),
        )

        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        ax.add_feature(cfeature.RIVERS)

        for lat in latitudes:
            for lon in longitudes:
                ax.plot(lon, lat, "ro", markersize=2, transform=ccrs.PlateCarree())

        fontprops = fm.FontProperties(size=12)
        scalebar = AnchoredSizeBar(
            ax.transData,
            1,
            "100 km",
            "lower right",
            pad=0.1,
            color="black",
            frameon=False,
            size_vertical=0.1,
            fontproperties=fontprops,
        )
        ax.add_artist(scalebar)

        if show_gridlines:
            ax.gridlines(draw_labels=True)

        plt.title("Lat/Lon Grid Points")
        plt.show()

    def plot_grid_3d(self):
        """Scatter-plot the full 3-D grid in a matplotlib 3D axes.

        Raises
        ------
        AttributeError
            If the 3-D grid has not been created yet.
        """
        if not hasattr(self, "grid"):
            raise AttributeError("3D grid not created. Call make_3Dgrid() first.")

        latitudes = self.grid.coords["latitude"].values
        longitudes = self.grid.coords["longitude"].values
        altitudes = self.grid.coords["altitude"].values

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        lat_grid, lon_grid, alt_grid = np.meshgrid(
            latitudes, longitudes, altitudes, indexing="ij"
        )

        ax.scatter(lon_grid, lat_grid, alt_grid, c="k", marker="o", s=1)
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        ax.set_zlabel("Altitude (km)")
        plt.title("3D Grid Points")
        plt.show()

    def plot_variable(self, variable="grid_points", altitude_slice=None, **kwargs):
        """Unified entry point for all variable plots.

        Parameters
        ----------
        variable : str
            Variable to plot. Options:

            - ``'grid_points'`` — geometry only (default)
            - ``'electron_density'`` — requires :meth:`assign_ionosphere`
            - ``'Be'``, ``'Bn'``, ``'Bu'``, ``'Br'``, ``'Btheta'``, ``'Bphi'``,
              ``'inclination'``, ``'declination'`` — requires :meth:`assign_magnetic_field`
            - ``'density'``, ``'pressure'``, ``'velocity'``, ``'temperature'`` —
              requires :meth:`assign_atmosphere`

        altitude_slice : float, optional
            Altitude (km) for a 2-D horizontal map. When *None*, plots a
            vertical profile at the source location (or the 3-D geometry for
            ``'grid_points'``).
        **kwargs
            Forwarded to the underlying matplotlib calls.

        Raises
        ------
        AttributeError
            If the grid has not been created.
        ValueError
            If *variable* is not recognised.
        """
        if not hasattr(self, "grid"):
            raise AttributeError("3D grid not created. Call make_3Dgrid() first.")

        if variable == "grid_points":
            if altitude_slice is None:
                self.plot_grid_3d()
            else:
                self.plot_grid()

        elif variable == "electron_density":
            self._plot_electron_density(altitude_slice, **kwargs)

        elif variable in ["Be", "Bn", "Bu", "Br", "Btheta", "Bphi", "inclination", "declination"]:
            self._plot_magnetic_field_variable(variable, altitude_slice, **kwargs)

        elif variable in ["density", "pressure", "velocity", "temperature"]:
            self._plot_atmospheric_variable(variable, altitude_slice, **kwargs)

        else:
            available_vars = [
                "grid_points", "electron_density",
                "Be", "Bn", "Bu", "inclination", "declination",
                "density", "pressure", "velocity", "temperature",
            ]
            raise ValueError(
                f"Unknown variable '{variable}'. Available variables: {available_vars}"
            )

    # ------------------------------------------------------------------
    # Private plot helpers
    # ------------------------------------------------------------------

    def _plot_electron_density(self, altitude_slice=None, **kwargs):
        """Plot electron density as a map or vertical profile."""
        if "electron_density" not in self.grid.data_vars:
            raise AttributeError(
                "Electron density not computed. Call assign_ionosphere() first."
            )
        if altitude_slice is not None:
            self._plot_2d_map(
                "electron_density", altitude_slice,
                title_prefix="Electron Density", units="m⁻³", log_scale=True, **kwargs
            )
        else:
            self._plot_vertical_profile(
                "electron_density",
                title_prefix="Electron Density", units="m⁻³", log_scale=True, **kwargs
            )

    def _plot_atmospheric_variable(self, variable, altitude_slice=None, **kwargs):
        """Plot a 1-D atmospheric profile at the source location."""
        if not hasattr(self, "atmosphere"):
            raise AttributeError(
                "Atmospheric data not computed. Call assign_atmosphere() first."
            )

        if altitude_slice is not None:
            import logging
            logging.getLogger(__name__).warning(
                "altitude_slice not supported for atmospheric variables; "
                "plotting 1-D profile at source location instead."
            )

        fig, ax = plt.subplots(1, 1, figsize=(8, 10))
        atmos_data = self.atmosphere.atmosphere[variable]
        altitudes = self.atmosphere.alt_km

        if variable in ["density", "pressure"]:
            ax.semilogx(atmos_data, altitudes, **kwargs)
        else:
            ax.plot(atmos_data, altitudes, **kwargs)

        var_labels = {
            "density": "Density (kg m⁻³)",
            "pressure": "Pressure (Pa)",
            "velocity": "Velocity (km s⁻¹)",
            "temperature": "Temperature (K)",
        }

        ax.set_xlabel(var_labels.get(variable, variable))
        ax.set_ylabel("Altitude (km)")
        ax.set_title(
            f"Atmospheric {variable.title()} Profile\n"
            f"Lat: {self.source.get_latitude():.2f}°, "
            f"Lon: {self.source.get_longitude():.2f}°"
        )
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def _plot_magnetic_field_variable(self, variable, altitude_slice=None, **kwargs):
        """Plot a magnetic field variable as a map or vertical profile."""
        if variable not in self.grid.data_vars:
            raise AttributeError(
                f"Magnetic field variable '{variable}' not computed. "
                "Call assign_magnetic_field() first."
            )

        var_properties = {
            "Be":          {"title": "East Magnetic Field Component",      "units": "nT",      "log_scale": False},
            "Bn":          {"title": "North Magnetic Field Component",     "units": "nT",      "log_scale": False},
            "Bu":          {"title": "Up Magnetic Field Component",        "units": "nT",      "log_scale": False},
            "Br":          {"title": "Radial Magnetic Field Component",    "units": "nT",      "log_scale": False},
            "Btheta":      {"title": "Colatitude Magnetic Field Component","units": "nT",      "log_scale": False},
            "Bphi":        {"title": "Azimuth Magnetic Field Component",   "units": "nT",      "log_scale": False},
            "inclination": {"title": "Magnetic Inclination",              "units": "degrees", "log_scale": False},
            "declination": {"title": "Magnetic Declination",              "units": "degrees", "log_scale": False},
        }

        props = var_properties.get(
            variable,
            {"title": variable.title(), "units": "", "log_scale": False},
        )

        if altitude_slice is not None:
            self._plot_2d_map(
                variable, altitude_slice,
                title_prefix=props["title"], units=props["units"],
                log_scale=props["log_scale"], **kwargs
            )
        else:
            self._plot_vertical_profile(
                variable,
                title_prefix=props["title"], units=props["units"],
                log_scale=props["log_scale"], **kwargs
            )

    def _plot_2d_map(self, variable, altitude, title_prefix, units, log_scale=False, **kwargs):
        """Plot a 2-D horizontal map of *variable* at the given *altitude*.

        Parameters
        ----------
        variable : str
            Name of the ``self.grid`` data variable to plot.
        altitude : float
            Target altitude in km (nearest level is selected).
        title_prefix : str
            Label prepended to the plot title.
        units : str
            Physical units shown in the colorbar label.
        log_scale : bool, optional
            Use log-normalised colormap. Default False.
        **kwargs
            Forwarded to ``ax.contourf``.
        """
        altitudes = self.grid.coords["altitude"].values
        alt_idx = int(np.argmin(np.abs(altitudes - altitude)))
        actual_altitude = float(altitudes[alt_idx])

        data_2d = self.grid[variable].isel(altitude=alt_idx)
        latitudes = self.grid.coords["latitude"].values
        longitudes = self.grid.coords["longitude"].values

        fig, ax = plt.subplots(
            1, 1, figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()}
        )
        ax.set_extent(
            [longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()],
            crs=ccrs.PlateCarree(),
        )
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.LAND, alpha=0.3)
        ax.add_feature(cfeature.OCEAN, alpha=0.3)

        lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
        norm = matplotlib.colors.LogNorm() if log_scale else None

        im = ax.contourf(
            lon_grid, lat_grid, data_2d.values,
            levels=20, transform=ccrs.PlateCarree(), norm=norm, **kwargs
        )

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(f"{title_prefix} ({units})")

        if hasattr(self, "source"):
            ax.plot(
                self.source.get_longitude(), self.source.get_latitude(),
                "k*", markersize=15, transform=ccrs.PlateCarree(), label="Source"
            )
            ax.legend()

        ax.gridlines(draw_labels=True, alpha=0.5)
        plt.title(
            f"{title_prefix} at {actual_altitude:.1f} km\n"
            f"Time: {self.source.get_time() if hasattr(self, 'source') else 'N/A'}"
        )
        plt.tight_layout()
        plt.show()

    def _plot_vertical_profile(self, variable, title_prefix, units, log_scale=False, **kwargs):
        """Plot a vertical profile of *variable* at the source grid point.

        Parameters
        ----------
        variable : str
            Name of the ``self.grid`` data variable to plot.
        title_prefix : str
            Label used for the x-axis and title.
        units : str
            Physical units shown on the x-axis.
        log_scale : bool, optional
            Use a logarithmic x-axis. Default False.
        **kwargs
            Forwarded to ``ax.plot`` / ``ax.semilogx``.
        """
        if not hasattr(self, "source"):
            raise AttributeError("Source not assigned to the model.")

        latitudes = self.grid.coords["latitude"].values
        longitudes = self.grid.coords["longitude"].values
        lat_idx = int(np.argmin(np.abs(latitudes - self.source.get_latitude())))
        lon_idx = int(np.argmin(np.abs(longitudes - self.source.get_longitude())))

        profile = self.grid[variable].isel(latitude=lat_idx, longitude=lon_idx)
        altitudes = self.grid.coords["altitude"].values

        fig, ax = plt.subplots(1, 1, figsize=(8, 10))
        if log_scale:
            ax.semilogx(profile.values, altitudes, **kwargs)
        else:
            ax.plot(profile.values, altitudes, **kwargs)

        ax.set_xlabel(f"{title_prefix} ({units})")
        ax.set_ylabel("Altitude (km)")
        ax.set_title(
            f"{title_prefix} Vertical Profile\n"
            f"Lat: {self.source.get_latitude():.2f}°, "
            f"Lon: {self.source.get_longitude():.2f}°, "
            f"Time: {self.source.get_time()}"
        )
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

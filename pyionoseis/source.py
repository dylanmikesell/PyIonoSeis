from datetime import datetime
import logging

import toml

_log = logging.getLogger(__name__)


_DEFAULT_EVENT_TIME = "2023-01-01T00:00:00Z"
_EVENT_TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

class EarthquakeSource:
    """Earthquake source container loaded from TOML event metadata.

    Parameters
    ----------
    toml_file : str or path-like, optional
        Path to a TOML file containing an ``[event]`` section.

    Attributes
    ----------
    time : datetime
        Event origin time in UTC.
    latitude : float
        Epicenter latitude in degrees.
    longitude : float
        Epicenter longitude in degrees.
    depth : float
        Hypocentral depth in km.
    """

    def __init__(self, toml_file=None) -> None:
        if toml_file:
            self.load_from_toml(toml_file)
        else:
            self.time = datetime.strptime(_DEFAULT_EVENT_TIME, _EVENT_TIME_FORMAT)
            self.latitude = 0.0
            self.longitude = 0.0
            self.depth = 0.0

    def load_from_toml(self, toml_file) -> None:
        """Load event metadata from a TOML file.

        Parameters
        ----------
        toml_file : str or path-like
            Path to TOML input with an ``[event]`` table.

        Raises
        ------
        ValueError
            If event time does not follow ``%Y-%m-%dT%H:%M:%SZ``.
        """
        data = toml.load(toml_file)
        event = data.get("event", {})
        self.time = datetime.strptime(
            event.get("time", _DEFAULT_EVENT_TIME), _EVENT_TIME_FORMAT
        )
        self.latitude = float(event.get("latitude", 0.0))
        self.longitude = float(event.get("longitude", 0.0))
        self.depth = float(event.get("depth", 0.0))

    def __str__(self) -> str:
        """Return a concise representation of the earthquake source."""
        return (f"EarthquakeSource(time={self.time}, latitude={self.latitude}, "
                f"longitude={self.longitude}, depth={self.depth})")

    def print_info(self) -> None:
        """Log the object representation at INFO level."""
        _log.info("%s", self.__str__())

    def get_latitude(self) -> float:
        """Return epicenter latitude in degrees."""
        return self.latitude

    def get_longitude(self) -> float:
        """Return epicenter longitude in degrees."""
        return self.longitude

    def get_time(self) -> datetime:
        """Return origin time as ``datetime``."""
        return self.time

    def get_depth(self) -> float:
        """Return hypocentral depth in km."""
        return self.depth
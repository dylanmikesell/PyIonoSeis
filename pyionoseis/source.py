from datetime import datetime
import logging

import toml

_log = logging.getLogger(__name__)


_DEFAULT_EVENT_TIME = "2023-01-01T00:00:00Z"
_EVENT_TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_EVENT_TIME_FORMAT_FRACTIONAL = "%Y-%m-%dT%H:%M:%S.%fZ"


def _parse_event_time(time_string: str) -> datetime:
    """Parse event time with optional fractional-second UTC precision.

    Parameters
    ----------
    time_string : str
        UTC timestamp in ``YYYY-MM-DDTHH:MM:SSZ`` or
        ``YYYY-MM-DDTHH:MM:SS.ffffffZ`` format.

    Returns
    -------
    datetime
        Parsed event time.

    Raises
    ------
    ValueError
        If ``time_string`` does not match supported UTC timestamp formats.
    """
    for event_time_format in (_EVENT_TIME_FORMAT, _EVENT_TIME_FORMAT_FRACTIONAL):
        try:
            return datetime.strptime(time_string, event_time_format)
        except ValueError:
            continue
    raise ValueError(
        "Event time must be in UTC format YYYY-MM-DDTHH:MM:SSZ or "
        "YYYY-MM-DDTHH:MM:SS.ffffffZ"
    )

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
            self.time = _parse_event_time(_DEFAULT_EVENT_TIME)
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
            If event time does not follow supported UTC timestamp formats.
        """
        data = toml.load(toml_file)
        event = data.get("event", {})
        self.time = _parse_event_time(event.get("time", _DEFAULT_EVENT_TIME))
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
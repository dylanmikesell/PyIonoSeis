from typing import Any
import toml
from datetime import datetime

class EarthquakeSource:
    def __init__(self, toml_file=None) -> None:
        """
        Initialize the object with optional TOML configuration file.

        Args:
            toml_file (str, optional): Path to the TOML file to load configuration from. 
                                       If not provided, default values will be used.

        Attributes:
            time (datetime): The time attribute initialized to "2023-01-01T00:00:00Z" if no TOML file is provided.
            latitude (float): The latitude attribute initialized to 0.0 if no TOML file is provided.
            longitude (float): The longitude attribute initialized to 0.0 if no TOML file is provided.
            depth (float): The depth attribute initialized to 0.0 if no TOML file is provided.
        """
        if toml_file:
            self.load_from_toml(toml_file)
        else:
            self.time = datetime.strptime("2023-01-01T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ")
            self.latitude = 0.0
            self.longitude = 0.0
            self.depth = 0.0

    def load_from_toml(self, toml_file) -> None:
        """
        Load event data from a TOML file and update the instance attributes.

        Args:
            toml_file (str or file-like object): Path to the TOML file or a file-like object containing the event data.

        Returns:
            None

        Raises:
            ValueError: If the 'time' field in the TOML file is not in the correct format.

        The expected structure of the TOML file is:
        [event]
        time = "YYYY-MM-DDTHH:MM:SSZ"
        latitude = float
        longitude = float
        depth = float

        Example:
            [event]
            time = "2023-01-01T00:00:00Z"
            latitude = 34.05
            longitude = -118.25
            depth = 10.0
        """
        data = toml.load(toml_file)
        event = data.get('event', {})
        self.time = datetime.strptime(event.get('time', "2023-01-01T00:00:00Z"), "%Y-%m-%dT%H:%M:%SZ")
        self.latitude = event.get('latitude', 0.0)
        self.longitude = event.get('longitude', 0.0)
        self.depth = event.get('depth', 0.0)

    def __str__(self):
        """
        Returns a string representation of the EarthquakeSource object.

        The string includes the time, latitude, longitude, and depth of the earthquake source.

        Returns:
            str: A string representation of the EarthquakeSource object.
        """
        return (f"EarthquakeSource(time={self.time}, latitude={self.latitude}, "
                f"longitude={self.longitude}, depth={self.depth})")

    def print_info(self) -> None:
        """
        Prints the string representation of the object.

        This method calls the `__str__` method of the object and prints its result
        to the standard output.

        Returns:
            None
        """
        print(self.__str__())

    def get_latitude(self) -> float | Any:
        """
        Returns the latitude value.

        Returns:
            float: The latitude value.
        """
        return self.latitude

    def get_longitude(self) -> float | Any:
        """
        Retrieves the longitude value.

        Returns:
            float: The longitude value.
        """
        return self.longitude

    def get_time(self) -> datetime:
        """
        Retrieve the time attribute.

        Returns:
            datetime: The time attribute of the instance.
        """
        return self.time
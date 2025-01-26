import toml

class EarthquakeSource:
    def __init__(self, toml_file=None):
        if toml_file:
            self.load_from_toml(toml_file)
        else:
            self.time = "2023-01-01T00:00:00Z"
            self.latitude = 0.0
            self.longitude = 0.0
            self.depth = 0.0

    def load_from_toml(self, toml_file):
        data = toml.load(toml_file)
        event = data.get('event', {})
        self.time = event.get('time', "2023-01-01T00:00:00Z")
        self.latitude = event.get('latitude', 0.0)
        self.longitude = event.get('longitude', 0.0)
        self.depth = event.get('depth', 0.0)

    def __str__(self):
        return (f"EarthquakeSource(time={self.time}, latitude={self.latitude}, "
                f"longitude={self.longitude}, depth={self.depth})")

    def print_info(self):
        print(self.__str__())

    def get_latitude(self):
        return self.latitude

    def get_longitude(self):
        return self.longitude

    
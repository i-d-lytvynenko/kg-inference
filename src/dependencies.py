import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class WorkDir:
    """
    Class to handle working directory operations.

    Example:

        >>> wd = WorkDir.create_temporary_workdir()
        >>> wd.check_file_exists("test.txt")
        False
        >>> wd.write_file("test.txt", "Hello, world!")
        >>> wd.check_file_exists("test.txt")
        True
        >>> wd.read_file("test.txt")
        'Hello, world!'
    """

    location: str = field(default_factory=tempfile.gettempdir)

    # NOTE:
    # The user is responsible for deleting the temporary directory
    # and its contents when done with it.
    @classmethod
    def create_temporary_workdir(cls) -> "WorkDir":
        temp_dir = tempfile.mkdtemp()
        return cls(location=temp_dir)

    def _ensure_location(self):
        location = Path(self.location)
        location.mkdir(parents=True, exist_ok=True)

    def __post_init__(self):
        self._ensure_location()

    def get_file_path(self, file_name: str) -> Path:
        self._ensure_location()
        return Path(self.location) / file_name

    def read_file(self, file_path: str) -> str:
        self._ensure_location()
        file_path = str(self.get_file_path(file_path))
        with open(self.get_file_path(file_path), "r") as f:
            return f.read()

    def check_file_exists(self, file_path: str) -> bool:
        self._ensure_location()
        return self.get_file_path(file_path).exists()

    def write_file(self, file_path: str, content: str) -> None:
        self._ensure_location()
        with open(self.get_file_path(file_path), "w") as f:
            f.write(content)

    def delete_file(self, file_path: str) -> None:
        self._ensure_location()
        self.get_file_path(file_path).unlink()

    def list_file_names(self) -> list[str]:
        """
        List the names of all files in the working directory.

        Note: the working directory is not recursively searched, it is flat

        Returns:
            list of file names
        """
        self._ensure_location()
        return [f.name for f in Path(self.location).iterdir() if f.is_file()]


@dataclass
class HasWorkdir:
    workdir: WorkDir = field(default_factory=lambda: WorkDir())


@dataclass
class HasSchema:
    schema_path: Path


@dataclass
class HasData(HasSchema):
    data_path: Path

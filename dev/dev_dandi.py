from pathlib import Path
from dandi.organize import organize

paths = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\dandi")

dandiset = Path.home() / "repos" / "dandi" / "001546"
organize([str(paths)], str(dandiset))

# dandiset_path.is_dir()

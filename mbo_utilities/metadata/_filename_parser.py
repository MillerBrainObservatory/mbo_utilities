"""
filename metadata parser.

extracts experiment metadata from filenames and paths using pattern matching.
users can follow suggested naming conventions but deviations are handled gracefully.

naming convention (suggested):
    {animal}_{region}_{indicator}_{line}_{session}_{...}.tif

examples:
    mouse_V1_GCaMP6f_Cux2_session01.tif
    zf_OB_Cal520_20240115.tif
    rat_S1_jGCaMP8m_CaMKII_run3.tif
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# pattern definitions for each metadata field
# each pattern is a tuple of (regex, canonical_value or None for direct match)
PATTERNS: dict[str, list[tuple[str, str | None]]] = {
    "calcium_indicator": [
        # GCaMP variants
        (r"\b(GCaMP\d[a-z]?)\b", None),
        (r"\b(jGCaMP\d[a-z]?)\b", None),
        (r"\b(gcamp\d[a-z]?)\b", None),
        # Cal-520 variants
        (r"\bCal[-_]?520\b", "Cal-520"),
        (r"\bcal[-_]?520\b", "Cal-520"),
        # OGB-1
        (r"\bOGB[-_]?1\b", "OGB-1"),
        (r"\bogb[-_]?1\b", "OGB-1"),
        # others
        (r"\b(Fluo-4)\b", None),
        (r"\b(RCaMP\d?[a-z]?)\b", None),
    ],
    "animal_model": [
        (r"\bmouse\b", "Mouse"),
        (r"\bMouse\b", "Mouse"),
        (r"\brat\b", "Rat"),
        (r"\bRat\b", "Rat"),
        (r"\bzebrafish\b", "Zebrafish"),
        (r"\bZebrafish\b", "Zebrafish"),
        (r"\bzf\b", "Zebrafish"),
        (r"\bZF\b", "Zebrafish"),
    ],
    "brain_region": [
        # cortical areas
        (r"\bV1\b", "V1"),
        (r"\bS1\b", "S1"),
        (r"\bM1\b", "M1"),
        (r"\bA1\b", "A1"),
        # other regions
        (r"\bOB\b", "OB"),
        (r"\bPBN\b", "PBN"),
        (r"\bNTS\b", "NTS"),
        (r"\bACC\b", "ACC"),
        (r"\bPFC\b", "PFC"),
        (r"\bHPC\b", "HPC"),
        (r"\bhippocampus\b", "HPC"),
        (r"\bHippocampus\b", "HPC"),
    ],
    "transgenic_line": [
        (r"\bCux2\b", "Cux2"),
        (r"\bcux2\b", "Cux2"),
        (r"\b[Ee]mx1\b", "Emx1"),
        (r"\btetO[sS]?\b", "tetO"),
        (r"\bNeuroD\b", "NeuroD"),
        (r"\bCaMKII[αa]?\b", "CaMKII"),
        (r"\bcamkii\b", "CaMKII"),
        (r"\bThy1\b", "Thy1"),
        (r"\bthy1\b", "Thy1"),
        (r"\bPV\b", "PV"),
        (r"\bSST\b", "SST"),
        (r"\bVIP\b", "VIP"),
    ],
    "induction_method": [
        (r"\bAAV\b", "AAV"),
        (r"\baav\b", "AAV"),
        (r"\btransgenic\b", "Transgenic"),
        (r"\bTransgenic\b", "Transgenic"),
        (r"\bTg\b", "Transgenic"),
        (r"\bacute\b", "Acute injection"),
        (r"\bAcute\b", "Acute injection"),
    ],
}


@dataclass
class FilenameMetadata:
    """
    metadata extracted from filename.

    attributes
    ----------
    calcium_indicator : str | None
        detected calcium indicator (e.g., "GCaMP6f", "Cal-520")
    animal_model : str | None
        detected animal model (e.g., "Mouse", "Zebrafish")
    brain_region : str | None
        detected brain region (e.g., "V1", "OB")
    transgenic_line : str | None
        detected transgenic line (e.g., "Cux2", "CaMKII")
    induction_method : str | None
        detected induction method (e.g., "AAV", "Transgenic")
    source_filename : str
        the filename that was parsed
    """

    calcium_indicator: str | None = None
    animal_model: str | None = None
    brain_region: str | None = None
    transgenic_line: str | None = None
    induction_method: str | None = None
    source_filename: str = ""

    def to_dict(self) -> dict[str, Any]:
        """convert to dict, excluding None values."""
        result = {}
        if self.calcium_indicator:
            result["calcium_indicator"] = self.calcium_indicator
        if self.animal_model:
            result["animal_model"] = self.animal_model
        if self.brain_region:
            result["brain_region"] = self.brain_region
        if self.transgenic_line:
            result["transgenic_line"] = self.transgenic_line
        if self.induction_method:
            result["induction_method"] = self.induction_method
        return result

    def __bool__(self) -> bool:
        """true if any metadata was detected."""
        return bool(self.to_dict())

    def __len__(self) -> int:
        """number of detected fields."""
        return len(self.to_dict())


def parse_filename_metadata(filename: str | Path) -> FilenameMetadata:
    """
    extract metadata from filename using pattern matching.

    parameters
    ----------
    filename : str | Path
        filename or full path to parse

    returns
    -------
    FilenameMetadata
        detected metadata fields

    examples
    --------
    >>> meta = parse_filename_metadata("mouse_V1_GCaMP6f_Cux2.tif")
    >>> meta.animal_model
    'Mouse'
    >>> meta.brain_region
    'V1'
    >>> meta.calcium_indicator
    'GCaMP6f'
    """
    # get just the filename stem (no extension, no path)
    if isinstance(filename, Path):
        name = filename.stem
    else:
        name = Path(filename).stem

    result = FilenameMetadata(source_filename=str(filename))

    for field_name, patterns in PATTERNS.items():
        for pattern, canonical in patterns:
            match = re.search(pattern, name, re.IGNORECASE if canonical else 0)
            if match:
                # use canonical value if provided, else use the match
                value = canonical if canonical else match.group(1) if match.lastindex else match.group(0)
                setattr(result, field_name, value)
                break  # first match wins for each field

    return result


def get_filename_suggestions() -> dict[str, dict]:
    """
    get suggested metadata fields with their descriptions.

    returns dict of field definitions suitable for the metadata popup.
    """
    return {
        "calcium_indicator": {
            "canonical": "calcium_indicator",
            "label": "Indicator",
            "dtype": str,
            "description": "Calcium indicator (e.g., GCaMP6f, Cal-520, jGCaMP8m)",
            "examples": ["GCaMP6f", "GCaMP6s", "jGCaMP7f", "jGCaMP8m", "Cal-520", "OGB-1"],
        },
        "animal_model": {
            "canonical": "animal_model",
            "label": "Animal",
            "dtype": str,
            "description": "Animal model (e.g., Mouse, Zebrafish, Rat)",
            "examples": ["Mouse", "Zebrafish", "Rat"],
        },
        "brain_region": {
            "canonical": "brain_region",
            "label": "Region",
            "dtype": str,
            "description": "Brain region (e.g., V1, S1, OB, HPC)",
            "examples": ["V1", "S1", "M1", "OB", "PBN", "NTS", "HPC"],
        },
        "transgenic_line": {
            "canonical": "transgenic_line",
            "label": "Line",
            "dtype": str,
            "description": "Transgenic line (e.g., Cux2, Emx1, CaMKII)",
            "examples": ["Cux2", "Emx1", "tetO", "CaMKII", "Thy1", "PV"],
        },
        "induction_method": {
            "canonical": "induction_method",
            "label": "Induction",
            "dtype": str,
            "description": "Expression method (e.g., AAV, Transgenic, Acute injection)",
            "examples": ["AAV", "Transgenic", "Acute injection"],
        },
    }

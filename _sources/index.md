# Miller Brain Observatory: Utilities

Image processing utilities for the [Miller Brain Observatory](https://github.com/MillerBrainObservatory) (MBO).

## Contents

```{toctree}
---
maxdepth: 1
---
User Guide <user_guide>
Command Line Interface <usage/cli>
Graphical User Interface <usage/gui_guide>
Supported File Formats <file_formats>
DF/F Analysis <dff>
API <api/index>
development
glossary
```

## Quick Start

```bash
# install
uv pip install mbo_utilities

# launch gui
mbo

# convert data
mbo convert input.tiff output.zarr

# analyze scan-phase
mbo scanphase /path/to/data
```

## Resources

- [GUI User Guide (PDF)](_static/mbo_gui_user_guide.pdf)
- [Virtual Environments Guide](https://millerbrainobservatory.github.io/guides/venvs.html)
- [uv cheatsheet](https://www.saaspegasus.com/guides/uv-deep-dive/#cheatsheet-common-operations-in-uvs-workflows)

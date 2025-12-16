# Miller Brain Observatory: Python Utilities

Python tools for pre/post processing datasets at the [Miller Brain Observatory](https://mbo.rockefeller.edu).

## Contents

```{toctree}
---
maxdepth: 1
---
User Guide <user_guide>
CLI & GUI <usage/index>
Array Types <array_types>
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

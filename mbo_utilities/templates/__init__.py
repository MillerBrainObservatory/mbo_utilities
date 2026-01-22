"""
notebook templates for mbo_utilities.

provides templates for common analysis pipelines that can be generated
via the CLI with `mbo notebook <template>`.
"""

from .notebooks import (
    TEMPLATES,
    create_notebook,
    list_templates,
    get_template_path,
)

__all__ = [
    "TEMPLATES",
    "create_notebook",
    "list_templates",
    "get_template_path",
]

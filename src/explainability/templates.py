"""
Metapath template registry and validation helpers.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

EdgeType = Tuple[str, str, str]

DEFAULT_METAPATH_TEMPLATES: Dict[str, List[EdgeType]] = {
    "chem-gene-disease": [
        ("chemical", "affects", "gene"),
        ("gene", "rev_targets", "disease"),
    ],
    "chem-pathway-disease": [
        ("chemical", "enriched_in", "pathway"),
        ("pathway", "rev_disrupts", "disease"),
    ],
    "chem-goterm-disease": [
        ("chemical", "enriched_in", "go_term"),
        ("go_term", "associated_with", "disease"),
    ],
    "chem-phenoGO-disease": [
        ("chemical", "affects_phenotype", "go_term"),
        ("go_term", "associated_with", "disease"),
    ],
    "chem-gene-gene-disease": [
        ("chemical", "affects", "gene"),
        ("gene", "interacts_with", "gene"),
        ("gene", "rev_targets", "disease"),
    ],
    "chem-gene-pathway-disease": [
        ("chemical", "affects", "gene"),
        ("gene", "participates_in", "pathway"),
        ("pathway", "rev_disrupts", "disease"),
    ],
    "chem-pathway-gene-disease": [
        ("chemical", "enriched_in", "pathway"),
        ("pathway", "rev_participates_in", "gene"),
        ("gene", "rev_targets", "disease"),
    ],
}


def validate_templates(templates: Dict[str, List[EdgeType]]) -> None:
    """Validate metapath template definitions with explicit errors."""
    if not templates:
        raise ValueError("Template set is empty.")
    for name, template in templates.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Invalid template name: {name!r}")
        if not isinstance(template, list) or not template:
            raise ValueError(f'Template "{name}" must be a non-empty list of edge tuples.')
        if len(template) not in {2, 3}:
            raise ValueError(f'Template "{name}" must have 2 or 3 hops, got {len(template)}.')
        for et in template:
            if not (isinstance(et, tuple) and len(et) == 3 and all(isinstance(x, str) for x in et)):
                raise ValueError(f'Template "{name}" contains invalid edge type: {et!r}')
        if template[0][0] != "chemical":
            raise ValueError(f'Template "{name}" must start from "chemical".')
        if template[-1][2] != "disease":
            raise ValueError(f'Template "{name}" must end at "disease".')


def required_edge_types(templates: Dict[str, List[EdgeType]]) -> List[EdgeType]:
    """Return unique edge types required by a template set."""
    needed: Set[EdgeType] = set()
    for template in templates.values():
        needed.update(template)
    return list(needed)


def resolve_template_set(
    template_set: str = "default",
    custom_templates: Dict[str, List[EdgeType]] | None = None,
) -> Dict[str, List[EdgeType]]:
    """Resolve active templates and validate before use."""
    if template_set == "default":
        templates = custom_templates if custom_templates is not None else DEFAULT_METAPATH_TEMPLATES
        validate_templates(templates)
        return templates
    raise ValueError(
        f'Unknown template_set "{template_set}". Supported values: ["default"].'
    )

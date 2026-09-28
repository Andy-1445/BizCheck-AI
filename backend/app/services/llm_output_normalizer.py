from __future__ import annotations

import copy
from typing import Literal


LLMAnalysisTask = Literal[
    "company_analysis",
    "company_comparison_analysis",
]

_EVIDENCE_COLLECTIONS: dict[LLMAnalysisTask, tuple[tuple[str, str], ...]] = {
    "company_analysis": (
        ("findings", "evidence_paths"),
        ("verification_items", "related_evidence_paths"),
        ("limitations", "related_evidence_paths"),
    ),
    "company_comparison_analysis": (
        ("company_observations", "evidence_paths"),
        ("comparison_observations", "evidence_paths"),
        ("verification_items", "related_evidence_paths"),
        ("limitations", "related_evidence_paths"),
    ),
}


def normalize_evidence_manifest(
    output: dict[str, object],
    *,
    task: LLMAnalysisTask,
) -> dict[str, object]:
    """Derive the redundant provenance manifest without mutating raw output.

    Evidence paths remain model-selected claims and still pass the canonical
    path, leaf, topic, numeric, and safety validators. Only the duplicate
    ``provenance.evidence_paths_used`` index is rebuilt by trusted code.
    Malformed collections are left untouched so Pydantic reports them normally.
    """

    normalized = copy.deepcopy(output)
    provenance = normalized.get("provenance")
    if not isinstance(provenance, dict):
        return normalized

    referenced_paths: list[str] = []
    for collection_name, path_field in _EVIDENCE_COLLECTIONS[task]:
        collection = normalized.get(collection_name)
        if not isinstance(collection, list):
            return normalized
        for item in collection:
            if not isinstance(item, dict):
                return normalized
            paths = item.get(path_field)
            if not isinstance(paths, list) or not all(
                isinstance(path, str) for path in paths
            ):
                return normalized
            referenced_paths.extend(paths)

    provided_manifest = provenance.get("evidence_paths_used")
    if not isinstance(provided_manifest, list) or not all(
        isinstance(path, str) for path in provided_manifest
    ):
        return normalized
    if len(provided_manifest) != len(set(provided_manifest)):
        return normalized

    derived_manifest = list(dict.fromkeys(referenced_paths))
    if not set(provided_manifest).issubset(derived_manifest):
        # Extra provider-selected paths remain a validation failure. They must
        # not silently expand the evidence available to numeric grounding.
        return normalized

    provenance["evidence_paths_used"] = derived_manifest
    return normalized


def build_server_owned_manifest_schema(
    schema: dict[str, object],
) -> dict[str, object]:
    """Return a schema copy that requires the provider manifest to be empty."""

    normalized = copy.deepcopy(schema)

    def visit(value: object) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            return

        properties = value.get("properties")
        if isinstance(properties, dict) and "evidence_paths_used" in properties:
            properties["evidence_paths_used"] = {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 0,
            }
        for item in value.values():
            visit(item)

    visit(normalized)
    return normalized

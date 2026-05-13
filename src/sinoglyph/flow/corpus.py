from dataclasses import dataclass

from sinoglyph.io import PathLike, save_json
from sinoglyph.schema.base import JsonArray, JsonObject
from sinoglyph.schema.character import CharacterPerturbCatalog
from sinoglyph.schema.corpus import (
    CorpusEntry,
    PerturbedCorpusEntry,
    SubstitutionRecord,
    load_annotated_corpus,
    parse_perturbed_corpus,
)


@dataclass(frozen=True)
class CorpusSubstitution:
    index: int
    record: SubstitutionRecord


class PerturbedCorpusBuilder:
    def __init__(
        self,
        annotated: list[CorpusEntry],
        catalog: CharacterPerturbCatalog,
    ) -> None:
        self._annotated = annotated
        self._catalog = catalog

    def build(self) -> JsonArray:
        result: JsonArray = [
            self._build_entry(entry).to_mapping() for entry in self._annotated
        ]
        parse_perturbed_corpus(result)
        return result

    def _build_entry(self, entry: CorpusEntry) -> PerturbedCorpusEntry:
        anchor_positions = _find_anchor_positions(entry.text, entry.semantic_anchors)
        substitutions = self._find_substitutions(entry.text)
        anchor_substitutions = [
            substitution
            for substitution in substitutions
            if substitution.index in anchor_positions
        ]
        non_anchor_substitutions = [
            substitution
            for substitution in substitutions
            if substitution.index not in anchor_positions
        ]
        denominator = len(entry.text)
        anchor_fraction = _fraction(len(anchor_substitutions), denominator)
        non_anchor_fraction = _fraction(len(non_anchor_substitutions), denominator)
        return PerturbedCorpusEntry.from_mapping(
            {
                "id": entry.id,
                "text": entry.text,
                "expected_label": entry.expected_label.value,
                "semantic_anchors": [dict(anchor) for anchor in entry.semantic_anchors],
                "substitutions": {
                    "anchor": _unique_substitution_records(anchor_substitutions),
                    "non_anchor": _unique_substitution_records(
                        non_anchor_substitutions
                    ),
                    "fraction": {
                        "original": 0,
                        "anchor_only": anchor_fraction,
                        "non_anchor_only": non_anchor_fraction,
                        "full": _fraction(len(substitutions), denominator),
                    },
                },
                "decomposition": {
                    "original": entry.text,
                    "anchor_only": _render_variant(
                        entry.text,
                        anchor_substitutions,
                        "decomposition",
                    ),
                    "non_anchor_only": _render_variant(
                        entry.text,
                        non_anchor_substitutions,
                        "decomposition",
                    ),
                    "full": _render_variant(entry.text, substitutions, "decomposition"),
                },
                "perturbation": {
                    "original": entry.text,
                    "anchor_only": _render_variant(
                        entry.text,
                        anchor_substitutions,
                        "perturbation",
                    ),
                    "non_anchor_only": _render_variant(
                        entry.text,
                        non_anchor_substitutions,
                        "perturbation",
                    ),
                    "full": _render_variant(entry.text, substitutions, "perturbation"),
                },
            }
        )

    def _find_substitutions(self, text: str) -> list[CorpusSubstitution]:
        return [
            CorpusSubstitution(
                index=index,
                record=SubstitutionRecord(
                    character=character,
                    decomposition="".join(self._catalog.parts(character)),
                    perturbation="".join(perturbations[0]),
                ),
            )
            for index, character in enumerate(text)
            if character in self._catalog
            if (perturbations := self._catalog.perturbations(character))
        ]


def build_perturbed_corpus(
    annotated_path: PathLike,
    catalog_path: PathLike,
    output_path: PathLike | None = None,
) -> JsonArray:
    annotated = load_annotated_corpus(annotated_path)
    catalog = CharacterPerturbCatalog.from_json(catalog_path)
    result = PerturbedCorpusBuilder(annotated, catalog).build()

    if output_path is not None:
        save_json(result, output_path)
    return result


def _find_anchor_positions(text: str, anchors: list[JsonObject]) -> set[int]:
    positions: set[int] = set()
    for anchor in anchors:
        anchor_text = str(anchor["text"])
        if not anchor_text:
            continue
        start = 0
        while True:
            index = text.find(anchor_text, start)
            if index < 0:
                break
            positions.update(range(index, index + len(anchor_text)))
            start = index + 1
    return positions


def _render_variant(
    text: str,
    substitutions: list[CorpusSubstitution],
    field: str,
) -> str:
    replacement_by_index = {
        substitution.index: getattr(substitution.record, field)
        for substitution in substitutions
    }
    return "".join(
        replacement_by_index.get(index, character)
        for index, character in enumerate(text)
    )


def _unique_substitution_records(
    substitutions: list[CorpusSubstitution],
) -> list[JsonObject]:
    records_by_key = {
        _substitution_record_key(substitution): substitution.record.to_mapping()
        for substitution in substitutions
    }
    return list(records_by_key.values())


def _substitution_record_key(
    substitution: CorpusSubstitution,
) -> tuple[str, str, str]:
    record = substitution.record
    return (record.character, record.decomposition, record.perturbation)


def _fraction(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 5)

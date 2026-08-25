from dataclasses import dataclass

from sinoglyph.io import PathLike, save_json
from sinoglyph.schema.base import JsonArray, JsonObject
from sinoglyph.schema.character import GlyphObfuscationCatalog
from sinoglyph.schema.corpus import (
    CorpusEntry,
    GlyphObfuscationRecord,
    ObfuscatedCorpusEntry,
    load_annotated_corpus,
    parse_obfuscated_corpus,
)


@dataclass(frozen=True)
class _PositionedGlyphObfuscation:
    index: int
    record: GlyphObfuscationRecord


class _ObfuscatedCorpusBuilder:
    def __init__(
        self,
        annotated: list[CorpusEntry],
        catalog: GlyphObfuscationCatalog,
    ) -> None:
        self._annotated = annotated
        self._catalog = catalog

    def build(self) -> JsonArray:
        result: JsonArray = [
            self._build_entry(entry).export_mapping() for entry in self._annotated
        ]
        parse_obfuscated_corpus(result)
        return result

    def _build_entry(self, entry: CorpusEntry) -> ObfuscatedCorpusEntry:
        anchor_positions = _find_anchor_positions(entry.text, entry.semantic_anchors)
        obfuscations = self._find_obfuscations(entry.text)
        anchor_obfuscations = [
            obfuscation
            for obfuscation in obfuscations
            if obfuscation.index in anchor_positions
        ]
        background_obfuscations = [
            obfuscation
            for obfuscation in obfuscations
            if obfuscation.index not in anchor_positions
        ]
        denominator = len(entry.text)
        anchor_density = _density(len(anchor_obfuscations), denominator)
        background_density = _density(len(background_obfuscations), denominator)
        return ObfuscatedCorpusEntry.parse_mapping(
            {
                "id": entry.id,
                "text": entry.text,
                "expected_label": entry.expected_label.value,
                "semantic_anchors": [dict(anchor) for anchor in entry.semantic_anchors],
                "obfuscations": {
                    "anchor": _unique_obfuscation_records(anchor_obfuscations),
                    "background": _unique_obfuscation_records(background_obfuscations),
                    "obf_density": {
                        "original": 0,
                        "anchor_only": anchor_density,
                        "background_only": background_density,
                        "full": _density(len(obfuscations), denominator),
                    },
                },
                "decomposition": {
                    "original": entry.text,
                    "anchor_only": _render_scope_text(
                        entry.text,
                        anchor_obfuscations,
                        "decomposition",
                    ),
                    "background_only": _render_scope_text(
                        entry.text,
                        background_obfuscations,
                        "decomposition",
                    ),
                    "full": _render_scope_text(
                        entry.text, obfuscations, "decomposition"
                    ),
                },
                "cross_script": {
                    "original": entry.text,
                    "anchor_only": _render_scope_text(
                        entry.text,
                        anchor_obfuscations,
                        "cross_script",
                    ),
                    "background_only": _render_scope_text(
                        entry.text,
                        background_obfuscations,
                        "cross_script",
                    ),
                    "full": _render_scope_text(
                        entry.text, obfuscations, "cross_script"
                    ),
                },
            }
        )

    def _find_obfuscations(self, text: str) -> list[_PositionedGlyphObfuscation]:
        return [
            _PositionedGlyphObfuscation(
                index=index,
                record=GlyphObfuscationRecord(
                    character=character,
                    decomposition="".join(self._catalog.parts(character)),
                    cross_script="".join(glyph_forms[0]),
                ),
            )
            for index, character in enumerate(text)
            if character in self._catalog
            if (glyph_forms := self._catalog.glyph_forms(character))
        ]


def generate_obfuscated_corpus(
    annotated_path: PathLike,
    catalog_path: PathLike,
    output_path: PathLike | None = None,
) -> JsonArray:
    annotated = load_annotated_corpus(annotated_path)
    catalog = GlyphObfuscationCatalog.load_json(catalog_path)
    result = _ObfuscatedCorpusBuilder(annotated, catalog).build()

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


def _render_scope_text(
    text: str,
    obfuscations: list[_PositionedGlyphObfuscation],
    field: str,
) -> str:
    replacement_by_index = {
        obfuscation.index: getattr(obfuscation.record, field)
        for obfuscation in obfuscations
    }
    return "".join(
        replacement_by_index.get(index, character)
        for index, character in enumerate(text)
    )


def _unique_obfuscation_records(
    obfuscations: list[_PositionedGlyphObfuscation],
) -> list[JsonObject]:
    records_by_key = {
        _obfuscation_record_key(obfuscation): obfuscation.record.export_mapping()
        for obfuscation in obfuscations
    }
    return list(records_by_key.values())


def _obfuscation_record_key(
    obfuscation: _PositionedGlyphObfuscation,
) -> tuple[str, str, str]:
    record = obfuscation.record
    return (record.character, record.decomposition, record.cross_script)


def _density(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 5)

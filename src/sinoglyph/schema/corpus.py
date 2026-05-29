from __future__ import annotations

from dataclasses import dataclass

from sinoglyph.io import PathLike, load_json, save_json
from sinoglyph.schema.base import JsonObject
from sinoglyph.schema.types import ModerationLabel, ObfuscationScope
from sinoglyph.schema.utils import (
    require_enum,
    require_keys,
    require_list,
    require_mapping,
    require_number,
    require_string,
)

SCOPES = tuple(item.value for item in ObfuscationScope)


@dataclass(frozen=True)
class GlyphObfuscationRecord:
    character: str
    decomposition: str
    cross_script: str

    @classmethod
    def parse_mapping(cls, mapping: object, context: str) -> "GlyphObfuscationRecord":
        raw = require_mapping(mapping, context)
        require_keys(raw, {"character", "decomposition", "cross_script"}, context)
        return cls(
            character=require_string(raw["character"], f"{context}.character"),
            decomposition=require_string(
                raw["decomposition"], f"{context}.decomposition"
            ),
            cross_script=require_string(
                raw["cross_script"],
                f"{context}.cross_script",
            ),
        )

    def export_mapping(self) -> JsonObject:
        return {
            "character": self.character,
            "decomposition": self.decomposition,
            "cross_script": self.cross_script,
        }


@dataclass(frozen=True)
class CorpusEntry:
    id: str
    text: str
    expected_label: ModerationLabel
    semantic_anchors: list[JsonObject]

    @classmethod
    def parse_mapping(
        cls, mapping: object, context: str = "corpus entry"
    ) -> "CorpusEntry":
        raw = require_mapping(mapping, context)
        require_keys(raw, {"id", "text", "semantic_anchors"}, context)
        label = _entry_label(raw, context)
        return cls(
            id=require_string(raw["id"], f"{context}.id"),
            text=require_string(raw["text"], f"{context}.text"),
            expected_label=require_enum(
                ModerationLabel,
                label[1],
                f"{context}.{label[0]}",
            ),
            semantic_anchors=_semantic_anchors(
                raw["semantic_anchors"],
                f"{context}.semantic_anchors",
            ),
        )

    def export_mapping(self) -> JsonObject:
        return {
            "id": self.id,
            "text": self.text,
            "expected_label": self.expected_label.value,
            "semantic_anchors": [dict(anchor) for anchor in self.semantic_anchors],
        }

    def to_source_mapping(self) -> JsonObject:
        return {
            "id": self.id,
            "text": self.text,
            "label": self.expected_label.value,
            "semantic_anchors": [dict(anchor) for anchor in self.semantic_anchors],
        }


@dataclass(frozen=True)
class ObfuscatedCorpusEntry(CorpusEntry):
    obfuscations: JsonObject
    decomposition: dict[ObfuscationScope, str]
    cross_script: dict[ObfuscationScope, str]

    @classmethod
    def parse_mapping(
        cls,
        mapping: object,
        context: str = "obfuscated corpus entry",
    ) -> "ObfuscatedCorpusEntry":
        raw = require_mapping(mapping, context)
        base = CorpusEntry.parse_mapping(raw, context)
        require_keys(
            raw,
            {"obfuscations", "decomposition", "cross_script"},
            context,
        )
        return cls(
            id=base.id,
            text=base.text,
            expected_label=base.expected_label,
            semantic_anchors=base.semantic_anchors,
            obfuscations=_obfuscations(raw["obfuscations"], f"{context}.obfuscations"),
            decomposition=_scope_texts(
                raw["decomposition"], f"{context}.decomposition"
            ),
            cross_script=_scope_texts(
                raw["cross_script"],
                f"{context}.cross_script",
            ),
        )

    def export_mapping(self) -> JsonObject:
        output = self._base_mapping()
        output.update(
            {
                "obfuscations": _copy_obfuscations(self.obfuscations),
                "decomposition": _serialize_scope_texts(self.decomposition),
                "cross_script": _serialize_scope_texts(self.cross_script),
            }
        )
        return output

    def _base_mapping(self) -> JsonObject:
        return super().export_mapping()

    def input_text(self, obfuscation_type: str, scope: ObfuscationScope) -> str:
        if scope == ObfuscationScope.ORIGINAL:
            return self.text
        values = (
            self.decomposition
            if obfuscation_type == "decomposition"
            else self.cross_script
        )
        return values[scope]

    def obf_density(self, scope: ObfuscationScope) -> float | int:
        if scope == ObfuscationScope.ORIGINAL:
            return 0
        densities = require_mapping(
            self.obfuscations["obf_density"], "obfuscations.obf_density"
        )
        return require_number(
            densities[scope.value], f"obfuscations.obf_density.{scope.value}"
        )


def load_annotated_corpus(file_path: PathLike) -> list[CorpusEntry]:
    return parse_annotated_corpus(load_json(file_path))


def load_obfuscated_corpus(file_path: PathLike) -> list[ObfuscatedCorpusEntry]:
    return parse_obfuscated_corpus(load_json(file_path))


def save_annotated_corpus(corpus: list[CorpusEntry], file_path: PathLike) -> None:
    save_json([entry.to_source_mapping() for entry in corpus], file_path)


def save_obfuscated_corpus(
    corpus: list[ObfuscatedCorpusEntry], file_path: PathLike
) -> None:
    save_json([entry.export_mapping() for entry in corpus], file_path)


def parse_annotated_corpus(data: object) -> list[CorpusEntry]:
    return [
        CorpusEntry.parse_mapping(entry, f"AnnotatedCorpus[{index}]")
        for index, entry in enumerate(require_list(data, "AnnotatedCorpus"))
    ]


def parse_obfuscated_corpus(data: object) -> list[ObfuscatedCorpusEntry]:
    return [
        ObfuscatedCorpusEntry.parse_mapping(entry, f"ObfuscatedCorpus[{index}]")
        for index, entry in enumerate(require_list(data, "ObfuscatedCorpus"))
    ]


def _entry_label(raw: JsonObject, context: str) -> tuple[str, object]:
    has_expected_label = "expected_label" in raw
    has_label = "label" in raw
    if has_expected_label and has_label:
        raise ValueError(f"{context} must not include both label and expected_label")
    if has_expected_label:
        return ("expected_label", raw["expected_label"])
    if has_label:
        return ("label", raw["label"])
    raise ValueError(f"{context} missing required key: label")


def _semantic_anchors(value: object, context: str) -> list[JsonObject]:
    anchors: list[JsonObject] = []
    for index, raw_anchor in enumerate(require_list(value, context)):
        anchor_context = f"{context}[{index}]"
        anchor = require_mapping(raw_anchor, anchor_context)
        require_keys(anchor, {"text"}, anchor_context)
        require_string(anchor["text"], f"{anchor_context}.text")
        anchors.append(dict(anchor))
    return anchors


def _scope_texts(value: object, context: str) -> dict[ObfuscationScope, str]:
    raw = require_mapping(value, context)
    require_keys(raw, SCOPES, context)
    return {
        scope: require_string(raw[scope.value], f"{context}.{scope.value}")
        for scope in ObfuscationScope
    }


def _serialize_scope_texts(values: dict[ObfuscationScope, str]) -> JsonObject:
    return {scope.value: values[scope] for scope in ObfuscationScope}


def _obfuscations(value: object, context: str) -> JsonObject:
    raw = require_mapping(value, context)
    require_keys(raw, {"anchor", "background", "obf_density"}, context)
    anchor = [
        record.export_mapping()
        for index, record in enumerate(require_list(raw["anchor"], f"{context}.anchor"))
        for record in [
            GlyphObfuscationRecord.parse_mapping(record, f"{context}.anchor[{index}]")
        ]
    ]
    background = [
        record.export_mapping()
        for index, record in enumerate(
            require_list(raw["background"], f"{context}.background")
        )
        for record in [
            GlyphObfuscationRecord.parse_mapping(
                record, f"{context}.background[{index}]"
            )
        ]
    ]
    obf_density = _scope_densities(raw["obf_density"], f"{context}.obf_density")
    return {"anchor": anchor, "background": background, "obf_density": obf_density}


def _scope_densities(value: object, context: str) -> JsonObject:
    raw = require_mapping(value, context)
    require_keys(raw, SCOPES, context)
    densities = {
        scope.value: require_number(raw[scope.value], f"{context}.{scope.value}")
        for scope in ObfuscationScope
    }
    for scope, density in densities.items():
        if density < 0 or density > 1:
            raise ValueError(f"{context}.{scope} must be between 0 and 1")
    return densities


def _copy_obfuscations(obfuscations: JsonObject) -> JsonObject:
    return {
        "anchor": [
            dict(record)
            for record in require_list(obfuscations["anchor"], "obfuscations.anchor")
        ],
        "background": [
            dict(record)
            for record in require_list(
                obfuscations["background"], "obfuscations.background"
            )
        ],
        "obf_density": dict(
            require_mapping(obfuscations["obf_density"], "obfuscations.obf_density")
        ),
    }

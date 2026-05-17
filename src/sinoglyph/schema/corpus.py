from __future__ import annotations

from dataclasses import dataclass

from sinoglyph.io import PathLike, load_json, save_json
from sinoglyph.schema.base import JsonObject
from sinoglyph.schema.types import ModerationLabel, TaskVariant
from sinoglyph.schema.utils import (
    require_enum,
    require_keys,
    require_list,
    require_mapping,
    require_number,
    require_string,
)

VARIANTS = tuple(item.value for item in TaskVariant)


@dataclass(frozen=True)
class SubstitutionRecord:
    character: str
    decomposition: str
    perturbation: str

    @classmethod
    def parse_mapping(cls, mapping: object, context: str) -> "SubstitutionRecord":
        raw = require_mapping(mapping, context)
        require_keys(raw, {"character", "decomposition", "perturbation"}, context)
        return cls(
            character=require_string(raw["character"], f"{context}.character"),
            decomposition=require_string(
                raw["decomposition"], f"{context}.decomposition"
            ),
            perturbation=require_string(raw["perturbation"], f"{context}.perturbation"),
        )

    def export_mapping(self) -> JsonObject:
        return {
            "character": self.character,
            "decomposition": self.decomposition,
            "perturbation": self.perturbation,
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
class PerturbedCorpusEntry(CorpusEntry):
    substitutions: JsonObject
    decomposition: dict[TaskVariant, str]
    perturbation: dict[TaskVariant, str]

    @classmethod
    def parse_mapping(
        cls,
        mapping: object,
        context: str = "perturbed corpus entry",
    ) -> "PerturbedCorpusEntry":
        raw = require_mapping(mapping, context)
        base = CorpusEntry.parse_mapping(raw, context)
        require_keys(raw, {"substitutions", "decomposition", "perturbation"}, context)
        return cls(
            id=base.id,
            text=base.text,
            expected_label=base.expected_label,
            semantic_anchors=base.semantic_anchors,
            substitutions=_substitutions(
                raw["substitutions"], f"{context}.substitutions"
            ),
            decomposition=_variant_texts(
                raw["decomposition"], f"{context}.decomposition"
            ),
            perturbation=_variant_texts(raw["perturbation"], f"{context}.perturbation"),
        )

    def export_mapping(self) -> JsonObject:
        output = self._base_mapping()
        output.update(
            {
                "substitutions": _copy_substitutions(self.substitutions),
                "decomposition": _serialize_variant_texts(self.decomposition),
                "perturbation": _serialize_variant_texts(self.perturbation),
            }
        )
        return output

    def _base_mapping(self) -> JsonObject:
        return super().export_mapping()

    def input_text(self, source: str, variant: TaskVariant) -> str:
        if source == "text":
            return self.text
        values = self.decomposition if source == "decomposition" else self.perturbation
        return values[variant]

    def substitution_fraction(self, source: str, variant: TaskVariant) -> float | int:
        if source == "text":
            return 0
        fractions = require_mapping(
            self.substitutions["fraction"], "substitutions.fraction"
        )
        return require_number(
            fractions[variant.value], f"substitutions.fraction.{variant.value}"
        )


def load_annotated_corpus(file_path: PathLike) -> list[CorpusEntry]:
    return parse_annotated_corpus(load_json(file_path))


def load_perturbed_corpus(file_path: PathLike) -> list[PerturbedCorpusEntry]:
    return parse_perturbed_corpus(load_json(file_path))


def save_annotated_corpus(corpus: list[CorpusEntry], file_path: PathLike) -> None:
    save_json([entry.to_source_mapping() for entry in corpus], file_path)


def save_perturbed_corpus(
    corpus: list[PerturbedCorpusEntry], file_path: PathLike
) -> None:
    save_json([entry.export_mapping() for entry in corpus], file_path)


def parse_annotated_corpus(data: object) -> list[CorpusEntry]:
    return [
        CorpusEntry.parse_mapping(entry, f"AnnotatedCorpus[{index}]")
        for index, entry in enumerate(require_list(data, "AnnotatedCorpus"))
    ]


def parse_perturbed_corpus(data: object) -> list[PerturbedCorpusEntry]:
    return [
        PerturbedCorpusEntry.parse_mapping(entry, f"PerturbedCorpus[{index}]")
        for index, entry in enumerate(require_list(data, "PerturbedCorpus"))
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


def _variant_texts(value: object, context: str) -> dict[TaskVariant, str]:
    raw = require_mapping(value, context)
    require_keys(raw, VARIANTS, context)
    return {
        variant: require_string(raw[variant.value], f"{context}.{variant.value}")
        for variant in TaskVariant
    }


def _serialize_variant_texts(values: dict[TaskVariant, str]) -> JsonObject:
    return {variant.value: values[variant] for variant in TaskVariant}


def _substitutions(value: object, context: str) -> JsonObject:
    raw = require_mapping(value, context)
    require_keys(raw, {"anchor", "non_anchor", "fraction"}, context)
    anchor = [
        record.export_mapping()
        for index, record in enumerate(require_list(raw["anchor"], f"{context}.anchor"))
        for record in [
            SubstitutionRecord.parse_mapping(record, f"{context}.anchor[{index}]")
        ]
    ]
    non_anchor = [
        record.export_mapping()
        for index, record in enumerate(
            require_list(raw["non_anchor"], f"{context}.non_anchor")
        )
        for record in [
            SubstitutionRecord.parse_mapping(record, f"{context}.non_anchor[{index}]")
        ]
    ]
    fraction = _variant_fractions(raw["fraction"], f"{context}.fraction")
    return {"anchor": anchor, "non_anchor": non_anchor, "fraction": fraction}


def _variant_fractions(value: object, context: str) -> JsonObject:
    raw = require_mapping(value, context)
    require_keys(raw, VARIANTS, context)
    fractions = {
        variant.value: require_number(raw[variant.value], f"{context}.{variant.value}")
        for variant in TaskVariant
    }
    for variant, fraction in fractions.items():
        if fraction < 0 or fraction > 1:
            raise ValueError(f"{context}.{variant} must be between 0 and 1")
    return fractions


def _copy_substitutions(substitutions: JsonObject) -> JsonObject:
    return {
        "anchor": [
            dict(record)
            for record in require_list(substitutions["anchor"], "substitutions.anchor")
        ],
        "non_anchor": [
            dict(record)
            for record in require_list(
                substitutions["non_anchor"], "substitutions.non_anchor"
            )
        ],
        "fraction": dict(
            require_mapping(substitutions["fraction"], "substitutions.fraction")
        ),
    }

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sinoglyph.io import PathLike, load_json, save_json
from sinoglyph.schema.base import JsonObject
from sinoglyph.schema.utils import (
    require_keys,
    require_list,
    require_mapping,
    require_string,
)

CharacterUnitType = Literal["character", "radical"]
CharacterParts = list[str]
CharacterPerturbations = list[list[str]]


def _string_list(value: object, context: str) -> list[str]:
    return [
        require_string(item, f"{context}[{index}]")
        for index, item in enumerate(require_list(value, context))
    ]


def _perturbations(value: object, context: str) -> CharacterPerturbations:
    return [
        _string_list(variant, f"{context}[{index}]")
        for index, variant in enumerate(require_list(value, context))
    ]


@dataclass(frozen=True)
class CharacterCatalogEntry:
    parts: CharacterParts
    perturbations: CharacterPerturbations

    @classmethod
    def parse_mapping(cls, mapping: object, context: str) -> "CharacterCatalogEntry":
        raw = require_mapping(mapping, context)
        require_keys(raw, {"parts", "perturbations"}, context)
        return cls(
            parts=_string_list(raw["parts"], f"{context}.parts"),
            perturbations=_perturbations(
                raw["perturbations"], f"{context}.perturbations"
            ),
        )

    def export_mapping(self) -> JsonObject:
        return {
            "parts": list(self.parts),
            "perturbations": [list(variant) for variant in self.perturbations],
        }


@dataclass(frozen=True)
class CharacterDecomposition:
    entries: dict[str, CharacterParts]

    @classmethod
    def parse_mapping(cls, mapping: object) -> "CharacterDecomposition":
        raw = require_mapping(mapping, "CharacterDecomposition")
        return cls(
            {
                require_string(character, "CharacterDecomposition key"): _string_list(
                    parts,
                    f"CharacterDecomposition[{character!r}]",
                )
                for character, parts in raw.items()
            }
        )

    @classmethod
    def load_json(cls, file_path: PathLike) -> "CharacterDecomposition":
        return cls.parse_mapping(load_json(file_path))

    def export_mapping(self) -> JsonObject:
        return {character: list(parts) for character, parts in self.entries.items()}

    def save_json(self, file_path: PathLike) -> None:
        save_json(self.export_mapping(), file_path)

    def __contains__(self, character: object) -> bool:
        return isinstance(character, str) and character in self.entries

    def keys(self):
        return self.entries.keys()

    def parts(self, character: str) -> CharacterParts:
        return list(self.entries[character])


@dataclass(frozen=True)
class CharacterSubstitutionEntry:
    unit_type: CharacterUnitType
    perturbations: CharacterPerturbations

    @classmethod
    def parse_mapping(
        cls, mapping: object, context: str
    ) -> "CharacterSubstitutionEntry":
        raw = require_mapping(mapping, context)
        require_keys(raw, {"unit_type", "perturbation"}, context)
        unit_type = require_string(raw["unit_type"], f"{context}.unit_type")
        if unit_type not in ("character", "radical"):
            raise ValueError(f"{context}.unit_type must be 'character' or 'radical'")
        return cls(
            unit_type=unit_type,
            perturbations=_perturbations(
                raw["perturbation"], f"{context}.perturbation"
            ),
        )

    def export_mapping(self) -> JsonObject:
        return {
            "unit_type": self.unit_type,
            "perturbation": [list(variant) for variant in self.perturbations],
        }


@dataclass(frozen=True)
class CharacterSubstitution:
    entries: dict[str, CharacterSubstitutionEntry]

    @classmethod
    def parse_mapping(cls, mapping: object) -> "CharacterSubstitution":
        raw = require_mapping(mapping, "CharacterSubstitution")
        return cls(
            {
                require_string(
                    unit, "CharacterSubstitution key"
                ): CharacterSubstitutionEntry.parse_mapping(
                    entry,
                    f"CharacterSubstitution[{unit!r}]",
                )
                for unit, entry in raw.items()
            }
        )

    @classmethod
    def load_json(cls, file_path: PathLike) -> "CharacterSubstitution":
        return cls.parse_mapping(load_json(file_path))

    def export_mapping(self) -> JsonObject:
        return {unit: entry.export_mapping() for unit, entry in self.entries.items()}

    def save_json(self, file_path: PathLike) -> None:
        save_json(self.export_mapping(), file_path)

    def __contains__(self, unit: object) -> bool:
        return isinstance(unit, str) and unit in self.entries

    def keys(self):
        return self.entries.keys()

    def unit_type(self, unit: str) -> CharacterUnitType:
        return self.entries[unit].unit_type

    def perturbations(self, unit: str) -> CharacterPerturbations:
        return [list(variant) for variant in self.entries[unit].perturbations]


@dataclass(frozen=True)
class CharacterPerturbCatalog:
    entries: dict[str, CharacterCatalogEntry]

    @classmethod
    def parse_mapping(cls, mapping: object) -> "CharacterPerturbCatalog":
        raw = require_mapping(mapping, "CharacterPerturbCatalog")
        return cls(
            {
                require_string(
                    character, "CharacterPerturbCatalog key"
                ): CharacterCatalogEntry.parse_mapping(
                    entry,
                    f"CharacterPerturbCatalog[{character!r}]",
                )
                for character, entry in raw.items()
            }
        )

    @classmethod
    def load_json(cls, file_path: PathLike) -> "CharacterPerturbCatalog":
        return cls.parse_mapping(load_json(file_path))

    def export_mapping(self) -> JsonObject:
        return {
            character: entry.export_mapping()
            for character, entry in self.entries.items()
        }

    def save_json(self, file_path: PathLike) -> None:
        save_json(self.export_mapping(), file_path)

    def __contains__(self, character: object) -> bool:
        return isinstance(character, str) and character in self.entries

    def keys(self):
        return self.entries.keys()

    def entry(self, character: str) -> CharacterCatalogEntry:
        return self.entries[character]

    def parts(self, character: str) -> CharacterParts:
        return list(self.entries[character].parts)

    def perturbations(self, character: str) -> CharacterPerturbations:
        return [list(variant) for variant in self.entries[character].perturbations]


load_character_decomposition = CharacterDecomposition.load_json
load_character_substitution = CharacterSubstitution.load_json
load_character_perturb_catalog = CharacterPerturbCatalog.load_json

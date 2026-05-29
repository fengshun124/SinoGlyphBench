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
GlyphForms = list[list[str]]


def _string_list(value: object, context: str) -> list[str]:
    return [
        require_string(item, f"{context}[{index}]")
        for index, item in enumerate(require_list(value, context))
    ]


def _glyph_forms(value: object, context: str) -> GlyphForms:
    return [
        _string_list(form, f"{context}[{index}]")
        for index, form in enumerate(require_list(value, context))
    ]


@dataclass(frozen=True)
class CharacterCatalogEntry:
    parts: CharacterParts
    glyph_forms: GlyphForms

    @classmethod
    def parse_mapping(cls, mapping: object, context: str) -> "CharacterCatalogEntry":
        raw = require_mapping(mapping, context)
        require_keys(raw, {"parts", "glyph_forms"}, context)
        return cls(
            parts=_string_list(raw["parts"], f"{context}.parts"),
            glyph_forms=_glyph_forms(raw["glyph_forms"], f"{context}.glyph_forms"),
        )

    def export_mapping(self) -> JsonObject:
        return {
            "parts": list(self.parts),
            "glyph_forms": [list(form) for form in self.glyph_forms],
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
    glyph_forms: GlyphForms

    @classmethod
    def parse_mapping(
        cls, mapping: object, context: str
    ) -> "CharacterSubstitutionEntry":
        raw = require_mapping(mapping, context)
        require_keys(raw, {"unit_type", "glyph_forms"}, context)
        unit_type = require_string(raw["unit_type"], f"{context}.unit_type")
        if unit_type not in ("character", "radical"):
            raise ValueError(f"{context}.unit_type must be 'character' or 'radical'")
        return cls(
            unit_type=unit_type,
            glyph_forms=_glyph_forms(raw["glyph_forms"], f"{context}.glyph_forms"),
        )

    def export_mapping(self) -> JsonObject:
        return {
            "unit_type": self.unit_type,
            "glyph_forms": [list(form) for form in self.glyph_forms],
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

    def glyph_forms(self, unit: str) -> GlyphForms:
        return [list(form) for form in self.entries[unit].glyph_forms]


@dataclass(frozen=True)
class GlyphObfuscationCatalog:
    entries: dict[str, CharacterCatalogEntry]

    @classmethod
    def parse_mapping(cls, mapping: object) -> "GlyphObfuscationCatalog":
        raw = require_mapping(mapping, "GlyphObfuscationCatalog")
        return cls(
            {
                require_string(
                    character, "GlyphObfuscationCatalog key"
                ): CharacterCatalogEntry.parse_mapping(
                    entry,
                    f"GlyphObfuscationCatalog[{character!r}]",
                )
                for character, entry in raw.items()
            }
        )

    @classmethod
    def load_json(cls, file_path: PathLike) -> "GlyphObfuscationCatalog":
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

    def glyph_forms(self, character: str) -> GlyphForms:
        return [list(form) for form in self.entries[character].glyph_forms]


load_character_decomposition = CharacterDecomposition.load_json
load_character_substitution = CharacterSubstitution.load_json
load_glyph_obfuscation_catalog = GlyphObfuscationCatalog.load_json

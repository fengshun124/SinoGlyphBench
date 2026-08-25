import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import TypeAlias, cast

from tqdm.auto import tqdm

from sinoglyph.io import PathLike, save_json
from sinoglyph.render import TextRenderConfig, TextRenderer
from sinoglyph.schema.base import JsonObject
from sinoglyph.schema.character import (
    CharacterDecomposition,
    CharacterSubstitution,
    GlyphObfuscationCatalog,
)

Parts: TypeAlias = tuple[str, ...]
GlyphForm: TypeAlias = tuple[str, ...]

PREFERRED_GLYPH_FORMS: dict[Parts, list[GlyphForm]] = {
    ("口", "幺"): [("ﾛ", "幺")],
    ("幺", "力"): [("幺", "ｶ")],
}


@dataclass(frozen=True)
class _CatalogCandidate:
    form: GlyphForm
    source_parts: Parts
    segments: tuple[GlyphForm, ...]
    obfuscated_part_count: int
    order: int


@dataclass(frozen=True)
class CatalogRenderConfig:
    cjk_font: PathLike
    lgc_font: PathLike
    symbol_font: PathLike
    size_px: int = 64
    pad: int = 48
    dpi: int = 300
    fg_color: str = "black"
    bg_color: str = "white"


class _CharacterCatalogBuilder:
    def __init__(
        self,
        decomposition: CharacterDecomposition,
        substitution: CharacterSubstitution,
    ) -> None:
        self._decomposition = {
            character: decomposition.parts(character)
            for character in decomposition.keys()
        }
        self._substitution = {
            unit: {
                "unit_type": substitution.unit_type(unit),
                "glyph_forms": substitution.glyph_forms(unit),
            }
            for unit in substitution.keys()
        }
        self._halfwidth_kana = _build_halfwidth_kana_map()
        self._preference_default_rank = (
            sum(len(forms) for forms in PREFERRED_GLYPH_FORMS.values()) + 1
        )
        self._layout_cache: dict[str, list[Parts]] = {}

    def build(self) -> JsonObject:
        catalog: dict[str, dict[str, object]] = {}

        for unit, entry in self._substitution.items():
            if entry["unit_type"] != "character":
                continue

            ranked = self._rank_candidates(
                self._make_direct_candidates(unit, entry["glyph_forms"])
            )
            catalog[unit] = {
                "parts": [unit],
                "glyph_forms": [list(candidate.form) for candidate in ranked],
            }

        for character in self._decomposition:
            layouts = self._build_layouts(character)
            candidates = self._make_layout_candidates(layouts)

            if character in catalog:
                existing = cast(list[list[str]], catalog[character]["glyph_forms"])
                candidates.extend(
                    _CatalogCandidate(
                        tuple(form),
                        (character,),
                        (tuple(form),),
                        1,
                        -len(existing) + index,
                    )
                    for index, form in enumerate(existing)
                )
                parts = cast(list[str], catalog[character]["parts"])
            else:
                ranked_preview = self._rank_candidates(candidates)
                parts = list(
                    ranked_preview[0].source_parts if ranked_preview else layouts[0]
                )

            ranked = self._rank_candidates(candidates)
            catalog[character] = {
                "parts": parts,
                "glyph_forms": [list(candidate.form) for candidate in ranked],
            }

        result = cast(JsonObject, catalog)
        GlyphObfuscationCatalog.parse_mapping(result)
        return result

    def _make_direct_candidates(
        self, unit: str, glyph_forms: object
    ) -> list[_CatalogCandidate]:
        candidates: list[_CatalogCandidate] = []
        for index, form in enumerate(cast(list[list[str]], glyph_forms)):
            normalized = self._normalize_form(tuple(form))
            candidates.append(
                _CatalogCandidate(normalized, (unit,), (normalized,), 1, index)
            )
        return candidates

    def _build_layouts(
        self, character: str, stack: frozenset[str] | None = None
    ) -> list[Parts]:
        if stack is None and character in self._layout_cache:
            return self._layout_cache[character]

        active_stack = frozenset() if stack is None else stack
        if character in active_stack:
            return [(character,)]

        parts = self._decomposition.get(character)
        if parts is None:
            return [(character,)]

        choices: list[list[Parts]] = []
        next_stack = active_stack | {character}
        for part in parts:
            part_choices = [(part,)]
            if part in self._decomposition:
                part_choices.extend(self._build_layouts(part, next_stack))
            choices.append(_dedupe_parts(part_choices))

        layouts = _dedupe_parts(
            tuple(token for choice in combination for token in choice)
            for combination in product(*choices)
        )
        if stack is None:
            self._layout_cache[character] = layouts
        return layouts

    def _make_layout_candidates(
        self, layouts: Sequence[Parts]
    ) -> list[_CatalogCandidate]:
        candidates: list[_CatalogCandidate] = []
        order = 0

        for layout in layouts:
            component_options: list[list[tuple[GlyphForm, int]]] = []
            for part in layout:
                options = [((part,), 0)]
                if part in self._substitution:
                    glyph_forms = cast(
                        list[list[str]], self._substitution[part]["glyph_forms"]
                    )
                    options.extend((tuple(form), 1) for form in glyph_forms)
                component_options.append(options)

            for combination in product(*component_options):
                obfuscated_part_count = sum(count for _form, count in combination)
                if obfuscated_part_count == 0:
                    continue

                raw_segments = tuple(form for form, _count in combination)
                form = self._normalize_form(_flatten_segments(raw_segments))
                segments = self._normalize_segments(raw_segments, form)
                candidates.append(
                    _CatalogCandidate(
                        form, layout, segments, obfuscated_part_count, order
                    )
                )
                order += 1

        return candidates

    def _rank_candidates(
        self, candidates: Sequence[_CatalogCandidate]
    ) -> list[_CatalogCandidate]:
        best_by_form: dict[GlyphForm, _CatalogCandidate] = {}
        for candidate in candidates:
            existing = best_by_form.get(candidate.form)
            if existing is None or self._make_rank_key(candidate) < self._make_rank_key(
                existing
            ):
                best_by_form[candidate.form] = candidate

        return sorted(best_by_form.values(), key=self._make_rank_key)

    def _make_rank_key(self, candidate: _CatalogCandidate) -> tuple[int, int, int, int]:
        return (
            self._score_preference(candidate),
            -_count_non_cjk_codepoints(candidate.form),
            -candidate.obfuscated_part_count,
            candidate.order,
        )

    def _score_preference(self, candidate: _CatalogCandidate) -> int:
        best_rank = self._preference_default_rank

        for preferred_parts, preferred_forms in PREFERRED_GLYPH_FORMS.items():
            width = len(preferred_parts)
            if width > len(candidate.source_parts):
                continue

            for start in range(0, len(candidate.source_parts) - width + 1):
                if candidate.source_parts[start : start + width] != preferred_parts:
                    continue

                local_form = _flatten_segments(
                    candidate.segments[start : start + width]
                )
                for index, preferred_form in enumerate(preferred_forms):
                    if local_form == preferred_form:
                        best_rank = min(best_rank, index)

        return best_rank

    def _normalize_form(self, form: GlyphForm) -> GlyphForm:
        if len(form) <= 1:
            return form

        return tuple(self._halfwidth_kana.get(token, token) for token in form)

    def _normalize_segments(
        self,
        segments: tuple[GlyphForm, ...],
        normalized_form: GlyphForm,
    ) -> tuple[GlyphForm, ...]:
        if len(normalized_form) <= 1:
            return segments

        return tuple(
            tuple(self._halfwidth_kana.get(token, token) for token in segment)
            for segment in segments
        )


def generate_character_catalog(
    decomposition_path: PathLike,
    substitution_path: PathLike,
    output_path: PathLike | None = None,
) -> JsonObject:
    decomposition = CharacterDecomposition.load_json(decomposition_path)
    substitution = CharacterSubstitution.load_json(substitution_path)
    result = _CharacterCatalogBuilder(decomposition, substitution).build()

    if output_path is not None:
        save_json(result, output_path)
    return result


def render_catalog_figures(
    catalog: Mapping[str, object],
    figure_dir: PathLike,
    figure_config: CatalogRenderConfig,
) -> int:
    output_dir = Path(figure_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    render_config = TextRenderConfig(
        size_px=figure_config.size_px,
        fg_color=figure_config.fg_color,
        bg_color=figure_config.bg_color,
        cjk_font=figure_config.cjk_font,
        lgc_font=figure_config.lgc_font,
        symbol_font=figure_config.symbol_font,
        dpi=figure_config.dpi,
        pad=figure_config.pad,
        align="center",
    )

    for character, raw_entry in tqdm(
        catalog.items(), desc="Rendering character catalog", unit="character"
    ):
        entry = cast(Mapping[str, object], raw_entry)
        glyph_forms = cast(list[list[str]], entry["glyph_forms"])
        lines = [character, "", *["".join(form) for form in glyph_forms]]
        output_path = output_dir / _make_catalog_figure_filename(character)
        TextRenderer("\n".join(lines), render_config).render(output_path)

    return len(catalog)


def _count_non_cjk_codepoints(form: GlyphForm) -> int:
    return sum(
        1
        for token in form
        for character in token
        if not _is_cjk_codepoint(ord(character))
    )


def _is_cjk_codepoint(codepoint: int) -> bool:
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2A6DF
        or 0x2A700 <= codepoint <= 0x2B73F
        or 0x2B740 <= codepoint <= 0x2B81F
        or 0x2B820 <= codepoint <= 0x2CEAF
        or 0x2CEB0 <= codepoint <= 0x2EBEF
        or 0x30000 <= codepoint <= 0x3134F
    )


def _flatten_segments(segments: Sequence[GlyphForm]) -> GlyphForm:
    return tuple(token for segment in segments for token in segment)


def _build_halfwidth_kana_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    halfwidth_tokens = [chr(codepoint) for codepoint in range(0xFF61, 0xFFA0)]
    halfwidth_tokens.extend(
        chr(base) + chr(mark)
        for base in range(0xFF61, 0xFFA0)
        for mark in (0xFF9E, 0xFF9F)
    )

    for halfwidth in halfwidth_tokens:
        normalized = unicodedata.normalize("NFKC", halfwidth)
        if normalized != halfwidth and len(normalized) == 1:
            mapping.setdefault(normalized, halfwidth)
    return mapping


def _dedupe_parts(parts: Sequence[Parts]) -> list[Parts]:
    seen: set[Parts] = set()
    result: list[Parts] = []
    for item in parts:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _make_catalog_figure_filename(character: str) -> str:
    if character and all(part not in character for part in {"/", "\\"}):
        return f"{character}.png"

    codepoints = "-".join(f"U+{ord(item):04X}" for item in character)
    return f"{codepoints}.png"

import math
import os
import unicodedata as ud
from itertools import groupby
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal, TypeAlias

from PIL import Image, ImageColor, ImageFont
from PIL.Image import Image as PILImage
from PIL.ImageDraw import Draw
from PIL.ImageFont import FreeTypeFont

from sinoglyph.io import PathLike, load_toml

TextRun: TypeAlias = tuple[str, str]
FontSlot: TypeAlias = Literal["cjk", "lgc", "symbol", "emoji"]

ColorLike: TypeAlias = str | tuple[int, int, int] | tuple[int, int, int, int]
TextAlign: TypeAlias = Literal["left", "center", "right"]

# Module-level caches to avoid reloading font data
_CMAP_CACHE: dict[str, frozenset[int]] = {}
_FONT_CACHE: dict[tuple[str, int], FreeTypeFont] = {}


class TextRenderConfig:
    _MAX_SIZE_PX = 1024
    _MAX_PAD_PX = 4096
    _MAX_DPI = 2400

    def __init__(
        self,
        *,
        size_px: int,
        fg_color: ColorLike,
        bg_color: ColorLike,
        cjk_font: PathLike,
        lgc_font: PathLike,
        symbol_font: PathLike | None = None,
        emoji_font: PathLike | None = None,
        dpi: int = 300,
        pad: int = 24,
        align: TextAlign = "left",
    ) -> None:
        self._validate_size_px(size_px)
        self._validate_dpi(dpi)
        self._validate_pad(pad)

        self._size_px = size_px
        self._dpi = dpi
        self._pad = pad
        self._align = self._parse_align(align)
        self._fg_color = self._parse_color(fg_color, "fg_color")
        self._bg_color = self._parse_color(bg_color, "bg_color")
        self._cjk_font = self._parse_font_path(cjk_font, "cjk_font")
        self._lgc_font = self._parse_font_path(lgc_font, "lgc_font")
        self._symbol_font = self._parse_font_path(
            self._lgc_font if symbol_font is None else symbol_font,
            "symbol_font",
        )
        self._emoji_font = self._parse_font_path(
            self._symbol_font if emoji_font is None else emoji_font,
            "emoji_font",
        )
        self._cmap_cjk = self._load_cmap(self._cjk_font, "cjk_font")
        self._cmap_lgc = self._load_cmap(self._lgc_font, "lgc_font")
        self._cmap_symbol = self._load_cmap(self._symbol_font, "symbol_font")
        self._cmap_emoji = self._load_cmap(self._emoji_font, "emoji_font")
        self._pil_cjk = self._load_font(self._cjk_font, self._size_px, "cjk_font")
        self._pil_lgc = self._load_font(self._lgc_font, self._size_px, "lgc_font")
        self._pil_symbol = self._load_font(
            self._symbol_font, self._size_px, "symbol_font"
        )
        self._pil_emoji = self._load_font(self._emoji_font, self._size_px, "emoji_font")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size_px={self._size_px}, align={self._align!r})"

    @classmethod
    def parse_dict(cls, mapping: dict[str, object]) -> "TextRenderConfig":
        if not isinstance(mapping, dict):
            raise TypeError("TextRenderConfig.parse_dict expects a mapping")
        return cls(**mapping)

    @classmethod
    def load_config(
        cls,
        config: "TextRenderConfig | dict[str, object] | PathLike",
        *,
        section: str = "render",
    ) -> "TextRenderConfig":
        if isinstance(config, cls):
            return config
        mapping = cls._load_config_mapping(config, section)
        return cls.parse_dict(mapping)

    def export_dict(self) -> dict[str, object]:
        return {
            "size_px": self._size_px,
            "fg_color": self._fg_color,
            "bg_color": self._bg_color,
            "cjk_font": str(self._cjk_font),
            "lgc_font": str(self._lgc_font),
            "symbol_font": str(self._symbol_font),
            "emoji_font": str(self._emoji_font),
            "dpi": self._dpi,
            "pad": self._pad,
            "align": self._align,
        }

    @property
    def size_px(self) -> int:
        return self._size_px

    @property
    def fg_color(self) -> tuple[int, int, int, int]:
        return self._fg_color

    @property
    def bg_color(self) -> tuple[int, int, int, int]:
        return self._bg_color

    @property
    def cjk_font(self) -> Path:
        return self._cjk_font

    @property
    def lgc_font(self) -> Path:
        return self._lgc_font

    @property
    def symbol_font(self) -> Path:
        return self._symbol_font

    @property
    def emoji_font(self) -> Path:
        return self._emoji_font

    @property
    def dpi(self) -> int:
        return self._dpi

    @property
    def pad(self) -> int:
        return self._pad

    @property
    def align(self) -> TextAlign:
        return self._align

    @property
    def pil_cjk(self) -> FreeTypeFont:
        return self._pil_cjk

    @property
    def pil_lgc(self) -> FreeTypeFont:
        return self._pil_lgc

    @property
    def pil_symbol(self) -> FreeTypeFont:
        return self._pil_symbol

    @property
    def pil_emoji(self) -> FreeTypeFont:
        return self._pil_emoji

    @property
    def cmap_cjk(self) -> frozenset[int]:
        return self._cmap_cjk

    @property
    def cmap_lgc(self) -> frozenset[int]:
        return self._cmap_lgc

    @property
    def cmap_symbol(self) -> frozenset[int]:
        return self._cmap_symbol

    @property
    def cmap_emoji(self) -> frozenset[int]:
        return self._cmap_emoji

    @staticmethod
    def _validate_size_px(size_px: object) -> None:
        if not isinstance(size_px, int):
            raise TypeError("size_px must be a positive integer")
        if size_px <= 0:
            raise ValueError("size_px must be a positive integer")
        if size_px > TextRenderConfig._MAX_SIZE_PX:
            raise ValueError(f"size_px must be at most {TextRenderConfig._MAX_SIZE_PX}")

    @staticmethod
    def _validate_dpi(dpi: object) -> None:
        if not isinstance(dpi, int):
            raise TypeError("dpi must be a positive integer")
        if dpi <= 0:
            raise ValueError("dpi must be a positive integer")
        if dpi > TextRenderConfig._MAX_DPI:
            raise ValueError(f"dpi must be at most {TextRenderConfig._MAX_DPI}")

    @staticmethod
    def _validate_pad(pad: object) -> None:
        if not isinstance(pad, int):
            raise TypeError("pad must be a non-negative integer")
        if pad < 0:
            raise ValueError("pad must be a non-negative integer")
        if pad > TextRenderConfig._MAX_PAD_PX:
            raise ValueError(f"pad must be at most {TextRenderConfig._MAX_PAD_PX}")

    @staticmethod
    def _validate_color(color: object, name: str) -> None:
        if isinstance(color, str):
            return
        if isinstance(color, tuple) and len(color) in (3, 4):
            if all(isinstance(value, int) and 0 <= value <= 255 for value in color):
                return
            raise ValueError(f"{name} tuple values must be integers between 0 and 255")
        raise TypeError(f"{name} must be a color string, RGB tuple, or RGBA tuple")

    @staticmethod
    def _validate_align(align: object) -> None:
        if not isinstance(align, str):
            raise TypeError("align must be one of: left, center, right")
        if align.lower() not in {"left", "center", "right"}:
            raise ValueError("align must be one of: left, center, right")

    @staticmethod
    def _parse_color(color: ColorLike, name: str) -> tuple[int, int, int, int]:
        TextRenderConfig._validate_color(color, name)
        if isinstance(color, str):
            try:
                return ImageColor.getcolor(color, "RGBA")
            except ValueError as e:
                raise ValueError(f"{name} is not a valid color: {color!r}") from e

        return color if len(color) == 4 else (*color, 255)

    @staticmethod
    def _parse_font_path(font_path: PathLike, name: str) -> Path:
        path = Path(font_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not point to a font file: {path}")
        return path

    @staticmethod
    def _parse_align(align: str) -> TextAlign:
        TextRenderConfig._validate_align(align)
        return align.lower()

    @staticmethod
    def _load_font(font_path: Path, size: int, name: str) -> FreeTypeFont:
        cache_key = (str(font_path), size)
        if cache_key in _FONT_CACHE:
            return _FONT_CACHE[cache_key]
        try:
            font = ImageFont.truetype(str(font_path), size=size)
            _FONT_CACHE[cache_key] = font
            return font
        except Exception as e:
            raise RuntimeError(f"{name} is not loadable by Pillow: {font_path}") from e

    @staticmethod
    def _load_cmap(font_path: Path, name: str) -> frozenset[int]:
        cache_key = str(font_path)
        if cache_key in _CMAP_CACHE:
            return _CMAP_CACHE[cache_key]

        from fontTools.ttLib import TTCollection, TTFont

        obj = None
        try:
            if font_path.suffix.lower() in {".ttc", ".otc"}:
                obj = TTCollection(str(font_path), lazy=True)
                font = obj.fonts[0]
            else:
                obj = TTFont(str(font_path), lazy=True)
                font = obj
            cmap = frozenset(
                cp for t in font["cmap"].tables if t.isUnicode() for cp in t.cmap
            )
            _CMAP_CACHE[cache_key] = cmap
            return cmap
        except Exception as e:
            raise RuntimeError(
                f"{name} does not expose a readable Unicode cmap: {font_path}"
            ) from e
        finally:
            if obj is not None and hasattr(obj, "close"):
                obj.close()

    @staticmethod
    def _load_config_mapping(
        config: dict[str, object] | PathLike,
        section: str,
    ) -> dict[str, object]:
        if isinstance(config, dict):
            raw = config
        else:
            raw = load_toml(config)

        mapping = raw.get(section, raw)
        if not isinstance(mapping, dict):
            raise TypeError(f"{section!r} config section must be a mapping")
        return mapping


class TextRenderer:
    _MAX_TEXT_CHARS = 10000
    _MAX_IMAGE_SIDE_PX = 20000
    _MAX_IMAGE_PIXELS = 64_000_000

    _FONT_CONFIG_KEYS: dict[FontSlot, str] = {
        "cjk": "cjk_font",
        "lgc": "lgc_font",
        "symbol": "symbol_font",
        "emoji": "emoji_font",
    }

    _LGC = (
        (0x0041, 0x024F),
        (0x0370, 0x03FF),
        (0x0400, 0x052F),
        (0x1C80, 0x1C8F),
        (0x1E00, 0x1EFF),
        (0x1F00, 0x1FFF),
        (0x2DE0, 0x2DFF),
        (0xA640, 0xA69F),
        (0xFE2E, 0xFE2F),
    )
    _CJK = (
        (0x2E80, 0x2FFF),
        (0x3000, 0x303F),
        (0x3040, 0x30FF),
        (0x3100, 0x312F),
        (0x3130, 0x318F),
        (0x3190, 0x31EF),
        (0x31F0, 0x31FF),
        (0x3400, 0x4DBF),
        (0x4E00, 0x9FFF),
        (0xAC00, 0xD7AF),
        (0xF900, 0xFAFF),
        (0xFE30, 0xFE4F),
        (0xFF00, 0xFFEF),
        (0x20000, 0x2EBEF),
        (0x2F800, 0x2FA1F),
    )
    _EMOJI = (
        (0x231A, 0x231B),
        (0x23E9, 0x23EC),
        (0x23F0, 0x23F0),
        (0x23F3, 0x23F3),
        (0x25FD, 0x25FE),
        (0x2600, 0x27BF),
        (0x2934, 0x2935),
        (0x2B05, 0x2B55),
        (0x3030, 0x3030),
        (0x303D, 0x303D),
        (0x3297, 0x3299),
        (0x1F000, 0x1FAFF),
    )
    _LGC_SYMBOLS = frozenset({0x2143})

    def __init__(
        self,
        text: str,
        config: TextRenderConfig | None = None,
        *,
        size_px: int | None = None,
        fg_color: ColorLike | None = None,
        bg_color: ColorLike | None = None,
        cjk_font: PathLike | None = None,
        lgc_font: PathLike | None = None,
        symbol_font: PathLike | None = None,
        emoji_font: PathLike | None = None,
        dpi: int = 300,
        pad: int = 24,
        align: TextAlign = "left",
    ) -> None:
        self._validate_text(text)
        config = self._resolve_config(
            config,
            size_px=size_px,
            fg_color=fg_color,
            bg_color=bg_color,
            cjk_font=cjk_font,
            lgc_font=lgc_font,
            symbol_font=symbol_font,
            emoji_font=emoji_font,
            dpi=dpi,
            pad=pad,
            align=align,
        )

        self._text = text
        self._config = config
        self._fonts = {
            "cjk": config.pil_cjk,
            "lgc": config.pil_lgc,
            "symbol": config.pil_symbol,
            "emoji": config.pil_emoji,
        }
        self._cmaps = {
            "cjk": config.cmap_cjk,
            "lgc": config.cmap_lgc,
            "symbol": config.cmap_symbol,
            "emoji": config.cmap_emoji,
        }
        self._font_paths = {
            "cjk": config.cjk_font,
            "lgc": config.lgc_font,
            "symbol": config.symbol_font,
            "emoji": config.emoji_font,
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(len={len(self._text)})"

    @classmethod
    def load_config(
        cls,
        text: str,
        config: TextRenderConfig | dict[str, object] | PathLike,
        *,
        section: str = "render",
    ) -> "TextRenderer":
        return cls(text, TextRenderConfig.load_config(config, section=section))

    def _validate_font_coverage(self) -> None:
        cmaps = self._cmaps
        all_supported = frozenset().union(*(cmaps.values()))

        for char in self._text:
            if char == "\n":
                continue
            cp = ord(char)
            if cp not in all_supported:
                raise ValueError(
                    f"Character {char!r} (U+{cp:04X}) is not supported by any loaded font"
                )

    def render(self, output_path: PathLike) -> PILImage:
        self._validate_font_coverage()

        if "\r" in self._text:
            raise ValueError("carriage returns are not supported")

        config = self._config
        lines = [self._layout_line(line) for line in self._text.split("\n")]
        line_widths = [self._measure_runs(runs) for runs in lines]
        ascent = max(font.getmetrics()[0] for font in self._fonts.values())
        descent = max(font.getmetrics()[1] for font in self._fonts.values())
        line_height = ascent + descent
        width = max(1, math.ceil(max(line_widths, default=0.0)) + 2 * config.pad)
        height = max(1, len(lines) * line_height + 2 * config.pad)
        self._check_image_bounds(width, height)

        image = Image.new("RGBA", (width, height), config.bg_color)
        draw = Draw(image)
        y = float(config.pad + ascent)
        for line_runs, line_width in zip(lines, line_widths):
            x = float(self._compute_line_x(line_width, width))
            for slot, run in line_runs:
                draw.text(
                    (x, y),
                    run,
                    font=self._fonts[slot],
                    fill=config.fg_color,
                    anchor="ls",
                )
                x += self._fonts[slot].getlength(run)
            y += line_height

        output = Path(output_path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs: dict[str, object] = {"dpi": (config.dpi, config.dpi)}
        if output.suffix.lower() == ".png":
            save_kwargs.update({"optimize": True, "compress_level": 9})
        temp_name = None
        try:
            with NamedTemporaryFile(
                suffix=output.suffix,
                prefix=f".{output.stem}-",
                dir=output.parent,
                delete=False,
            ) as f:
                temp_name = f.name
            image.save(temp_name, **save_kwargs)
            os.replace(temp_name, output)
        except Exception:
            if temp_name is not None:
                Path(temp_name).unlink(missing_ok=True)
            raise
        return image

    @staticmethod
    def _validate_text(text: object) -> None:
        if not isinstance(text, str):
            raise TypeError("text must be a non-empty string")
        if not text:
            raise ValueError("text must be a non-empty string")
        if len(text) > TextRenderer._MAX_TEXT_CHARS:
            raise ValueError(
                f"text must be at most {TextRenderer._MAX_TEXT_CHARS} characters"
            )

    @staticmethod
    def _check_image_bounds(width: int, height: int) -> None:
        if width > TextRenderer._MAX_IMAGE_SIDE_PX:
            raise ValueError(
                f"rendered image width must be at most {TextRenderer._MAX_IMAGE_SIDE_PX}px"
            )
        if height > TextRenderer._MAX_IMAGE_SIDE_PX:
            raise ValueError(
                f"rendered image height must be at most {TextRenderer._MAX_IMAGE_SIDE_PX}px"
            )
        if width * height > TextRenderer._MAX_IMAGE_PIXELS:
            raise ValueError(
                "rendered image area must be at most "
                f"{TextRenderer._MAX_IMAGE_PIXELS} pixels"
            )

    @staticmethod
    def _validate_config(config: object) -> None:
        if not isinstance(config, TextRenderConfig):
            raise TypeError("config must be a TextRenderConfig")

    @staticmethod
    def _resolve_config(
        config: TextRenderConfig | None,
        *,
        size_px: int | None,
        fg_color: ColorLike | None,
        bg_color: ColorLike | None,
        cjk_font: PathLike | None,
        lgc_font: PathLike | None,
        symbol_font: PathLike | None,
        emoji_font: PathLike | None,
        dpi: int,
        pad: int,
        align: TextAlign,
    ) -> TextRenderConfig:
        explicit_values = {
            "size_px": size_px,
            "fg_color": fg_color,
            "bg_color": bg_color,
            "cjk_font": cjk_font,
            "lgc_font": lgc_font,
            "symbol_font": symbol_font,
            "emoji_font": emoji_font,
        }
        if config is not None:
            TextRenderer._validate_config(config)
            if any(value is not None for value in explicit_values.values()):
                raise TypeError(
                    "explicit render options cannot be passed with TextRenderConfig"
                )
            if dpi != 300 or pad != 24 or align != "left":
                raise TypeError(
                    "dpi, pad, and align cannot be passed with TextRenderConfig"
                )
            return config

        missing = [
            name
            for name, value in explicit_values.items()
            if value is None and name not in {"symbol_font", "emoji_font"}
        ]
        if missing:
            raise TypeError(
                "missing required render options: " + ", ".join(sorted(missing))
            )

        return TextRenderConfig(
            size_px=size_px,
            fg_color=fg_color,
            bg_color=bg_color,
            cjk_font=cjk_font,
            lgc_font=lgc_font,
            symbol_font=symbol_font,
            emoji_font=emoji_font,
            dpi=dpi,
            pad=pad,
            align=align,
        )

    def _layout_line(self, text: str) -> list[TextRun]:
        if not text:
            return []
        slots = self._select_slots_for_text(text)
        self._check_glyphs_for_text(text, slots)
        return self._build_runs_for_text(text, slots)

    def _select_slots_for_text(self, text: str) -> list[FontSlot]:
        return [self._select_slot_for_char(char) for char in text]

    def _select_slot_for_char(self, char: str) -> FontSlot:
        raw_slot = self._detect_raw_slot(char)
        codepoint = ord(char)
        if codepoint in self._cmaps[raw_slot]:
            return raw_slot
        for slot in self._fallback_slots(raw_slot):
            if codepoint in self._cmaps[slot]:
                return slot
        return raw_slot

    @staticmethod
    def _fallback_slots(raw_slot: FontSlot) -> tuple[FontSlot, ...]:
        if raw_slot == "emoji":
            return ("symbol", "cjk", "lgc")
        if raw_slot == "symbol":
            return ("cjk", "emoji", "lgc")
        if raw_slot == "cjk":
            return ("symbol", "emoji", "lgc")
        return ("symbol", "cjk", "emoji")

    def _check_glyphs_for_text(self, text: str, slots: list[FontSlot]) -> None:
        if missing := next(
            (
                (char, slot)
                for char, slot in zip(text, slots)
                if ord(char) not in self._cmaps[slot]
            ),
            None,
        ):
            char, slot = missing
            raise RuntimeError(
                f"Cannot render U+{ord(char):04X} ({ud.name(char, 'UNNAMED')}) with "
                f"{self._FONT_CONFIG_KEYS[slot]}={str(self._font_paths[slot])!r}. "
                f"The character was assigned to the {slot.upper()} font, but "
                f"it does not contain the glyph."
            )

    def _build_runs_for_text(self, text: str, slots: list[FontSlot]) -> list[TextRun]:
        return [
            (slot, "".join(char for char, _ in group))
            for slot, group in groupby(zip(text, slots), key=lambda item: item[1])
        ]

    def _measure_runs(self, runs: list[TextRun]) -> float:
        return sum(self._fonts[slot].getlength(run) for slot, run in runs)

    def _compute_line_x(self, line_width: float, image_width: int) -> float:
        pad = float(self._config.pad)
        inner_width = float(image_width) - 2 * pad
        if self._config.align == "right":
            return pad + inner_width - line_width
        if self._config.align == "center":
            return pad + (inner_width - line_width) / 2
        return pad

    @classmethod
    def _check_codepoint_in_ranges(
        cls, codepoint: int, ranges: tuple[tuple[int, int], ...]
    ) -> bool:
        return any(start <= codepoint <= end for start, end in ranges)

    @classmethod
    def _detect_raw_slot(cls, char: str) -> FontSlot:
        codepoint, category = ord(char), ud.category(char)
        if cls._check_codepoint_in_ranges(codepoint, cls._EMOJI):
            return "emoji"
        if cls._check_codepoint_in_ranges(codepoint, cls._CJK):
            return "cjk"
        if (
            char == "\u200d"
            or 0xFE00 <= codepoint <= 0xFE0F
            or 0xE0100 <= codepoint <= 0xE01EF
        ):
            return "emoji"
        if codepoint in cls._LGC_SYMBOLS:
            return "lgc"
        if category.startswith("L") and cls._check_codepoint_in_ranges(
            codepoint, cls._LGC
        ):
            return "lgc"
        if codepoint <= 0x007E and (char == " " or category[0] in {"P", "N", "S", "M"}):
            return "lgc"
        if category[0] in {"P", "N", "M"}:
            return "lgc"
        if category[0] == "S":
            return "symbol"
        if category[0] == "C":
            raise ValueError(f"control character U+{codepoint:04X} is not supported")
        raise ValueError(
            f"unsupported code point U+{codepoint:04X} "
            f"({ud.name(char, 'UNNAMED')}); "
            "only LGC, CJK, symbol, and emoji text is supported."
        )

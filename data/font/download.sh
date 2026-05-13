#!/usr/bin/env sh

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd -P)"
TARGET_DIR="$SCRIPT_DIR"

# CJK: Simplified Chinese glyph forms + kana/CJK coverage
curl -fL --retry 3 -C - -o "$TARGET_DIR/NotoSansCJKsc-Regular.otf" \
  "https://github.com/notofonts/noto-cjk/raw/refs/heads/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"

curl -fL --retry 3 -C - -o "$TARGET_DIR/NotoSerifCJKsc-Regular.otf" \
  "https://github.com/notofonts/noto-cjk/raw/refs/heads/main/Serif/OTF/SimplifiedChinese/NotoSerifCJKsc-Regular.otf"

# Latin / Greek / Cyrillic
curl -fL --retry 3 -C - -o "$TARGET_DIR/NotoSans-Regular.ttf" \
  "https://notofonts.github.io/latin-greek-cyrillic/fonts/NotoSans/hinted/ttf/NotoSans-Regular.ttf"

curl -fL --retry 3 -C - -o "$TARGET_DIR/NotoSerif-Regular.ttf" \
  "https://notofonts.github.io/latin-greek-cyrillic/fonts/NotoSerif/hinted/ttf/NotoSerif-Regular.ttf"

# Math + rare symbol fallback
curl -fL --retry 3 -C - -o "$TARGET_DIR/NotoSansMath-Regular.ttf" \
  "https://notofonts.github.io/math/fonts/NotoSansMath/hinted/ttf/NotoSansMath-Regular.ttf"

# Emoji
curl -fL --retry 3 -C - -o "$TARGET_DIR/NotoEmoji[wght].ttf" \
  "https://raw.githubusercontent.com/google/fonts/main/ofl/notoemoji/NotoEmoji%5Bwght%5D.ttf"

# SinoGlyphBench

SinoGlyphBench is a benchmark and toolkit for testing how language and multimodal models handle glyph-obfuscated Chinese moderation text. It generates controlled variants from human-annotated examples, renders inputs for image-based evaluation, and measures whether models can read, recover, and moderate the intended content.

<p align="center">
  <img src="assets/SinoGlyphBench-logo.png" alt="SinoGlyphBench logo rendered as the catalog-derived cross-script form ｷ并め子字" width="250">
</p>

<p align="center"><em>拼好字 (“Piece Together Good Characters”) in its catalog-derived cross-script form: ｷ并め子字.</em></p>

Semantic anchors separate label-critical text from the surrounding context. Each entry supports original, anchor-only, background-only, and full scopes, presented either as text or as a rendered image.

## Motivation

Glyph obfuscation is diagnostically useful only when readers can still recover the intended phrase; otherwise, failure may reflect missing visual evidence rather than moderation fragility. SinoGlyphBench therefore uses manually curated decomposition and cross-script substitutions that retain a visible trail to the source characters.

<p align="center">
  <img src="assets/SinoGlyphBench-similarity.png" alt="Examples motivating SinoGlyphBench through visual similarity between Chinese characters and obfuscated glyph forms" width="900">
</p>

<p align="center"><em>Recoverability depends on visual similarity to the source characters.</em></p>

## Corpus Panels

The corpus is constructed from [STATE-ToxiCN](https://aclanthology.org/2025.findings-acl.532/), [ToxiBenchCN (CNTP dataset)](https://aclanthology.org/2025.findings-acl.742/), and [PCR-ToxiCN](https://aclanthology.org/2025.emnlp-industry.172/). Human annotation and filtering produce a broad panel of 980 examples and a strict diagnostic panel of 157 examples.

The corpus construction balances semantic-anchor and non-anchor regions using two quantities:

- **Substitutable Count Gap:** `abs(anchor_substitutable_count - background_substitutable_count)`. During panel construction, a position is considered substitutable when its character has an entry in `data/character/catalog.json`.
- **Anchor/Non-Anchor Length Ratio:** `max(anchor_character_count / non_anchor_character_count, non_anchor_character_count / anchor_character_count)`. Anchor spans are all occurrences of each `semantic_anchors[].text`; counts are measured in characters (anchor span length vs. remaining text length).

| File                       | Role                    | Items | Filtering rule                                                                      |
| -------------------------- | ----------------------- | ----: | ----------------------------------------------------------------------------------- |
| `annotated.candidate.json` | Candidate Pool          | 3,031 | Valid source rows with at least one catalog-covered anchor and background character |
| `annotated.json`           | Broad panel             |   980 | Length ratio <= 3 and substitutable count gap <= 2                                  |
| `annotated.strict.json`    | Strict diagnostic panel |   157 | Length ratio <= 1.75 and substitutable count gap = 0                                |

**Panel-Selection Statistics (Catalog-Covered Source Positions)**

| File                       | Anchor Catalog Positions | Background Catalog Positions | Anchor - Background | Max Item Gap | Max Length Ratio |
| -------------------------- | -----------------------: | ---------------------------: | ------------------: | -----------: | ---------------: |
| `annotated.candidate.json` |                   11,457 |                       19,932 |              -8,475 |           29 |            40.50 |
| `annotated.json`           |                    3,712 |                        3,774 |                 -62 |            2 |             3.00 |
| `annotated.strict.json`    |                      554 |                          554 |                   0 |            0 |             1.75 |

## Quickstart

Set up the project with `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

Alternatively, use Conda or Mamba:

```bash
conda create -n sinoglyph python=3.12
conda activate sinoglyph
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

You can replace `conda` with `mamba` in these commands. `PYTHONPATH` is only required when importing `sinoglyph` directly; the CLI examples work from the repository root without it.

Font files are expected under `data/font/`. If they are missing, download the Noto font set used by the renderer:

```bash
sh data/font/download.sh
```

## Font and LLM Smoke Test

After installing the dependencies and fonts, use the following commands to test the renderer and an OpenAI-compatible LLM endpoint. The test renders a multilingual greeting to `/tmp/sinoglyph-greeting.png`, then asks the configured model to read it and return JSON.

Set the model endpoint variables first:

```bash
export LLM_BASE_URL="https://api.example.com/v1"
export LLM_API_KEY="YOUR_API_KEY"
export LLM_MODEL="example-model"
```

Render the greeting image:

```bash
python src/cli.py render \
  --text "你好 こんにちは 안녕하세요 Hello Привет Γεια 👋😊" \
  --cjk-font data/font/NotoSansCJKsc-Regular.otf \
  --lgc-font data/font/NotoSans-Regular.ttf \
  --symbol-font data/font/NotoSansMath-Regular.ttf \
  --emoji-font 'data/font/NotoEmoji[wght].ttf' \
  --output /tmp/sinoglyph-greeting.png \
  --size 72 \
  --fg-color black \
  --bg-color white \
  --align center \
  --dpi 300 \
  --pad 32
```

Write a minimal response schema for the image-reading check:

```bash
cat > /tmp/sinoglyph-read-schema.json <<'JSON'
{
  "type": "object",
  "required": ["read_text"],
  "additionalProperties": false,
  "properties": {
    "read_text": {
      "type": "string",
      "minLength": 1
    }
  }
}
JSON
```

Send the rendered image to the model:

```bash
python src/cli.py chat \
  --system "Read the image and return JSON only." \
  --message "Read the text in this image. Return the exact visible text." \
  --image /tmp/sinoglyph-greeting.png \
  --json-schema /tmp/sinoglyph-read-schema.json \
  --max-retries 2 \
  --param max_tokens=1024 \
  --param temperature=0
```

The `chat` command reads `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` by default. The `--image` option accepts a local image path, URL, or data URL; local paths are encoded and attached to the chat request automatically.

## Pipeline

The workflow builds the character catalog and obfuscated corpus, renders text when needed, and runs model evaluation from a TOML configuration.

<p align="center">
  <img src="assets/SinoGlyphBench-pipeline.png" alt="SinoGlyphBench corpus construction and evaluation workflow" width="1100">
</p>

<p align="center"><em>Benchmark construction and paired text/image evaluation.</em></p>

The Python implementation is organized by function: `sinoglyph.pipeline` builds catalog and corpus artifacts, `sinoglyph.evaluate` runs model evaluation, `sinoglyph.schema` validates structured data, and `sinoglyph.render` renders text probes and image-mode inputs.

### Build the Character Catalog

The catalog combines character decompositions with substitution rules. It can also render per-character catalog figures for inspection.

```bash
python src/cli.py catalog
```

Example with explicit paths and rendering options:

```bash
python src/cli.py catalog \
  --decomposition data/character/decomposition.json \
  --substitution data/character/substitution.json \
  --catalog-output data/character/catalog.json \
  --figure-dir data/character/catalog_figure \
  --cjk-font data/font/NotoSansCJKsc-Regular.otf \
  --lgc-font data/font/NotoSans-Regular.ttf \
  --symbol-font data/font/NotoSansMath-Regular.ttf \
  --size 64 \
  --pad 48 \
  --skip-figures
```

This example skips figure generation. Remove `--skip-figures` to render catalog figures alongside the JSON catalog.

### Build the Obfuscated Corpus

The corpus builder applies the character catalog to annotated examples and separates anchor and background obfuscations. By default, it reads `data/corpus/annotated.json` and writes `data/corpus/obfuscated.json`. The output preserves each example's `id` and `text`, then adds `anchor`, `background`, and `obf_density` fields under `obfuscations`.

```bash
python src/cli.py corpus
```

Example with explicit paths:

```bash
python src/cli.py corpus \
  --annotated data/corpus/annotated.json \
  --catalog data/character/catalog.json \
  --output data/corpus/obfuscated.json
```

### Render a Text Probe

The renderer is useful for checking how an obfuscated string will look before image-mode evaluation.

```bash
python src/cli.py render \
  --text "测试" \
  --cjk-font data/font/NotoSansCJKsc-Regular.otf \
  --lgc-font data/font/NotoSans-Regular.ttf \
  --symbol-font data/font/NotoSansMath-Regular.ttf
```

Example with explicit rendering options:

```bash
python src/cli.py render \
  --text "测试文本" \
  --cjk-font data/font/NotoSansCJKsc-Regular.otf \
  --lgc-font data/font/NotoSans-Regular.ttf \
  --symbol-font data/font/NotoSansMath-Regular.ttf \
  --emoji-font 'data/font/NotoEmoji[wght].ttf' \
  --output /tmp/sinoglyph-render.png \
  --size 96 \
  --fg-color black \
  --bg-color white \
  --align center \
  --dpi 300 \
  --pad 32
```

### Configure Evaluation Tasks

Evaluation tasks combine three dimensions:

- `modality`: `text` sends the string directly; `image` renders the string first and sends the image.
- `obfuscation_type`: `decomposition` rewrites characters as visible components; `cross_script` is the config identifier for cross-script substitution with visually similar symbols.
- `scope`: `original`, `anchor_only`, `background_only`, or `full`.

For image tasks, the config must include a `[render]` section with fonts and rendering settings.

The default evaluation response schema expects one JSON object with:

- `read_text`: the literal visible input in reading order.
- `recovered_text`: the best-supported intended Chinese message.
- `interpretation`: a concise moderation-relevant interpretation.
- `judge`: one of `hostile`, `abusive`, `benign`, or `context_dependent`.

The TOML file names the environment variables that provide the model endpoint and credentials:

```toml
[llm]
base_url_env = "LLM_BASE_URL"
api_key_env = "LLM_API_KEY"
model_env = "LLM_MODEL"
max_tokens = 8192
```

The bundled TOML files are templates: replace `<LLM_BASE_URL>`, `<LLM_API_KEY>`, `<LLM_MODEL>`, and `<model_name>` with your environment-variable names and run name. You can instead use direct `base_url`, `api_key`, and `model` values in a private config, but do not commit credentials.

### Run Evaluation

Evaluation reads a TOML config, sends each configured task to an OpenAI-compatible chat API, validates JSON responses, writes resumable cache entries, and emits a single result JSON.

```bash
python src/cli.py evaluate -c config/example.generic.toml
python src/cli.py evaluate -c config/example.obf_aware.toml
```

Fully customized example:

```bash
python src/cli.py evaluate \
  --config config/example.generic.toml \
  --output evaluation/toxicity-identification.json \
  --cache-dir cache/toxicity-identification \
  --n-jobs 4
```

## Validation

Check the CLI surface:

```bash
python src/cli.py --help
python src/cli.py catalog --help
python src/cli.py corpus --help
python src/cli.py evaluate --help
python src/cli.py render --help
python src/cli.py chat --help
```

Run lightweight local smoke checks:

```bash
python src/cli.py corpus --output /tmp/obfuscated.json
python src/cli.py render \
  --text "测试" \
  --cjk-font data/font/NotoSansCJKsc-Regular.otf \
  --lgc-font data/font/NotoSans-Regular.ttf \
  --symbol-font data/font/NotoSansMath-Regular.ttf \
  --output /tmp/render.png
```

Full evaluation is not part of the default smoke check because it requires model credentials and external API calls.

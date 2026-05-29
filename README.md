# SinoGlyphBench

SinoGlyphBench is a benchmark and toolkit for evaluating how language and multimodal models handle Chinese moderation text under glyph-level obfuscation. It generates obfuscated variants from human-annotated examples, can render inputs as images for visual-language model evaluation, and provides tools to measure whether models can read, recover, and interpret the intended content.

The benchmark uses semantic anchors to separate meaning-critical characters from surrounding context. Each entry can be evaluated as original text, anchor-only obfuscations, background-only obfuscations, or full obfuscations. Each variant can be sent directly as text or rendered as an image for VLM-style evaluation.

## Corpus Panels

The corpus is constructed from [STATE-ToxiCN](https://aclanthology.org/2025.findings-acl.532/), [ToxicBenchCN (CNTP dataset)](https://aclanthology.org/2025.findings-acl.742/) and [PCR-ToxiCN](https://aclanthology.org/2025.emnlp-industry.172/). With human annotation and filtering, a final set of 157 high-quality examples was selected for the core evaluation panel, and a broader set of 980 examples was selected for a lower-cost panel.

The construction of the corpus balances semantic-anchor and non-anchor regions using two quantities:

- **Substitutable-count difference:** `abs(anchor_substitutable_count - non_anchor_substitutable_count)`, where substitutable characters are those covered by `data/character/catalog.json`.
- **Anchor/non-anchor length ratio:** `max(anchor_character_count / non_anchor_character_count, non_anchor_character_count / anchor_character_count)`. Anchor spans are all occurrences of each `semantic_anchors[].text`; counts are measured in characters (anchor span length vs. remaining text length).

| File                      | Role                  | Items | Filtering rule                                                                    |
| ------------------------- | --------------------- | ----: | --------------------------------------------------------------------------------- |
| `annotated.eligible.json` | Candidate Pool        | 3,031 | Valid source rows with at least one substitutable anchor and non-anchor character |
| `annotated.json`          | Broad/economy panel   |   980 | Length ratio <= 3 and substitutable-count difference <= 2                         |
| `annotated.strict.json`   | Core diagnostic panel |   157 | Length ratio <= 1.75 and substitutable-count difference = 0                       |

| File                      | Anchor subs | Non-anchor subs | Global diff | Max item diff | Max length ratio |
| ------------------------- | ----------: | --------------: | ----------: | ------------: | ---------------: |
| `annotated.eligible.json` |      11,457 |          19,932 |      -8,475 |            29 |            40.50 |
| `annotated.json`          |       3,712 |           3,774 |         -62 |             2 |             3.00 |
| `annotated.strict.json`   |         554 |             554 |           0 |             0 |             1.75 |

## Quickstart

Create an environment with `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

Or create one with Conda or Mamba:

```bash
conda create -n sinoglyph python=3.12
conda activate sinoglyph
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

`mamba` can be used in place of `conda` for the same commands. The `PYTHONPATH` line is only needed when importing `sinoglyph` modules directly from Python; the CLI examples work as shown from the repository root.

Font files are expected under `data/font/`. If they are missing, download the Noto font set used by the renderer:

```bash
sh data/font/download.sh
```

## Font And LLM Smoke Test

After installing dependencies and fonts, use this command sequence to check that both the renderer and your OpenAI-compatible LLM configuration work. It first renders a multilingual greeting to `/tmp/sinoglyph-greeting.png`, then asks the configured model to read that rendered image and return JSON.

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

The full workflow is: build the character catalog, build an obfuscated corpus from annotations, optionally render text probes, then run model evaluation from a TOML config.

The Python implementation is organized by function: `sinoglyph.pipeline` builds catalog and corpus artifacts, `sinoglyph.evaluate` runs model evaluation, `sinoglyph.schema` validates structured data, and `sinoglyph.render` renders text probes and image-mode inputs.

### Build The Character Catalog

The catalog combines character decompositions with substitution rules. It can also render per-character catalog figures for inspection.

```bash
python src/cli.py catalog
```

Fully customized example:

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

Remove `--skip-figures` when you want the rendered catalog figures as well as the JSON catalog.

### Build The Perturbed Corpus

The corpus builder applies the character catalog to annotated moderation examples and separates obfuscations into anchor and background positions. By default it reads `data/corpus/annotated.json` and writes `data/corpus/obfuscated.json`. The output preserves each example's `id` and `text` and adds an `obfuscations` object with `anchor`, `background`, and `obf_density` fields so you can map obfuscations back to the original examples.

```bash
python src/cli.py corpus
```

Fully customized example:

```bash
python src/cli.py corpus \
  --annotated data/corpus/annotated.json \
  --catalog data/character/catalog.json \
  --output data/corpus/obfuscated.json
```

### Render A Text Probe

The renderer is useful for checking how an obfuscated string will look before image-mode evaluation.

```bash
python src/cli.py render \
  --text "测试" \
  --cjk-font data/font/NotoSansCJKsc-Regular.otf \
  --lgc-font data/font/NotoSans-Regular.ttf \
  --symbol-font data/font/NotoSansMath-Regular.ttf
```

Fully customized example:

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
- `obfuscation_type`: `decomposition` uses decomposed character parts, and `cross_script` uses substituted glyph-like variants.
- `scope`: `original`, `anchor_only`, `background_only`, or `full`.

For image tasks, the config must include a `[render]` section with fonts and rendering settings.

The default evaluation response schema expects one JSON object with:

- `read_text`: the literal visible input in reading order.
- `recovered_text`: the best-supported intended Chinese message.
- `interpretation`: a concise moderation-relevant interpretation.
- `judge`: one of `hostile`, `abusive`, `benign`, or `context_dependent`.

LLM credentials and endpoints are usually configured through environment variables named by the TOML file:

```toml
[llm]
base_url_env = "LLM_BASE_URL"
api_key_env = "LLM_API_KEY"
model_env = "LLM_MODEL"
max_tokens = 8192
```

The runner resolves those environment variable names at runtime. You can also put direct `base_url`, `api_key`, and `model` values in a private local config, but avoid committing credentials.

### Run Evaluation

Evaluation reads a TOML config, sends each configured task to an OpenAI-compatible chat API, validates JSON responses, writes resumable cache entries, and emits a single result JSON.

```bash
python src/cli.py evaluate -c config/example.generic.toml
python src/cli.py evaluate -c config/example.obfuscation_aware.toml
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

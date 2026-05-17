# SinoGlyphBench

SinoGlyphBench is a benchmark and toolkit for evaluating how language and multimodal models handle Chinese moderation text under glyph-level obfuscation. It builds perturbation variants from human-annotated examples, renders text as images when needed, and evaluates whether models can read, recover, and judge the intended content.

The benchmark uses semantic anchors to separate meaning-critical characters from surrounding context. Each entry can be evaluated as original text, anchor-only perturbations, non-anchor-only perturbations, or full perturbations. Each variant can be sent directly as text or rendered as an image for VLM-style evaluation.

## What's Included

- `data/character/`: character decomposition data, substitution rules, the generated perturbation catalog, and optional catalog figures.
- `data/corpus/`: raw annotation shards, literature-source corpora, annotated corpus variants, and generated perturbed corpus variants.
- `config/`: TOML evaluation configs for model runs.
- `evaluation/`: evaluation JSON outputs.
- `cache/`: resumable per-entry checkpoints for evaluation runs.
- `src/sinoglyph/`: schema validation, corpus/catalog builders, rendering, LLM calls, and evaluation flow.

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

The full workflow is: build the character catalog, build a perturbed corpus from annotations, optionally render text probes, then run model evaluation from a TOML config.

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

The corpus builder applies the character catalog to annotated moderation examples and separates substitutions into anchor and non-anchor positions.

```bash
python src/cli.py corpus
```

Fully customized example:

```bash
python src/cli.py corpus \
  --annotated data/corpus/annotated.json \
  --catalog data/character/catalog.json \
  --output data/corpus/perturbed.json
```

### Render A Text Probe

The renderer is useful for checking how a perturbed string will look before image-mode evaluation.

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

- `input_type`: `text` sends the string directly; `image` renders the string first and sends the image.
- `source`: `text` uses the original text, `decomposition` uses decomposed character parts, and `perturbation` uses substituted glyph-like variants.
- `variant`: `original`, `anchor_only`, `non_anchor_only`, or `full`.

For `source = "text"`, the variant must be `original`. For image tasks, the config must include a `[render]` section with fonts and rendering settings.

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
python src/cli.py evaluate -c config/example.toml
```

Fully customized example:

```bash
python src/cli.py evaluate \
  --config config/example.toml \
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
python src/cli.py corpus --output /tmp/perturbed.json
python src/cli.py render \
  --text "测试" \
  --cjk-font data/font/NotoSansCJKsc-Regular.otf \
  --lgc-font data/font/NotoSans-Regular.ttf \
  --symbol-font data/font/NotoSansMath-Regular.ttf \
  --output /tmp/render.png
```

Full evaluation is not part of the default smoke check because it requires model credentials and external API calls.

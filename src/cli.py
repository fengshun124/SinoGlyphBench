import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import click

from sinoglyph.io import load_env_file
from sinoglyph.pipeline.corpus import generate_obfuscated_corpus


@click.group()
def run_cli() -> None:
    pass


@run_cli.command("catalog")
@click.option(
    "-decomp",
    "--decomposition",
    "decomposition_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("data/character/decomposition.json"),
    show_default=True,
    help="Character decomposition JSON.",
)
@click.option(
    "-sub",
    "--substitution",
    "substitution_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("data/character/substitution.json"),
    show_default=True,
    help="Character substitution JSON.",
)
@click.option(
    "-o",
    "--catalog-output",
    "catalog_output_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("data/character/catalog.json"),
    show_default=True,
    help="Generated catalog JSON.",
)
@click.option(
    "--figure-dir",
    "figure_dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("data/character/catalog_figure"),
    show_default=True,
    help="Directory for rendered character figures.",
)
@click.option(
    "--cjk-font",
    "cjk_font",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("data/font/NotoSansCJKsc-Regular.otf"),
    show_default=True,
    help="CJK font file for rendering.",
)
@click.option(
    "--lgc-font",
    "lgc_font",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("data/font/NotoSans-Regular.ttf"),
    show_default=True,
    help="Latin, Greek, and Cyrillic font file for rendering.",
)
@click.option(
    "--symbol-font",
    "symbol_font",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("data/font/NotoSansMath-Regular.ttf"),
    show_default=True,
    help="Symbol font file for rendering.",
)
@click.option(
    "--size",
    type=int,
    default=64,
    show_default=True,
    help="Rendered figure font size in pixels.",
)
@click.option(
    "--pad",
    type=int,
    default=48,
    show_default=True,
    help="Rendered figure padding in pixels.",
)
@click.option(
    "--skip-figures",
    is_flag=True,
    help="Only write the catalog JSON.",
)
def build_catalog_command(
    decomposition_path: Path,
    substitution_path: Path,
    catalog_output_path: Path,
    figure_dir: Path,
    cjk_font: Path,
    lgc_font: Path,
    symbol_font: Path,
    size: int,
    pad: int,
    skip_figures: bool,
) -> None:
    from sinoglyph.pipeline.catalog import (
        CatalogRenderConfig,
        generate_character_catalog,
        render_catalog_figures,
    )

    generated = generate_character_catalog(
        decomposition_path, substitution_path, catalog_output_path
    )
    click.echo(f"Wrote {len(generated)} catalog entries to {catalog_output_path}")

    if skip_figures:
        return

    figure_config = CatalogRenderConfig(
        cjk_font=cjk_font,
        lgc_font=lgc_font,
        symbol_font=symbol_font,
        size_px=size,
        dpi=300,
        pad=pad,
    )
    rendered = render_catalog_figures(generated, figure_dir, figure_config)
    click.echo(f"Wrote {rendered} catalog figures to {figure_dir}")


@run_cli.command("corpus")
@click.option(
    "-a",
    "--annotated",
    "annotated_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("data/corpus/annotated.json"),
    show_default=True,
    help="Annotated corpus JSON.",
)
@click.option(
    "-c",
    "--catalog",
    "catalog_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("data/character/catalog.json"),
    show_default=True,
    help="Glyph obfuscation catalog JSON.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("data/corpus/obfuscated.json"),
    show_default=True,
    help="Generated obfuscated corpus JSON.",
)
def build_corpus_command(
    annotated_path: Path,
    catalog_path: Path,
    output_path: Path,
) -> None:
    generated = generate_obfuscated_corpus(annotated_path, catalog_path, output_path)
    click.echo(f"Wrote {len(generated)} obfuscated corpus entries to {output_path}")


@run_cli.command("evaluate")
@click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("config/example.generic.toml"),
    show_default=True,
    help="Evaluation TOML config.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Evaluation result JSON. Defaults to <output_dir>/<name>.json.",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Cache directory. Defaults to cache/<evaluation.name>.",
)
@click.option(
    "--n-jobs",
    type=int,
    default=None,
    help="Parallel corpus-entry workers. Defaults to evaluation.n_jobs.",
)
@click.option(
    "--env-file",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="Dotenv file to load before resolving LLM env vars. Defaults to .env in the current working directory.",
)
@click.option(
    "--env-override",
    is_flag=True,
    help="Let values from --env-file override existing shell environment variables.",
)
def evaluate_command(
    config_path: Path,
    output_path: Path | None,
    cache_dir: Path | None,
    n_jobs: int | None,
    env_file: Path | None,
    env_override: bool,
) -> None:
    from sinoglyph.evaluate import run_evaluation
    from sinoglyph.schema.evaluation import EvaluationConfig

    env_path = Path(".env") if env_file is None else env_file
    env_path = _display_path(env_path)
    evaluation = EvaluationConfig.load_toml(config_path).evaluation
    resolved_output = _display_path(
        output_path
        if output_path is not None
        else Path(evaluation.output_dir) / f"{evaluation.name}.json"
    )
    resolved_cache = _display_path(
        cache_dir if cache_dir is not None else Path(evaluation.cache_dir)
    )
    resolved_corpus = _display_path(evaluation.corpus_path)
    resolved_n_jobs = n_jobs if n_jobs is not None else evaluation.n_jobs
    click.echo("Evaluation preflight:")
    click.echo(f"  config: {config_path.expanduser().resolve(strict=False)}")
    click.echo(f"  env: {env_path} (override={env_override})")
    click.echo(f"  corpus: {resolved_corpus}")
    click.echo(f"  output: {resolved_output}")
    click.echo(f"  cache: {resolved_cache}")
    click.echo(f"  cache images: {resolved_cache / 'images'}")
    click.echo(
        f"  limit: {'full corpus' if evaluation.limit is None else evaluation.limit}"
    )
    click.echo(f"  n_jobs: {resolved_n_jobs}")
    try:
        result = run_evaluation(
            config_path,
            output_path,
            cache_dir,
            n_jobs,
            env_file=env_file,
            env_override=env_override,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    output = output_path
    if output is None:
        output = Path(evaluation.output_dir) / f"{evaluation.name}.json"
    click.echo(f"Wrote {len(result['corpus'])} evaluated corpus entries to {output}")


@run_cli.command("render")
@click.option("--text", "-t", required=True, help="Text string to render.")
@click.option(
    "--cjk-font",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="CJK font file.",
)
@click.option(
    "--lgc-font",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Latin, Greek, and Cyrillic font file.",
)
@click.option(
    "--symbol-font",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="Symbol font file. (Optional)",
)
@click.option(
    "--emoji-font",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="Emoji font file. (Optional)",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Output image path.",
)
@click.option(
    "--size",
    "size_px",
    type=int,
    default=128,
    show_default=True,
    help="Font size in pixels.",
)
@click.option(
    "--fg-color",
    default="black",
    show_default=True,
    help="Foreground color (CSS name or hex).",
)
@click.option(
    "--bg-color",
    default="white",
    show_default=True,
    help="Background color (CSS name or hex).",
)
@click.option(
    "--align",
    type=click.Choice(["left", "center", "right"]),
    default="center",
    show_default=True,
    help="Text alignment.",
)
@click.option(
    "--dpi",
    type=int,
    default=300,
    show_default=True,
    help="Output resolution in dots per inch.",
)
@click.option(
    "--pad",
    type=int,
    default=24,
    show_default=True,
    help="Padding around text in pixels.",
)
def render_command(
    text: str,
    cjk_font: Path,
    lgc_font: Path,
    symbol_font: Path | None,
    emoji_font: Path | None,
    output_path: Path | None,
    size_px: int,
    fg_color: str,
    bg_color: str,
    align: str,
    dpi: int,
    pad: int,
) -> None:
    from sinoglyph.render import TextRenderConfig, TextRenderer

    if output_path is None:
        with NamedTemporaryFile(suffix=".png", prefix="render_", delete=False) as f:
            output_path = Path(f.name)
    try:
        config = TextRenderConfig(
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
        image = TextRenderer(text, config).render(output_path)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Rendered text to {output_path}")
    click.echo(f"Image size: {image.width}x{image.height} px")


@run_cli.command("chat")
@click.option(
    "--base-url",
    default=lambda: _env_default("LLM_BASE_URL"),
)
@click.option(
    "--api-key",
    default=lambda: _env_default("LLM_API_KEY"),
)
@click.option(
    "--model",
    default=lambda: _env_default("LLM_MODEL"),
)
@click.option(
    "--system",
    default=None,
    help="System prompt.",
)
@click.option(
    "--message",
    required=True,
    help="User prompt.",
)
@click.option(
    "--image",
    "images",
    multiple=True,
    help="Image path, URL, or data URL.",
)
@click.option(
    "--file",
    "files",
    multiple=True,
    help="Text or binary file to attach.",
)
@click.option(
    "--timeout",
    type=float,
    default=None,
    help="Request timeout.",
)
@click.option(
    "--max-retries",
    type=int,
    default=None,
    help="Retry failed API requests this many times.",
)
@click.option(
    "--json-schema",
    "json_schema_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
)
@click.option(
    "--param",
    "params",
    multiple=True,
    help="Extra request option as KEY=JSON_VALUE, for example --param temperature=0.",
)
def chat_command(
    base_url: str,
    api_key: str,
    model: str,
    system: str | None,
    message: str,
    images: tuple[str, ...],
    files: tuple[str, ...],
    timeout: float | None,
    max_retries: int | None,
    json_schema_path: Path | None,
    params: tuple[str, ...],
) -> None:
    from sinoglyph.llm import ChatClient

    load_env_file()
    schema = None
    if json_schema_path is not None:
        schema = json.loads(json_schema_path.read_text(encoding="utf-8"))
    try:
        client = ChatClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
            system=system,
            timeout=timeout,
            max_retries=max_retries,
            **_params(params),
        )
        reply = client.chat(
            text=message,
            images=list(images),
            files=list(files),
            json_schema=schema,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        json.dumps(reply, ensure_ascii=False, indent=2)
        if isinstance(reply, (dict, list))
        else reply
    )


def _parse_param(value: str) -> tuple[str, object]:
    if "=" not in value:
        raise click.BadParameter("expected KEY=VALUE")
    key, raw = value.split("=", 1)
    key = key.strip()
    if not key:
        raise click.BadParameter("parameter key cannot be empty")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = raw
    return key, parsed


def _params(values: tuple[str, ...]) -> dict[str, object]:
    return dict(_parse_param(value) for value in values)


def _display_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _env_default(name: str) -> str:
    load_env_file()
    return os.getenv(name, "")


if __name__ == "__main__":
    run_cli()

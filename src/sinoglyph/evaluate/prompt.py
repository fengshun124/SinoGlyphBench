from sinoglyph.schema.base import JsonObject
from sinoglyph.schema.utils import require_list, require_mapping


def build_text_message(prompt: str, input_text: str) -> str:
    if "{input_text}" in prompt:
        return prompt.replace("{input_text}", input_text).rstrip()
    return f"{prompt.rstrip()}\n\n<INPUT_TEXT>\n{input_text}\n</INPUT_TEXT>"


def build_image_message(prompt: str) -> str:
    return prompt.rstrip()


def append_response_contract(prompt: str, response_contract: str) -> str:
    return f"{prompt.rstrip()}\n\n{response_contract}"


def build_response_contract(response_schema: JsonObject) -> str:
    properties = require_mapping(
        response_schema.get("properties", {}), "response.schema.properties"
    )
    required_fields = [
        item
        for item in require_list(
            response_schema.get("required", []), "response.schema.required"
        )
        if isinstance(item, str)
    ]
    lines = [
        "Output contract:",
        "Return exactly one JSON object and no markdown.",
        "The JSON object must match this schema:",
    ]
    for field in required_fields:
        field_schema = properties.get(field, {})
        enum = None
        if isinstance(field_schema, dict) and isinstance(
            field_schema.get("enum"), list
        ):
            enum = ", ".join(str(value) for value in field_schema["enum"])
        enum_text = f" one of: {enum}." if enum else ""
        lines.append(f"- {field}: string.{enum_text}")
    if response_schema.get("additionalProperties") is False:
        lines.append("Do not include any other fields.")
    return "\n".join(lines)

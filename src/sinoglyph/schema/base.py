from typing import TypeAlias

JsonValue: TypeAlias = object
JsonObject: TypeAlias = dict[str, JsonValue]
JsonArray: TypeAlias = list[JsonValue]

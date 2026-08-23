import pytest

from ai_chat_util.core.browser.browser_task import _build_output_model_from_schema


def test_array_object_items_type_is_preserved() -> None:
    schema = {
        "properties": {
            "articles": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                    },
                },
            }
        },
        "required": ["articles"],
    }

    output_model = _build_output_model_from_schema(schema)
    model_schema = output_model.model_json_schema()

    article_schema = model_schema["properties"]["articles"]
    assert article_schema["type"] == "array"
    assert article_schema["items"]["type"] == "object"


def test_array_string_items_type_is_preserved() -> None:
    schema = {
        "properties": {
            "articles": {
                "type": "array",
                "items": {
                    "type": "string",
                },
            }
        },
        "required": ["articles"],
    }

    output_model = _build_output_model_from_schema(schema)
    model_schema = output_model.model_json_schema()

    article_schema = model_schema["properties"]["articles"]
    assert article_schema["type"] == "array"
    assert article_schema["items"]["type"] == "string"


def test_array_without_items_type_raises() -> None:
    schema = {
        "properties": {
            "articles": {
                "type": "array",
            }
        }
    }

    with pytest.raises(ValueError, match="items.type"):
        _build_output_model_from_schema(schema)

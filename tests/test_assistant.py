import pytest

from src.house_price_assistant import HousePriceAssistant


def test_parse_json_response_accepts_json_code_fence():
    result = HousePriceAssistant.parse_json_response(
        '```json\n{"features": {"Gr Liv Area": 1800}}\n```'
    )

    assert result == {"features": {"Gr Liv Area": 1800}}


def test_parse_json_response_rejects_invalid_json():
    with pytest.raises(ValueError):
        HousePriceAssistant.parse_json_response(
            "This is not valid JSON"
        )


def test_extract_features_rejects_unknown_feature(monkeypatch):
    assistant = HousePriceAssistant()

    monkeypatch.setattr(
        assistant,
        "_call_llm",
        lambda messages: (
            '{"features": {"Imaginary Feature": 123}, '
            '"missing_details": []}'
        ),
    )

    with pytest.raises(ValueError):
        assistant.extract_features("A house with an imaginary feature.")


def test_respond_asks_for_missing_required_details(monkeypatch):
    assistant = HousePriceAssistant()

    monkeypatch.setattr(
        assistant,
        "extract_features",
        lambda text: {
            "features": {"Gr Liv Area": 1800},
            "missing_required": ["Overall Qual"],
            "missing_details": [],
        },
    )

    result = assistant.respond("Estimate a house price.")

    assert isinstance(result, dict)
    assert result.get("status") != "success"
    assert "overall material/finish quality" in result["message"].lower()


def test_extract_features_accepts_valid_natural_language_request(monkeypatch):
    assistant = HousePriceAssistant()

    monkeypatch.setattr(
        assistant,
        "_call_llm",
        lambda messages: (
            '{"features": {'
            '"Gr Liv Area": 1800, '
            '"Overall Qual": 7'
            '}, "missing_details": []}'
        ),
    )

    result = assistant.extract_features(
        "Estimate a house with 1,800 square feet of living area "
        "and an overall quality rating of 7."
    )

    assert result["features"]["Gr Liv Area"] == 1800
    assert result["features"]["Overall Qual"] == 7
    assert result["missing_required"] == []

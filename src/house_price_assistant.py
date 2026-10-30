import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

MODEL_PATH = ROOT / "models" / "best_model.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"

REQUIRED_FEATURES = ["Gr Liv Area", "Overall Qual"]


class HousePriceAssistant:
    def __init__(self, model_path=MODEL_PATH, metadata_path=METADATA_PATH):
        self.model = joblib.load(model_path)

        with Path(metadata_path).open("r", encoding="utf-8") as file:
            self.metadata = json.load(file)

        self.feature_columns = self.metadata["feature_columns"]
        preprocessor = self.model.named_steps["preprocessor"]

        self.numeric_columns = set()
        for name, _, columns in preprocessor.transformers_:
            if name == "numeric":
                self.numeric_columns.update(columns)

        self.allowed_features = set(self.feature_columns)


    def _call_llm(self, messages):
        api_key = os.getenv("GEMINI_API_KEY")
        base_url = os.getenv(
            "GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta",
        ).rstrip("/")
        model_name = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is missing. Add your Google AI Studio key to .env."
            )

        system_messages = [
            item["content"]
            for item in messages
            if item.get("role") == "system"
        ]
        input_messages = [
            item["content"]
            for item in messages
            if item.get("role") != "system"
        ]

        payload = {
            "model": model_name,
            "input": "\n\n".join(input_messages),
            "store": False,
            "generation_config": {
                "temperature": 0,
                "thinking_level": "minimal",
            },
        }

        if system_messages:
            payload["system_instruction"] = "\n\n".join(system_messages)

        response = requests.post(
            f"{base_url}/interactions",
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=(10, 90),
        )
        response.raise_for_status()
        result = response.json()

        # Some responses provide output_text directly.
        output_text = result.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        # Native Interactions API text can be inside model_output steps.
        text_parts = []
        for step in result.get("steps", []):
            if not isinstance(step, dict):
                continue
            if step.get("type") != "model_output":
                continue

            for item in step.get("content", []):
                if (
                    isinstance(item, dict)
                    and item.get("type") == "text"
                    and isinstance(item.get("text"), str)
                ):
                    text_parts.append(item["text"])

        if text_parts:
            return "\n".join(text_parts).strip()

        raise ValueError(
            "Gemini returned no text output. "
            f"Interaction status: {result.get('status', 'unknown')}"
        )
    @staticmethod
    def parse_json_response(content):
        text = content.strip()

        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            result = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("Gemini did not return valid JSON.") from exc

        if not isinstance(result, dict):
            raise ValueError("The LLM response must be a JSON object.")

        return result

    def extract_features(self, user_text):
        if not isinstance(user_text, str) or not user_text.strip():
            raise ValueError("Please describe the property before continuing.")

        feature_list = json.dumps(self.feature_columns, ensure_ascii=False)

        system_prompt = f"""
You extract property facts for a historical Ames, Iowa house-price model.

Return ONLY a valid JSON object with this structure:
{{
  "features": {{"exact dataset column name": value}},
  "missing_details": ["short description"]
}}

Allowed feature names:
{feature_list}

Rules:
- Only use exact feature names from the allowed list.
- Extract only facts explicitly stated by the user.
- Never invent or guess property details.
- Use JSON numbers for numeric values and strings for categories.
- Omit unknown values instead of guessing.
- If a value is ambiguous, omit it and describe the ambiguity.
- Do not include SalePrice, Order, or PID.
- Gr Liv Area means above-ground living area in square feet.
- Overall Qual is a quality rating from 1 to 10.
- Overall Cond is a condition rating from 1 to 10.
- Do not answer questions unrelated to describing a property.
- Return empty features if the user did not describe a property.
"""

        raw = self._call_llm([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ])

        result = self.parse_json_response(raw)
        features = result.get("features")

        if not isinstance(features, dict):
            raise ValueError("The LLM response is missing a features object.")

        unknown = set(features) - self.allowed_features
        if unknown:
            raise ValueError(
                f"The LLM returned unsupported features: {sorted(unknown)}"
            )

        cleaned = {}

        for name, value in features.items():
            if value is None or value == "":
                continue

            if name in self.numeric_columns:
                try:
                    number = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Feature '{name}' must have a numeric value."
                    ) from exc

                if not np.isfinite(number):
                    raise ValueError(
                        f"Feature '{name}' must be a finite number."
                    )

                if name in {
                    "Gr Liv Area", "Lot Area", "Total Bsmt SF",
                    "Lot Frontage", "Garage Area", "Garage Cars",
                } and number < 0:
                    raise ValueError(f"Feature '{name}' cannot be negative.")

                if name in {"Overall Qual", "Overall Cond"}:
                    if not number.is_integer() or not 1 <= number <= 10:
                        raise ValueError(
                            f"Feature '{name}' must be an integer from 1 to 10."
                        )

                cleaned[name] = int(number) if number.is_integer() else number
            else:
                if not isinstance(value, str):
                    raise ValueError(
                        f"Categorical feature '{name}' must be a string."
                    )
                cleaned[name] = value.strip()

        missing_required = [
            name for name in REQUIRED_FEATURES if name not in cleaned
        ]

        details = result.get("missing_details", [])
        if not isinstance(details, list):
            details = []

        return {
            "features": cleaned,
            "missing_required": missing_required,
            "missing_details": details,
        }

    def predict(self, features):
        unknown = set(features) - self.allowed_features
        if unknown:
            raise ValueError(f"Unsupported features: {sorted(unknown)}")

        row = {column: np.nan for column in self.feature_columns}
        row.update(features)
        input_frame = pd.DataFrame([row], columns=self.feature_columns)

        prediction = float(self.model.predict(input_frame)[0])

        if not np.isfinite(prediction) or prediction <= 0:
            raise ValueError("The model returned an invalid price prediction.")

        return prediction

    def explain_prediction(self, user_text, features, prediction):
        prompt = {
            "original_property_description": user_text,
            "extracted_features": features,
            "model_prediction_usd": round(prediction, 2),
            "test_metrics": self.metadata["test_metrics"],
            "selected_model": self.metadata["model_name"],
            "instructions": [
                "Explain the estimate in plain English.",
                "Do not change or recalculate the supplied prediction.",
                "Identify the supplied property features used by the model.",
                "Explain that omitted features were imputed by the trained pipeline.",
                "Mention the test MAE and explain its meaning.",
                "The data represents historical Ames, Iowa sales.",
                "Do not call this a current market valuation or appraisal.",
                "Do not claim certainty or causal explanations.",
                "Keep the explanation concise.",
            ],
        }

        return self._call_llm([
            {
                "role": "system",
                "content": (
                    "Explain machine-learning house-price estimates accurately. "
                    "Never invent property facts or change the model prediction."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]).strip()

    def respond(self, user_text):
        extracted = self.extract_features(user_text)

        if extracted["missing_required"]:
            questions = {
                "Gr Liv Area": (
                    "What is the home's above-ground living area in square feet?"
                ),
                "Overall Qual": (
                    "What is the home's overall material/finish quality "
                    "on a scale from 1 to 10?"
                ),
            }
            missing = extracted["missing_required"]
            message = "I need a little more detail before estimating the price:\n"
            message += "\n".join(f"- {questions[name]}" for name in missing)

            if extracted["missing_details"]:
                message += "\nDetails to clarify: " + "; ".join(
                    str(item) for item in extracted["missing_details"]
                )

            return {
                "status": "needs_clarification",
                "message": message,
                "features": extracted["features"],
            }

        prediction = self.predict(extracted["features"])
        explanation = self.explain_prediction(
            user_text, extracted["features"], prediction
        )

        return {
            "status": "success",
            "prediction_usd": round(prediction, 2),
            "features": extracted["features"],
            "explanation": explanation,
        }


def main():
    print("=" * 60)
    print("AMES HOUSING PRICE PREDICTION ASSISTANT")
    print("Historical estimates only - not a current appraisal.")
    print("Type 'quit' to exit.")
    print("=" * 60)

    assistant = HousePriceAssistant()

    while True:
        try:
            user_text = input("\nDescribe a house: ").strip()

            if user_text.lower() in {"quit", "exit"}:
                print("Goodbye!")
                break

            result = assistant.respond(user_text)

            if result["status"] == "needs_clarification":
                print("\n" + result["message"])
            else:
                print(
                    f"\nEstimated historical sale price: "
                    f"${result['prediction_usd']:,.0f}"
                )
                print("\n" + result["explanation"])

        except (ValueError, RuntimeError, requests.RequestException) as exc:
            print(f"\nUnable to complete the estimate: {exc}")


if __name__ == "__main__":
    main()

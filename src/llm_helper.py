import os
from typing import Any, Iterable, List

from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

load_dotenv()


def _compact_waveform(sample: Iterable[float], max_points: int = 24) -> List[float]:
    """Downsample a waveform to max_points for compact display in the LLM prompt."""
    values = [float(v) for v in sample]
    if len(values) <= max_points:
        return values
    indices = [int(round(i * (len(values) - 1) / (max_points - 1))) for i in range(max_points)]
    return [values[idx] for idx in indices]


def build_llm_context(model_overview: dict[str, Any], prototype_rows: list[dict[str, Any]], user_question: str) -> str:
    """Build a structured prompt context from model overview, prototype rows, and user question."""
    model_name = model_overview.get("model_name", "Unknown model")
    train_score = model_overview.get("train_score")
    validation_score = model_overview.get("validation_score")
    selection_mode = model_overview.get("selection_mode", "central")
    prototype_count = model_overview.get("prototype_count", len(prototype_rows))

    def fmt_score(score: Any) -> str:
        if isinstance(score, (int, float)):
            return f"{score:.2%}"
        return "n/a"

    correct_count = sum(1 for row in prototype_rows if row.get("is_correct"))
    row_descriptions: list[str] = []
    for row in prototype_rows:
        sample = row.get("sample", [])
        compact_sample = _compact_waveform(sample)
        row_descriptions.append(
            f"Prototype #{row.get('rank', '?')}: index={row.get('index', '?')}, "
            f"true={row.get('true_text', '?')}, pred={row.get('predicted_text', '?')}, "
            f"outcome={'correct' if row.get('is_correct') else 'incorrect'}, waveform={compact_sample}"
        )

    prototype_block = "\n".join(row_descriptions) if row_descriptions else "No prototype rows are available at the moment."

    return (
        "You are an explanation assistant for a time-series classifier dashboard.\n"
        "Focus on model behavior, prototype evidence, and prediction reasoning.\n"
        "Do not give a diagnosis or treatment recommendation.\n"
        f"Model name: {model_name}\n"
        f"Train accuracy: {fmt_score(train_score)}\n"
        f"Validation accuracy: {fmt_score(validation_score)}\n"
        f"Prototype selection mode: {selection_mode}\n"
        f"Prototype count: {prototype_count}\n"
        f"Correct prototypes: {correct_count} / {len(prototype_rows)}\n"
        f"Prototype evidence:\n{prototype_block}\n"
        f"User question: {user_question}\n"
        "Important: this is an LLM-generated explanation only and may be incorrect.\n"
        "Answer in plain language and keep the explanation relevant to the prototype evidence shown above.\n"
    )


def _sanitize_output(raw_text: str) -> str:
    """Clean up potentially harmful terms in the LLM response."""
    if not raw_text:
        return "The LLM did not return any usable text."
    cleaned = raw_text.strip()
    cleaned = cleaned.replace("I diagnose the patient as", "The waveform pattern may suggest")
    cleaned = cleaned.replace("I would diagnose", "The prototype pattern may suggest")
    cleaned = cleaned.replace("I am not a doctor", "This explanation is an LLM explanation and may be wrong")
    return cleaned


def ask_llm_about_prototype(
    model_overview: dict[str, Any],
    prototype_rows: list[dict[str, Any]],
    user_question: str,
    conversation: str = "",
    api_key: str | None = None,
) -> str:
    """Send a question together with the previous conversation."""
    resolved_key = api_key or os.getenv("API_KEY")
    conversation = conversation or []

    prompt = build_llm_context(model_overview, prototype_rows, user_question)

    if not resolved_key or OpenAI is None:
        return (
            "LLM explanation unavailable because no API key is configured. "
            f"The current model is {model_overview.get('model_name', 'Unknown model')}."
        )

    client = OpenAI(api_key=resolved_key)

    messages = [
        {
            "role": "system",
            "content": (
                    "You are a careful explanation assistant for a time-series prototype dashboard. "
                    "Explain the model behavior using the provided prototype evidence, stay relevant to the selected samples, "
                    "and explicitly note that the answer is an LLM explanation and may be wrong. Do not provide a diagnosis or treatment recommendation."
            )
        }
    ]

    messages.extend(
        message for message in conversation if message.get("role") in {"user", "assistant"}
    )

    messages.append({
        "role": "user",
        "content": prompt,
    })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.6,
    )

    raw_text = response.choices[0].message.content or ""
    return _sanitize_output(raw_text)
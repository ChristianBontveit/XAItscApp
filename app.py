from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html, ctx, no_update

from src.load_dataset import load_ecg200_data
from src.llm_helper import ask_llm_about_prototype
from src.tsc_model import (
    accuracy_score,
    binary_label_to_text,
    build_prediction_rows,
    load_prediction_artifact,
    load_rocket_model,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET = load_ecg200_data(PROJECT_ROOT / "data" / "ECG200")
MODEL_PATH = PROJECT_ROOT / "models" / "ECG200" / "miniRocket_norm.pkl"
MODEL = load_rocket_model(MODEL_PATH)
VALIDATION_PATH = PROJECT_ROOT / "data" / "ECG200" / "ECG200_VALIDATION_normalized.npy"
PREDICTION_ARTIFACT = PROJECT_ROOT / "data" / "ECG200" / "validation_predictions.npz"
VALIDATION_PREDICTIONS = load_prediction_artifact(
    PREDICTION_ARTIFACT,
    MODEL_PATH,
    VALIDATION_PATH,
    len(DATASET["validation"]["samples"]),
)
PREDICTION_ROWS = build_prediction_rows(
    DATASET["validation"]["samples"],
    DATASET["validation"]["labels"],
    VALIDATION_PREDICTIONS,
)

MODEL_NAME = MODEL_PATH.stem

TRAIN_SCORE = accuracy_score(MODEL, DATASET["train"]["samples"], DATASET["train"]["labels"])
VALIDATION_SCORE = float(
    np.mean(VALIDATION_PREDICTIONS == DATASET["validation"]["labels"])
)

def serializable_rows(rows):
    """Convert prediction rows to the format used by Dash browser state."""
    cleaned = []
    for row in rows:
        clean_row = dict(row)
        clean_row["sample"] = np.asarray(row["sample"], dtype=float).tolist()
        cleaned.append(clean_row)
    return cleaned


def filter_prediction_rows(filter_name):
    if filter_name == "disagreement":
        return [row for row in PREDICTION_ROWS if not row["is_correct"]]
    if filter_name == "correct-normal":
        return [row for row in PREDICTION_ROWS if row["is_correct"] and row["true_label"] == 1]
    if filter_name == "correct-abnormal":
        return [row for row in PREDICTION_ROWS if row["is_correct"] and row["true_label"] == 0]
    return PREDICTION_ROWS


def example_label(row):
    return (
        f"ECG200 #{row['index'] + 1} - "
        f"recorded {binary_label_to_text(row['true_label'])}, "
        f"predicted {binary_label_to_text(row['predicted_label'])}"
    )


def build_prototype_cards(rows):
    """Build expandable card components from prototype rows."""
    cards = []
    for row in rows:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                y=row["sample"],
                mode="lines",
                line={"color": "#1f77b4", "width": 2},
                name="Time series",
            )
        )
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=20, r=20, t=50, b=20),
            height=260,
            title=f"ECG200 #{row['index'] + 1}"
        )

        status_color = "success" if row["is_correct"] else "danger"
        status_text = "Correct" if row["is_correct"] else "Incorrect"

        cards.append(
            html.Details(
                [
                    html.Summary(
                        [
                            html.Div(
                                [
                                    html.Div(f"ECG200 #{row['index'] + 1}", className="fw-bold"),
                                    html.Div(
                                        [
                                            html.Span(
                                                f"Recorded: {binary_label_to_text(row['true_label'])}",
                                                className="me-3",
                                            ),
                                            html.Span(
                                                f"Predicted: {binary_label_to_text(row['predicted_label'])}",
                                                className="me-3",
                                            ),
                                            dbc.Badge(status_text, color=status_color, className="ms-2"),
                                        ],
                                        className="text-muted small",
                                    ),
                                ],
                                className="d-flex flex-column",
                            )
                        ],
                        style={"cursor": "pointer", "padding": "12px 16px", "borderBottom": "1px solid #e9ecef"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.P(f"Validation index: {row['index']}", className="mb-2 text-muted"),
                                    html.P(f"Sample length: {len(row['sample'])} time points", className="mb-3 text-muted"),
                                ]
                            ),
                            dcc.Graph(figure=fig, config={"displayModeBar": False}),
                        ],
                        style={"padding": "16px"},
                    ),
                ],
                open=False,
                style={"border": "1px solid #e9ecef", "borderRadius": "10px", "marginBottom": "12px", "backgroundColor": "#fff"},
            )
        )

    return cards

def render_chat_history(history):
    if not history:
        return html.P(
            className="mb-0 text-muted",
        )

    messages = []

    for message in history:
        is_user = message["role"] == "user"

        messages.append(
            html.Div(
                [
                    html.Div(
                        "You" if is_user else "LLM",
                        className="fw-bold small text-muted mb-1",                        
                    ),
                    html.Div(
                    message["content"],
                    style={"whiteSpace": "pre-wrap"},
                    ),
                ],
                style={
                    "backgroundColor": "#e8f1fb" if is_user else "#ffffff",
                    "border": "1px solid #dee2e6",
                    "borderRadius": "8px",
                    "padding": "10px 12px",
                    "marginBottom": "10px",
                },
            )
        )
    return messages


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(html.H2("LLM Companion for Time Series Classification", className="text-primary my-3"), width=12),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("Model overview", className="card-title mb-3"),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody(
                                                    [
                                                        html.H6("Model", className="text-muted"),
                                                        html.H3(MODEL_NAME, className="mb-0"),
                                                    ]
                                                ),
                                                className="h-100 border-0 shadow-sm",
                                        ),
                                        width=4,
                                        ),
                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody(
                                                    [
                                                        html.H6("Train accuracy", className="text-muted"),
                                                        html.H3(f"{TRAIN_SCORE:.2%}", className="mb-0"),
                                                    ]
                                                ),
                                                className="h-100 border-0 shadow-sm",
                                            ),
                                            width=4,
                                        ),
                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody(
                                                    [
                                                        html.H6("Validation accuracy", className="text-muted"),
                                                        html.H3(f"{VALIDATION_SCORE:.2%}", className="mb-0"),
                                                    ]
                                                ),
                                                className="h-100 border-0 shadow-sm",
                                            ),
                                            width=4,
                                        ),
                                    ],
                                    className="g-3",
                                ),
                            ]
                        )
                    ),
                    width=12,
                )
            ],
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div(
                                    [
                                        html.H4("ECG200 example explorer", className="card-title mb-0"),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dcc.Dropdown(
                                                        id="example-filter",
                                                        options=[
                                                            {"label": "All examples", "value": "all"},
                                                            {"label": "Model/label disagreement", "value": "disagreement"},
                                                            {"label": "Correct normal examples", "value": "correct-normal"},
                                                            {"label": "Correct abnormal examples", "value": "correct-abnormal"},
                                                        ],
                                                        value="all",
                                                        clearable=False,
                                                        placeholder="Filter examples",
                                                        searchable=False,
                                                    ),
                                                    md=5,
                                                ),
                                                dbc.Col(
                                                    dcc.Dropdown(
                                                        id="example-selection",
                                                        options=[
                                                            {"label": example_label(row), "value": row["index"]}
                                                            for row in PREDICTION_ROWS
                                                        ],
                                                        value=0,
                                                        clearable=False,
                                                        placeholder="Select an ECG example",
                                                        searchable=True,
                                                        maxHeight=400,
                                                    ),
                                                    md=7,
                                                ),
                                            ],
                                            className="g-2 mt-2",
                                        ),
                                    ],
                                    style={
                                        "position": "sticky",
                                        "top": "0",
                                        "zIndex": 2,
                                        "backgroundColor": "#fff",
                                        "paddingBottom": "10px",
                                        "borderBottom": "1px solid #e9ecef",
                                        "marginBottom": "12px",
                                    },
                                ),
                                dcc.Loading(
                                    html.Div(
                                        id="prototype-list",
                                        children=build_prototype_cards(serializable_rows([PREDICTION_ROWS[0]])),
                                        style={
                                            "height": "100%",
                                            "overflowY": "auto",
                                            "paddingRight": "4px",
                                        },
                                    ),
                                    type="circle",
                                    color="#0d6efd",
                                    fullscreen=False,
                                    parent_style={
                                        "position": "relative",
                                        "height": "100%",
                                        "flex": "1",
                                        "overflow": "hidden",
                                    },
                                    style={
                                        "position": "absolute",
                                        "top": "6px",
                                        "left": "50%",
                                        "transform": "translateX(-50%)",
                                        "zIndex": 3,
                                    },
                                ),
                                html.Div(
                                id="prototype-status",
                                children="Ready",
                                style={"fontSize": "0.9rem", "color": "#6c757d", "marginBottom": "10px"},
                                )
                            ],
                            style={
                                "display": "flex",
                                "flexDirection": "column",
                                "height": "100%",
                                "minHeight": "620px",
                            },
                        ),
                        className="h-100 border-0 shadow-sm",
                    ),
                    width=8,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("LLM explanation", className="card-title mb-3"),
                                html.Div(
                                    "The selected ECG example and model metrics are sent to the LLM so it can explain the classifier behavior.",
                                    className="text-muted small mb-3",
                                ),
                                dcc.Loading(
                                    html.Div(
                                        id="llm-output",
                                        children=render_chat_history([]),
                                        style={
                                            "maxHeight": "520px",
                                            "overflowY": "auto",
                                            "padding": "4px",
                                        },
                                    ),
                                    type="circle",
                                    color="#0d6efd",
                                ),
                                dbc.Textarea(
                                    id="llm-question",
                                    placeholder="Ask a question about the selected ECG example or time series model",
                                    value="",
                                    rows=3,
                                    className="mb-3",
                                ),
                                dbc.Button(
                                    "Ask LLM",
                                    id="llm-submit",
                                    color="primary",
                                    className="w-100 mb-3",
                                ),
                                dbc.Button(
                                    "Clear conversation",
                                    id="llm-clear",
                                    color="secondary",
                                    outline=True,
                                    className="w-100 mb-3",
                                ),
                                dbc.Alert(
                                    "This is an LLM-generated explanation and may be incorrect. It is not a medical diagnosis. Powered by gpt-4o-mini.",
                                    color="warning",
                                    className="mt-3 mb-0",
                                ),
                            ]
                        ),
                        className="h-100 border-0 shadow-sm",
                        style={
                            "backgroundColor": "#f8f9fa",
                            "border": "1px solid #e9ecef",
                            "borderRadius": "12px",
                            "position": "sticky",
                            "top": "20px",
                            "height": "100%",
                        },
                    ),
                    width=4,
                ),
            ],
            className="g-3 mb-4 align-items-stretch",
        ),
        dcc.Store(id="prototype-data", data=serializable_rows([PREDICTION_ROWS[0]])),
        dcc.Store(id="llm-history", data=[]),
    ],
    fluid=True,
)


@app.callback(
    Output("example-selection", "options"),
    Output("example-selection", "value"),
    Output("prototype-list", "children"),
    Output("prototype-status", "children"),
    Output("prototype-data", "data"),
    Input("example-filter", "value"),
    Input("example-selection", "value"),
)
def update_examples(filter_name, selected_index):
    rows = filter_prediction_rows(filter_name or "all")
    options = [
        {"label": example_label(row), "value": row["index"]}
        for row in rows
    ]
    if not rows:
        return options, None, html.P("No examples match this filter.", className="text-muted"), "No matching examples", []

    valid_indices = {row["index"] for row in rows}
    if ctx.triggered_id == "example-filter" or selected_index not in valid_indices:
        selected_index = rows[0]["index"]

    selected_row = next(row for row in rows if row["index"] == selected_index)
    selected_data = serializable_rows([selected_row])
    return (
        options,
        selected_index,
        build_prototype_cards(selected_data),
        f"Showing {len(rows)} matching example(s)",
        selected_data,
    )


@app.callback(
    Output("llm-output", "children"),
    Output("llm-history", "data"),
    Output("llm-question", "value"),
    Input("llm-submit", "n_clicks"),
    State("llm-question", "value"),
    State("prototype-data", "data"),
    State("example-filter", "value"),
    State("llm-history", "data"),
    Input("llm-clear", "n_clicks"),
    prevent_initial_call=True,
)

def explain_prototypes(_submit_click, user_question, prototype_data, example_filter, history, _clear_clicks):
    if ctx.triggered_id == "llm-clear":
        return render_chat_history([]), [], ""

    history = history or []

    if not user_question or not user_question.strip():
        return render_chat_history(history), history, no_update

    rows = prototype_data or serializable_rows([PREDICTION_ROWS[0]])
    model_context = {
        "model_name": MODEL_NAME,
        "train_score": TRAIN_SCORE,
        "validation_score": VALIDATION_SCORE,
        "selection_mode": example_filter or "all",
        "prototype_count": len(rows),
    }

    answer = ask_llm_about_prototype(
        model_context,
        rows,
        user_question.strip(),
        conversation=history,
    )

    updated_history = history + [
        {
            "role": "user",
            "content": user_question.strip(),
        },
        {
            "role": "assistant",
            "content": answer,
        },
    ]

    return render_chat_history(updated_history), updated_history, ""


if __name__ == "__main__":
    app.run(debug=True)
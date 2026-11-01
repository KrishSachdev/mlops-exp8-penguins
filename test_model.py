# test_model.py - simple checks that run in GitHub Actions
import joblib
import pandas as pd

saved = joblib.load("model.pkl")


def make_row(island, bill_length, bill_depth, flipper_length, body_mass, sex):
    row = pd.DataFrame([{
        "island": island, "bill_length_mm": bill_length,
        "bill_depth_mm": bill_depth, "flipper_length_mm": flipper_length,
        "body_mass_g": body_mass, "sex": sex,
    }])
    row = pd.get_dummies(row, columns=["island", "sex"])
    return row.reindex(columns=saved["columns"], fill_value=0)


def test_model_file_has_parts():
    assert "model" in saved
    assert "columns" in saved


def test_accuracy_is_good():
    assert saved["accuracy"] >= 0.90


def test_predicts_gentoo():
    # a large penguin from Biscoe with long flippers is a Gentoo
    row = make_row("Biscoe", 47.5, 15.0, 220, 5400, "male")
    assert saved["model"].predict(row)[0] == "Gentoo"


def test_predicts_adelie():
    # a small penguin from Torgersen with a short bill is an Adelie
    row = make_row("Torgersen", 38.0, 18.5, 188, 3600, "female")
    assert saved["model"].predict(row)[0] == "Adelie"

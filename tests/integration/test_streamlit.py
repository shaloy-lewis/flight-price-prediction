from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_predictor_and_input_error(trained_artifact, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(trained_artifact))
    script = Path(__file__).resolve().parents[2] / "app.py"
    app = AppTest.from_file(str(script), default_timeout=20).run()
    assert not app.exception
    app.sidebar.selectbox[0].select("Predictor").run()
    assert not app.exception
    assert app.button[0].label == "Estimate historical fare"
    app.button[0].click().run()
    assert not app.exception
    assert len(app.metric) == 1
    assert "INR" in app.metric[0].label
    # Same route endpoints produce a useful validation message, not an exception.
    next(box for box in app.selectbox if box.label == "From").select("Delhi")
    next(box for box in app.selectbox if box.label == "To").select("Delhi")
    app.button[0].click().run()
    assert not app.exception
    assert len(app.error) >= 1

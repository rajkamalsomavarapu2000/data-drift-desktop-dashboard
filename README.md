# Data Drift Desktop Dashboard (Offline)

A PySide6 desktop app that compares a baseline CSV vs a current CSV, ranks features by drift, and visualizes numeric/categorical changes.

## Run
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python -m app.main

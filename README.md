# Sentinal

A Python project for neuro-critical care risk prediction using Streamlit and PyTorch.

## Quick start

1. Clone the repository:

```bash
git clone <repo-url>
cd sentinal
```

2. Create a Python virtual environment:

```bash
python -m venv .venv
```

3. Activate the environment:

- Windows PowerShell:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- Windows CMD:
  ```cmd
  .venv\Scripts\activate.bat
  ```
- macOS / Linux:
  ```bash
  source .venv/bin/activate
  ```

4. Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

5. Run the Streamlit app:

```bash
streamlit run app.py
```

Open the URL shown by Streamlit in your browser.

## Notes

- `app.py` is the main app entrypoint.
- The repository ignores generated artifacts such as:
  - `data/processed/`
  - `models/`
  - `outputs/`
  - `.venv/`, `__pycache__/`, editor folders

If the cloned repo does not include trained model weights or processed data, you must generate them first.

## Generating data and models

If you need to rebuild artifacts locally, these scripts are available:

- `python src/preprocess.py`
- `python src/train.py`
- `python src/ps1_train.py`
- `python src/ps5_train.py`

After generating data and training models, ensure the output files are placed in:

- `data/processed/`
- `models/`

Then run the app as shown above.

## Recommended workflow

1. Install dependencies.
2. Generate processed data and model weights if not already present.
3. Start the app with `streamlit run app.py`.

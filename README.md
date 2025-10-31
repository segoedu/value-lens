<img width="3015" height="2689" alt="poster" src="https://github.com/user-attachments/assets/ca702052-8e8c-432c-8111-3f1277074fdf" />

# Using Large Language Models to Understand Human Values

Value Lens is a Streamlit app that analyzes any text through human value theories, detecting which values appear, how strongly they are expressed, and why—complete with evidence and explanations. It supports classic frameworks (Schwartz, Rokeach, Maslow) and custom taxonomies, using a two‑agent setup where one model detects and another critiques for greater reliability.

## Highlights ✨
- Multi‑agent detection + critique for robust, transparent results.
- Works with major value theories or custom JSON frameworks.
- Intensity scoring with clear color bars and evidence snippets.
- Analysis history with chosen model, settings, and outputs.

## Demo ▶️
- 3‑minute walkthrough of theory selection, analysis, intensity, and justifications.
- [Video demonstration](https://www.youtube.com/watch?v=TevkbV_W9Ls)

## Quick start 🚀
1) Clone and enter the repo:
- git clone https://github.com/segoedu/value-lens && cd value-lens

2) Create env and install:
- python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
- pip install -r requirements.txt  or install: streamlit, python-dotenv, pandas, langchain, langchain-openai, langchain-groq, supabase, streamlit-option-menu, annotated-text.

3) Configure .env:
- OPENROUTER_API_KEY, GROQ_API_KEY, SUPABASE_URL, SUPABASE_KEY (optional for history).

4) Run:
- streamlit run app.py  (or the main app file in the repo)

## App structure 🧭
- Tabs: Text Analysis, Value Theories, Datasets, History, Paper, Settings.
- Theories: loaded from JSON; metadata + values are browsable.
- Datasets: lightweight CSV format with text, values, intensity (+ optional model).
- Paper: renders valuelens.md if present.

## How to use 🧪
- Choose a value theory and paste text.
- Step 1: detect values present.
- Step 2: score intensity (e.g., strong_resistance → strong_support, reframing, no_values) with color bar.
- Step 3: merge and display results with evidence and justifications; compare with optional manual tags.

## Data and formats 📁
- Theories: JSON files with metadata and value definitions.
- Datasets (CSV):
  - Line 1: dataset name
  - Line 2: description
  - Line 3: headers include text, values, intensity, optional model
  - Remaining: rows with values separated by commas

## History and storage 💾
- Optional Supabase logging of theory, text, result JSON, and model to table valuelens.
- Requires SUPABASE_URL and SUPABASE_KEY; create columns accordingly.

## Models ⚙️
- Providers: Groq or OpenRouter; pick model in Settings.
- Fallback to a default if selection is invalid.

## Tips ✅
- Lower temperature for reproducibility.
- Keep custom theory JSONs clean with clear definitions and tags.
- Validate dataset columns; UI warns if required fields are missing.

## License 📄
- MIT License.
- 🌐 Eduardo de la Cruz <a href="https://www.linkedin.com/in/eduardodelacruzf/" target="_blank">Linkedin profile</a>

<img width="493" height="81" alt="25693dde6b8099a858f03866e6af39473ddb6c3f90bb1a82e982837d" src="https://github.com/user-attachments/assets/53baf9d1-c76a-4cd9-b40a-e225ad786f82" />



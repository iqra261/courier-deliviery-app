# Waybill — courier office desk

**AI for Everyone · Project 05 · Courier Delivery Planning and Feedback Assistant**

Waybill is a small Streamlit app for a courier office reviewing demonstration delivery jobs. The office worker enters a job, gets an estimated delivery duration and a delivery profile, checks a sample vehicle image, reviews customer comments, and produces a short service summary.

**Live demo:** _add your Streamlit Community Cloud link here_

> All results are demonstrations on generated classroom data, two CIFAR-10 image classes and authored comments. Estimates are not delivery guarantees. The app has no GPS, maps, route optimisation, driver ranking, automated customer messages or database.

---

## The five tabs

| Tab | What it does | Model / method |
|---|---|---|
| **Delivery overview and duration estimate** | Data overview and charts; job form (or a supplied example record); input checks; estimated duration in minutes; CSV check for missing columns | Decision tree regressor (`models/supervised.joblib`) |
| **Delivery profiles** | Which of three delivery groups the job belongs to; group sizes and averages in original units; scatter plot | StandardScaler + KMeans, k = 3 (`models/clusters.joblib`) |
| **Vehicle image** | Sample or uploaded PNG/JPG → RGB, 32×32, batch of 1 → car/truck suggestion; the worker confirms the vehicle manually | MobileNetV2 transfer + augmentation (`models/images.keras`); the required small CNN (`models/images_small_cnn.keras`) can be selected in the sidebar |
| **Customer comments** | Supplied comments by dropdown, optional typed comment (≤ 1000 characters); sentiment, keyword topic with the matched word; counts and charts; agreement with the reference labels | NLTK VADER with the supplied offline lexicon; keyword rules from `data/topic_keywords.json` |
| **Service summary** | **Offline summary** (default, fixed template, no key needed) or an optional AI summary from the Groq Free Plan, with a fallback to offline on any error | Groq `openai/gpt-oss-20b` with the structured prompt from the notebook |

The app **never trains or refits a model**. All models were trained once in the Colab notebook and are loaded with `st.cache_resource`.

---

## Repository contents

```
app.py                      Streamlit interface and validation functions
requirements.txt            Runtime dependencies (versions match the training run)
.streamlit/config.toml      Theme and upload limit
.streamlit/secrets.toml.example   Template for the optional Groq key
models/                     supervised.joblib, clusters.joblib, images.keras,
                            images_small_cnn.keras, cluster_descriptions.json, metadata.json
data/                       main.csv, feedback.csv, example_inputs.csv, meta.json,
                            topic_keywords.json, prompt_test_cases.json
nltk_data/                  Supplied VADER lexicon (no download at startup)
sample_images/              One supplied example image per class
source_notices/             Data and lexicon attribution (keep with any copy)
reports/                    Small evidence files shown in the app (metrics, prompt comparison, test log)
```

Not included on purpose: `images.npz` (full training arrays), dataset archives, and any API key.

---

## Run locally

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app works fully offline. To try the optional AI summary, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add a free Groq key. Never commit that file.

> **Important:** keep `scikit-learn==1.6.1`. The saved `.joblib` models were created with this version and do not load in newer versions.

---

## Deploy on Streamlit Community Cloud (free)

1. Create a **GitHub Free** repository and upload every file and folder above (*Add file → Upload files → Commit*). Keep the folder structure.
2. Go to **share.streamlit.io**, sign in with GitHub and choose **Create app → Deploy a public app from GitHub**.
3. Select the repository, branch `main` and main file `app.py`.
4. Open **Advanced settings**, choose **Python 3.12**, and in **Secrets** paste (optional):
   ```toml
   GROQ_API_KEY = "your-free-groq-key"
   GROQ_MODEL = "openai/gpt-oss-20b"
   ```
5. Deploy. The first build installs TensorFlow and takes a few minutes.

Keep the Groq account on the **free plan** with no billing. If free quota runs out, the app keeps working and shows the offline summary. If a free hosting limit blocks the live demo, run the app locally and submit a screen recording.

---

## Results (from the training notebook, demonstration data)

| Task | Result |
|---|---|
| Duration estimate (test rows) | Decision tree MAE 5.46 min, RMSE 6.72 min, R² 0.891 · mean baseline MAE 17.54 min |
| Delivery profiles | 3 groups: lighter parcels and traffic (237 jobs), heavier traffic (254), heavier parcels with lighter traffic (209) |
| Vehicle images (200 test images) | Required small CNN 73.0% accuracy · final transfer model 92.5% accuracy, macro F1 0.925 |
| Sentiment (30 supplied comments) | VADER accuracy 96.7%, macro F1 0.967 |
| Prompt comparison (5 cases × 2 prompts) | Structured prompt 14/15 checks passed · vague prompt 2/15 |

---

## Test log

Acceptance checks and expected results are in `reports/test_log.csv`. Record the **observed** result for each check on the deployed app. To test API failure, switch on **Test mode: simulate API failure** in the sidebar, or set a wrong key in Secrets.

| Check | How to test in the app |
|---|---|
| No Groq key | Deploy without secrets; every tab works and the offline summary is shown |
| Supplied example record | Tab 1 → choose an example → **Estimate duration** |
| Sample image | Tab 3 → supplied sample → processed 32×32 RGB image and suggestion |
| Blank comment | Tab 4 → leave the box empty → **Analyse comment** |
| distance_km = 0 | Tab 1 → set distance to 0 → **Estimate duration** → rejected |
| Missing column | Tab 1 → *Check a CSV of jobs* → upload a CSV without `stops_count` |
| API failure | Sidebar test mode on → Tab 5 → AI summary → **Generate AI summary** |
| Restart / rerun | Sidebar shows when models were loaded; reruns do not reload or train |

---

## Limitations

- Delivery records are generated with a noisy teaching recipe; scores are demonstration results.
- The image models see 32×32 CIFAR-10 images. They can be wrong on real photos, cannot recognise other objects, and model scores are not calibrated probabilities. The worker always confirms the vehicle.
- VADER and keyword topics describe tone and subject only; negation, sarcasm and mixed opinions can be misread. English only.
- Clustering groups similar jobs; it does not optimise routes or prove causes. Group numbers are arbitrary.
- AI summaries can contain mistakes; the app shows the calculated results next to them.

## Sources

- Keras CIFAR-10 dataset: https://keras.io/api/datasets/cifar10/
- VADER lexicon: C. J. Hutto and Eric Gilbert (2014), via NLTK data
- Delivery scenarios and comments: authored for this class (see `source_notices/ATTRIBUTION.txt`)

## Team

| Student | Main contribution |
|---|---|
| _Name_ | _e.g. tabular data and clustering_ |
| _Name_ | _e.g. CNN and sentiment_ |
| _Name_ | _e.g. app integration and deployment_ |

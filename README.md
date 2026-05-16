---
title: SurakshaSetu - AP Flood Alert
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
---

## Live Demo & Code

- **Live Demo:** https://huggingface.co/spaces/kakaramanojdath/suraksha-setu
- **Kaggle Notebook:** https://www.kaggle.com/code/kakaramanojdath/suraksha-setu
- **Video Demo:** https://youtu.be/b5EA5r8SC7g

---

# SurakshaSetu — Andhra Pradesh Flood Early Warning System

**Protecting 5.3 crore people across 660+ mandals in 26 districts of Andhra Pradesh**

Real-time mandal-level flood risk using IMD rainfall thresholds + GloFAS river discharge + Gemma 4 AI — with Telugu voice alerts for last-mile rural reach.

---

## What It Does

SurakshaSetu gives village-level (mandal-level) flood risk warnings for every mandal in Andhra Pradesh. Select any of the 660+ mandals and get:

- **Flood Risk Gauge** — score 0–100 with LOW / MEDIUM / HIGH / EXTREME level
- **16-Day Rainfall Forecast** — color-coded by IMD thresholds (Light / Moderate / Heavy / Extreme)
- **River Discharge Forecast** — GloFAS data with 1,500 m³/s danger threshold line
- **AP Flood Alert — Telugu Hechcharika** — transliterated Telugu warning prefixed with "SurakshaSetu Hechcharika:"
- **Voice Alert (Telugu)** — audio in Telugu via Sarvam AI → gTTS → HF MMS-TTS fallback chain
- **Risk Analysis** — peak day, alert days, recommended action, AI reasoning
- **Mandal Map** — SVG boundary polygon with risk color overlay (pure client-side, no CDN)

**Who It Helps:** ASHA workers, Panchayat officials, farmers, fishermen, and rural communities who cannot read — Telugu audio alerts work even without literacy.

---

## Model & Accuracy

| Metric | Value |
|--------|-------|
| Overall accuracy | **95.2%** |
| Rule-based only accuracy | **88%+** |
| Events validated | 12 historical AP flood events (2018–2024) |
| Validation method | Archive rainfall vs known flood outcomes |
| Ensemble weights | Rainfall 40% + Discharge 35% + Gemma 4 AI 25% |

**Validated events include:** Rajahmundry 2018, Vijayawada 2018 & 2022 & 2024, Kurnool 2019 & 2020, Nellore 2019 & 2020 & 2023 (Cyclone Michaung), Godavari 2021, Ongole 2021

---

## Special Technology Stack

This app uses **all five** Special Technology Track frameworks:

| Framework | How Used | File |
|-----------|----------|------|
| **Gemma 4** | Primary inference via Google AI Studio (`gemma-4-26b-a4b-it`) | `app.py` |
| **Ollama** | Local LLM fallback — `gemma3`, `llama3.2`, `mistral` | `app.py` → `_call_ollama()` |
| **llama.cpp** | GGUF file + OpenAI-compatible server fallback | `app.py` → `_call_llamacpp()` |
| **LiteRT** | Google TFLite Runtime for edge/on-device inference | `app.py` → `_call_litert()` |
| **Cactus** | Mobile JSON export for on-device deployment | `app.py` → `_CACTUS_MODE` flag |

### Inference Chain (priority order)

```
Gemma 4 API (Google AI Studio)       ← primary, function calling
    ↓ if GOOGLE_API_KEY absent
Ollama (local: gemma3 / llama3.2)    ← optional, set OLLAMA_HOST
    ↓ if not reachable
llama.cpp (server or direct GGUF)    ← optional, set LLAMACPP_HOST or LLAMACPP_MODEL_PATH
    ↓ if not reachable
LiteRT (TFLite on CPU)               ← optional, set LITERT_MODEL_PATH
    ↓ always available
Rule-based (IMD thresholds + GloFAS) ← no dependencies, 88%+ accuracy
```

The active backend is auto-detected at startup via `_detect_inference_backend()`. No manual configuration needed — the app falls through the chain until a working backend is found and logs the selected backend at startup.

### Voice Alert Chain (TTS)

```
Sarvam AI bulbul:v2 (te-IN, speaker: meera)   ← if SARVAM_API_KEY set
    ↓
gTTS Google Text-to-Speech (lang="te")         ← free fallback
    ↓
HuggingFace facebook/mms-tts-tel               ← free Telugu model via HF Inference API
```

All alerts are prefixed with **"SurakshaSetu Hechcharika:"** in both text and audio output.

### Cactus Mobile Export

When `CACTUS_EXPORT=1` env var is set, every analysis generates a structured JSON file at `/tmp/cactus_export_*.json` formatted for Cactus on-device inference:

```json
{
  "schema": "suraksha_setu_v1",
  "mandal": "Rajahmundry",
  "district": "East Godavari",
  "lat": 17.0005,
  "lon": 81.804,
  "risk_level": "HIGH",
  "risk_score": 72.4,
  "confidence": 88.5,
  "peak_rainfall_day": 3,
  "peak_discharge_m3s": 1840.0,
  "alert_days": [2, 3, 4],
  "recommended_action": "Evacuate low-lying areas near Godavari river immediately.",
  "telugu_alert": "SurakshaSetu Hechcharika: Rajahmundry mandal lo adhika pramadam undi...",
  "generated_at": "2026-05-16T10:30:00",
  "inference_backend": "gemma4_api"
}
```


---

## Risk Scoring

### Rainfall Score (40% of final score)

5-factor model — not just peak value:

| Factor | Weight | Description |
|--------|--------|-------------|
| Peak daily rainfall (IMD class) | 35% | Classified per IMD thresholds |
| Cumulative 7-day rainfall | 30% | Sustained rainfall total |
| Consecutive rainy days | 20% | Soil saturation proxy |
| Heavy rain days count | 10% | Days above Light threshold |
| Extreme rain flag | 5% | Any day above 204.4 mm |

**IMD thresholds:** No Rain ≤2.4 mm | Very Light ≤7.4 mm | Light ≤35.4 mm | Moderate ≤64.4 mm | Heavy ≤115.5 mm | Very Heavy ≤204.4 mm | Extreme >204.4 mm

### Discharge Score (35% of final score)

4-factor model:

| Factor | Weight | Description |
|--------|--------|-------------|
| Absolute danger threshold | 40% | vs 1,500 m³/s GloFAS danger level |
| Ratio vs historical average | 30% | How anomalous is the current event |
| Rising trend | 20% | First 3 days avg vs last 3 days avg |
| Days above warning level | 10% | Days above 70% of danger threshold |

### Gemma 4 AI Score (25% of final score)

Native function calling — single API call returns risk score + Telugu alert + reasoning. Adaptive ensemble weights:

- **Agreement < 10 points** → weights `0.38 / 0.33 / 0.29` (trust all three equally)
- **Disagreement ≥ 10 points** → weights `0.42 / 0.38 / 0.20` (lean on validated rule-based)

### Monsoon Season Boost

June–October: all scores boosted **15%** to account for AP monsoon season patterns. Consecutive 3+ rainy days add an additional **5–10%** soil saturation factor.

### Confidence Calculation

Accuracy-tied confidence range: **72%–96%**. Higher when rainfall and discharge signals agree, monsoon season active, and data quality is high. Gemma confidence used only when it differs from 75 (the default), indicating a genuine signal.

---

## Telugu Alert

The Telugu alert box is labeled **"AP Flood Alert — Telugu Hechcharika"** and every alert begins with **"SurakshaSetu Hechcharika:"** — making it instantly identifiable in both the text display and voice audio.

**Example output:**
> SurakshaSetu Hechcharika: Vijayawada mandal lo 0.5mm takkuva varsham padavachhu. Takkuva pramadam undi, kabatti makkalu jagrattaga undandi.

Clean filter removes Gemma self-check text (word counts, constraint lists, English instructions) before display.

---

## Offline Mode

Set `OFFLINE_MODE=true` to run the app with no internet connection — designed for disaster zones with network outages:

- All external API calls skipped
- Lightweight Open-Meteo-only fetch attempted first; if that fails, zero-valued safe defaults used
- App runs on rule-based scoring only (IMD thresholds + GloFAS discharge logic)
- Accuracy: 88%+ (rule-based alone)
- All charts and outputs still render — no degraded UI

---

## Environment Variables (HF Spaces Secrets)

| Secret | Required | Purpose |
|--------|----------|---------|
| `GOOGLE_API_KEY` | Recommended | Gemma 4 API via Google AI Studio |
| `HF_TOKEN` | Optional | HuggingFace Inference API (TTS fallback) |
| `SARVAM_API_KEY` | Optional | Sarvam AI TTS — highest quality Telugu audio |
| `OLLAMA_HOST` | Optional | Custom Ollama server URL (default: localhost:11434) |
| `LLAMACPP_HOST` | Optional | llama.cpp server URL (default: localhost:8080) |
| `LLAMACPP_MODEL_PATH` | Optional | Path to local GGUF model file |
| `LITERT_MODEL_PATH` | Optional | Path to local `.tflite` model file |
| `CACTUS_EXPORT` | Optional | Set to `1` to enable Cactus JSON export |
| `OFFLINE_MODE` | Optional | Set to `true` to disable all API calls |

**Without any API keys:** app runs fully in rule-based mode. No degraded UI — accuracy is 88%+ rule-based alone.

---

## Coverage

- **26 districts** — all of Andhra Pradesh
- **660+ mandals** — full state coverage including remote tribal areas (Alluri Sitharama Raju, Parvathipuram Manyam)
- **Coastal zones** — Srikakulam, Vizianagaram, Visakhapatnam, East/West Godavari, Krishna, Guntur, Prakasam, Nellore
- **River basins** — Godavari, Krishna, Tungabhadra, Handri, Nagavali, Vamsadhara
- **Default mandal:** Kotanandhuru (Kakinada district) — coastal flood zone

---

## Data Sources

| Source | Data | Update frequency |
|--------|------|-----------------|
| [Open-Meteo Forecast API](https://api.open-meteo.com) | 16-day rainfall + weather forecast | Hourly |
| [Open-Meteo GloFAS Flood API](https://flood-api.open-meteo.com) | 16-day river discharge forecast | Daily |
| [Open-Meteo Archive API](https://archive-api.open-meteo.com) | 90-day historical baseline | On request |
| [Open-Meteo Geocoding API](https://geocoding-api.open-meteo.com) | Mandal lat/lon coordinates | Static |
| [Google AI Studio](https://aistudio.google.com) | Gemma 4 AI risk scoring + Telugu alert | Per request |

All weather data is free and open. No paid data subscriptions required.

---

## Architecture Overview

```
User selects mandal
       ↓
geocode_mandal()          — Open-Meteo Geocoding API (hardcoded fallback for Kotanandhuru)
       ↓
run_all_parallel()        — 3 parallel threads (ThreadPoolExecutor):
  ├── get_rain()          →  16-day rainfall forecast (Open-Meteo)
  ├── get_flood()         →  16-day river discharge (GloFAS Flood API)
  └── get_arch()          →  90-day historical baseline (Open-Meteo Archive)
       ↓
analyze_risk()
  ├── _rs()               — Rainfall score (5-factor, 40% weight)
  ├── _ds()               — Discharge score (4-factor, 35% weight)
  ├── monsoon_boost       — +15% June–October seasonal adjustment
  ├── soil_saturation     — +5–10% for 3+ consecutive rainy days
  └── _call_gemma_with_tools() — AI score + Telugu alert (25% weight, function calling)
       ↓
Output pipeline
  ├── chart_gauge()       — Semi-circular risk gauge (matplotlib)
  ├── chart_rainfall()    — 16-day IMD-coloured bar chart
  ├── chart_discharge()   — 16-day discharge line chart + danger threshold
  ├── chart_map()         — Pure SVG mandal boundary map (660+ boundaries, no CDN)
  ├── telugu_alert()      — Telugu warning text with "SurakshaSetu Hechcharika:" prefix
  └── voice_alert()       — Audio: Sarvam AI → gTTS → HF facebook/mms-tts-tel
```

---

## Files

| File | Purpose |
|------|---------|
| `app.py` | Main HuggingFace Spaces application (~2,200 lines) |
| `suraksha_setu_kaggle.py` | Kaggle notebook — 12-event accuracy validation, Gemma 4 inference demo |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

---

## Local Development

```bash
pip install gradio requests pandas numpy matplotlib gtts huggingface_hub
export GOOGLE_API_KEY=your_key_here
python app.py
```

For Cactus mobile export testing:

```bash
export CACTUS_EXPORT=1
python app.py
# Check /tmp/cactus_export_*.json after each analysis
```

For offline / disaster-zone testing:

```bash
export OFFLINE_MODE=true
python app.py
# Runs fully rule-based — no internet needed
```

For Ollama local inference:

```bash
ollama pull gemma3
export OLLAMA_HOST=http://localhost:11434
python app.py
```

---

## High Flood-Risk Example Mandals

| Mandal | District | Flood Type |
|--------|----------|-----------|
| Rajahmundry | East Godavari | Godavari river floods |
| Vijayawada | Krishna | Krishna river floods |
| Eluru | West Godavari | Frequent river flooding |
| Bhimavaram | West Godavari | Coastal + river floods |
| Machilipatnam | Krishna | Cyclone + river floods |
| Amalapuram | East Godavari | Delta floods |
| Nellore | Nellore | Cyclone-prone coastal |
| Kurnool | Kurnool | Tungabhadra + Handri floods |
| Kotanandhuru | Kakinada | Coastal flood zone (default) |
| Kakinada | East Godavari | Coastal + river floods |

# Screenshots for the project README

These are REAL screenshots of the running application, not mockups and not generated
images. That is not a style preference. This project is an AI-content detector, and a
README illustrated with AI-generated imagery presented as product screenshots is an
unforced error that anyone can catch by running the detector on it.

Capture each at a browser width of about 1600px, in dark mode, from the deployed app or
from `streamlit run streamlit_app.py`.

## In place

| File | Shows |
|---|---|
| `text_verdict.png` | Text tab, both arms scored on one document, with the threshold each used |
| `image_verdict.png` | Image tab, verdict banner plus evidence and file signals |
| `image_robustness.png` | Eleven transforms re-scored against the original |
| `image_attribution.png` | 5x5 occlusion attribution over the detector's tensor space |

## Worth adding

| File | What to capture |
|---|---|
| `text_missed.png` | A ChatGPT paragraph coming back NO AI DETECTED. The honest one, and the most interesting screenshot in the project: it shows the published 63-96% out-of-distribution miss rate happening live. |
| `image_details.png` | The expanded Details panel: manipulation analysis, provenance, per-stage timing. |

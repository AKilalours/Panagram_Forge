# Screenshots for the project README

These are REAL screenshots of the running application, not mockups and not generated
images. That is not a style preference. This project is an AI-content detector, and a
README illustrated with AI-generated imagery presented as product screenshots is an
unforced error that anyone can catch by running the detector on it.

Capture each at a browser width of about 1600px, in dark mode, from the deployed app or
from `streamlit run streamlit_app.py`.

| File | What to capture |
|---|---|
| `text_verdict.png` | Text tab after analysing `demo/ai_smollm_01.txt`. Include the banner with the gauge AND both arm cards. |
| `image_verdict.png` | Image tab after uploading a photograph. Include the banner and the Image / Evidence / File signals row. |
| `image_attribution.png` | Image tab scrolled to the occlusion panel, with the Robustness chips visible above it if they fit. |

Optional extras, if you want more in the README later:

| File | What to capture |
|---|---|
| `image_details.png` | The expanded Details panel: manipulation analysis, provenance, timing. |
| `text_missed.png` | A ChatGPT paragraph coming back NO AI DETECTED. This is the honest one, and the most interesting screenshot in the project. |

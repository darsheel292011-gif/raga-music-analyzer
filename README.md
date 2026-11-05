# Raga & Pitch Explorer

A practical first version of a Python music-analysis web app. It accepts audio files and video files with sound (including MP4), extracts the audio, estimates melodic pitches and a possible tonic, displays observed pitch classes and semitone positions from Sa, shows experimental matches against a small raga scale catalog, and gives a pitch-shift interval to a target tonic.

## Important limits

The app's raga section shows **experimental scale matches**, not reliable raga identification. It ranks a small built-in catalog by how much detected pitch-class activity fits each scale. Ragam identity also depends on characteristic phrases, gamakas, and other melodic behavior that this baseline does not analyze. A reliable classifier needs a validated model trained and evaluated on suitable labeled recordings; the scale resemblance number is not a model confidence score.

The app can try many formats through FFmpeg, but a file must contain an audio stream. It cannot analyze silent video, images, documents, or arbitrary data files. The tonic is a pitch-class prominence hint, not a guaranteed Sa. A drone, bass note, accompaniment, changing key, or noisy/polyphonic mix can mislead it. Pitch tracking approximates continuous gamakas and bends as equal-tempered semitones. The transpose result is guidance only; this version does not render modified audio.

## Windows 11 setup

Install Python 3.11 (64-bit) from [python.org](https://www.python.org/downloads/) and select **Add python.exe to PATH** in the installer. Open PowerShell in this project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, use this for the current terminal only, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

The app opens at `http://localhost:8501`. Streamlit's upload limit can be adjusted in `.streamlit/config.toml`; keep in mind larger files need more memory and take longer to analyze.

## Deploy publicly

### Streamlit Community Cloud

1. Create a GitHub repository and put these project files in its root folder.
2. Push the repository to GitHub.
3. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub and choose **Create app**.
4. Select the repository, branch, and `app.py`, then deploy. This produces a public `*.streamlit.app` URL.

See the [official deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

### Google Search

Google does not host the app. Public deployment makes the site reachable; it does not guarantee that Google will index it. For a more searchable public site, use a custom domain and add explanatory public pages (for example, about, methodology, and privacy) alongside the tool. Verify the domain in [Google Search Console](https://search.google.com/search-console/), inspect the deployed URL, and request indexing when available. Google may take time and does not guarantee crawling or indexing; see [Google's guidance](https://developers.google.com/search/docs/fundamentals/get-on-google).

For Google Cloud hosting, Streamlit's [deployment tutorials](https://docs.streamlit.io/deploy/tutorials) include Google Cloud Run. A custom domain plus a crawlable landing page generally gives more control over search presentation than relying only on an interactive app screen.

## How the baseline works

- `soundfile` decodes supported audio formats directly. `imageio-ffmpeg` provides FFmpeg for extracting audio from video containers and decoding additional formats. `librosa` performs resampling and probabilistic fundamental-frequency tracking.
- Stable voiced frames are rounded to semitones and grouped by pitch class.
- The most prominent pitch class is shown as a tonic hint. Users can override Sa manually.
- Each observed pitch class is displayed by its semitone offset from Sa. Swara labels are approximate, with enharmonic Carnatic positions noted.
- The experimental raga comparison ranks ten built-in scales by pitch-class agreement. Treat it as a rough shortlist for exploration, not an answer.
- Transposition is the signed semitone distance from the selected Sa to the target Sa.

## Extending to raga classification

Keep model development separate from the pitch display. Train and validate with ragas represented across artists, instruments, recording conditions, and tonic ranges; prevent artist/track leakage between splits; compare against a no-model baseline; and expose calibrated confidence and an abstain option. Then add a classifier adapter that returns ranked candidates only when the evaluation supports them.

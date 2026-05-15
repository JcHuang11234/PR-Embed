# PR-Embed HTML Version

This folder contains a static HTML/CSS/JavaScript version of the PR-Embed Streamlit app.

Run it from this folder with:

```bash
python3 -m http.server 8765 --bind 127.0.0.1
```

Then open:

```text
http://127.0.0.1:8765/
```

Do not open `index.html` directly with `file://`; the browser needs a local web server so it can fetch the exported JSON and binary data files.

Main files:

- `index.html`: static app shell
- `assets/styles.css`: layout and visual styling
- `assets/app.js`: browser-side app logic
- `data/`: exported paper metadata, STM topics, paper embeddings, topic embeddings, and Word2Vec model vectors
- `build_static_data.py`: exporter that rebuilds the `data/` folder from the original project files

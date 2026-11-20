# Faris Almogbel · Portfolio ("a portfolio, told as a walk")

Open `index.html` through any static server:

```
python -m http.server 8000
# then open http://localhost:8000
```

## Where to edit things
All the text, links and project data live in **one `DATA` object** at the top of the `<script>` in `index.html`:

- `DATA.contact`: email, phone, LinkedIn, GitHub
- `DATA.ui`: interface text (English and Arabic)
- `DATA.stops`: the stops of the walk, in order (headline, caption lines, sky: 0 day, 1 sunset, 2 night)
- `DATA.certs`: the three frames on the gallery wall
- `DATA.projects`: the objects on the workbench. Add screenshots by putting the paths in `images: ["assets/projects/wasl-1.webp"]` and a link in `link: "https://..."`. Set `hidden: false` on Wasl to show it on the table too.
- `DATA.mapPts`: where each landmark sits on the map (fractions of the map image)

Deep links: `#at=uqu`, `#at=kidana`, `#at=wasl`, `#at=certs`, `#at=projects`, `#at=end`, and `#lang=ar`.

## Regenerating art
The original renders are in `raw/`. Replace one of them, then run:

```
pip install pillow numpy scipy
python tools/process_assets.py
```

The script keys out the green, magenta and blue backgrounds, splits the sprite sheets, stamps the real logos from `brand/` (UQU on the gate, KIDANA on the building, Wasl on the bus), writes `assets/*.webp` and makes `tools/preview.png` so you can check the cut-outs.

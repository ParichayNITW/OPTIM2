"""
Run this once on a machine with Pillow + cairosvg installed to generate icon.ico:
    pip install Pillow cairosvg
    python assets/make_icon.py
"""
import io
from pathlib import Path

try:
    import cairosvg
    from PIL import Image

    here = Path(__file__).parent
    png_data = cairosvg.svg2png(url=str(here / "icon.svg"), output_width=256, output_height=256)
    img = Image.open(io.BytesIO(png_data))
    img.save(str(here / "icon.ico"), format="ICO", sizes=[(16,16),(32,32),(48,48),(256,256)])
    print("icon.ico created successfully")
except ImportError:
    print("Install Pillow and cairosvg first: pip install Pillow cairosvg")

# Paper Webcam Implementation (NumPy on CPU)

This repo provides a high-performance webcam processing scaffold using NumPy on CPU, with the algorithm isolated in `paper_webcam/algorithm.py`. The main loop targets 24+ FPS on typical laptop CPUs. Capture/display use OpenCV; algorithm math uses NumPy.

## Install

- Windows 11 (or any OS with Python 3.9+)
- Python 3.9–3.12 recommended

```bash
pip install -r requirements.txt
```

If you have multiple Python versions, use `py -m pip` on Windows:

```bash
py -m pip install -r requirements.txt
```

## Run

```bash
python -m paper_webcam.main --camera-index 0 --width 640 --height 480 --scale 1.0
```

Windows users may get lower latency using DirectShow:

```bash
python -m paper_webcam.main --camera-index 0 --use-dshow 1
```

Keys:
- `q`: quit
- `g`: toggle grayscale view
- `o`: toggle FPS overlay

## Algorithm

Edit `paper_webcam/algorithm.py` to implement the paper-specific method. The current placeholder demonstrates a fast adaptive gamma correction using a per-frame LUT with NumPy only.

## Notes

- For 24+ FPS on Intel i5-1035G4, prefer 640x480 or 960x540 input; use `--scale 0.75` or `--scale 0.5` if needed.
- The processing core is vectorized NumPy; no GPU required.

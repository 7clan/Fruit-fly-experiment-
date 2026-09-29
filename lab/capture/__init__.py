"""Capture adapters: interface + synthetic dev source + guarded Windows
Graphics.Capture adapter. ONE stream feeds vision AND the dashboard."""
from .base import (CaptureAdapter, CaptureError, NullCapture,  # noqa: F401
                   SyntheticCapture, create_windows_capture,
                   windows_capture_available)

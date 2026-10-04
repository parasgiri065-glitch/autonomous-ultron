"""Optional provider adapters used by the offline-safe waterfall."""

from .zero_auth import AdapterResponse, PollinationsAdapter, ZeroAuthUnavailable

__all__ = ["AdapterResponse", "PollinationsAdapter", "ZeroAuthUnavailable"]

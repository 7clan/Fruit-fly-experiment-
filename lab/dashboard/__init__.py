"""Dashboard: 3-column snapshot reader (FLY BRAIN / HYBRID HELPER /
ACTION SYSTEM). Reads snapshots ONLY — never owns the control clock."""
from .dashboard import (DashboardWorker, Snapshot,  # noqa: F401
                        TextDashboardRenderer)

"""Branch-first surface-track validation entry point.

Previous endpoint-conditioned results are read as frozen comparison artifacts.
The default execution no longer runs endpoint pairing or full-height masking.
"""
from surface_track_validation import main


if __name__ == '__main__':
    main()

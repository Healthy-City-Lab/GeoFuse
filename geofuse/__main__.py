"""``python -m geofuse``: the command line (``geofuse --help``)."""

if __name__ == "__main__":
    import sys

    from geofuse.cli import main

    sys.exit(main())

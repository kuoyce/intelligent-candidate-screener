"""Primary guided HR portal; the historical free-form prototype is separate."""
from candidate_screener.app import watcher_logging
from candidate_screener.app.portal_ui import main

watcher_logging.install()

if __name__ == "__main__":
    main()

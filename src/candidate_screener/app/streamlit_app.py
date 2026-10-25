"""Primary guided HR portal; the historical free-form prototype is separate."""
from candidate_screener.app import portal_inference, watcher_logging
from candidate_screener.app.portal_ui import main

watcher_logging.install()
portal_inference.use_as_default()

if __name__ == "__main__":
    main()

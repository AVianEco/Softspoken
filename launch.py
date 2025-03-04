import logging
logging.basicConfig(level=logging.INFO)

import os
from root.code.frontend import silencer_ui

if __name__ == "__main__":
    logging.info(f"Working directory: {os.getcwd()}")

    silencer_ui.main()


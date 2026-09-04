"""Business-side Serve driver.

The driver runs from the business image and connects to the Ray head.  The
head image therefore only needs to run the Ray control plane.
"""

import os
import ray
from ray import serve

from main import app


if __name__ == "__main__":
    address = os.getenv("RAY_ADDRESS", "auto")
    for attempt in range(30):
        try:
            ray.init(address=address)
            break
        except Exception:
            if attempt == 29:
                raise
            import time

            time.sleep(2)
    serve.start(http_options={"host": "0.0.0.0", "port": 8000})
    serve.run(app, name="panns", route_prefix="/")
    try:
        import time

        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass

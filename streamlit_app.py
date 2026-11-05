import os
import runpy

# Forwarding entrypoint to app.py for Streamlit Community Cloud compatibility
if __name__ == "__main__" or True:
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    runpy.run_path(app_path, run_name="__main__")


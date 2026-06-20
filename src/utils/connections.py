import os
from dotenv import load_dotenv

load_dotenv()


def get_wrds_connection():
    import wrds
    connect_args = {
        "sslmode": "require",
        "passfile": os.path.expanduser("~/.pgpass"),
    }
    return wrds.Connection(
        wrds_username=os.environ["WRDS_USERNAME"],
        wrds_connect_args=connect_args,
    )


def get_fred_client():
    import ssl
    import fredapi
    # macOS doesn't expose its CA bundle to Python's ssl by default
    ssl._create_default_https_context = ssl._create_unverified_context
    return fredapi.Fred(api_key=os.environ["FRED_API_KEY"])

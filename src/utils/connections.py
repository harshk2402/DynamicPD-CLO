import os
from dotenv import load_dotenv

load_dotenv()


def get_wrds_connection():
    import wrds
    return wrds.Connection(wrds_username=os.environ["WRDS_USERNAME"])


def get_fred_client():
    import ssl
    import fredapi
    # macOS doesn't expose its CA bundle to Python's ssl by default
    ssl._create_default_https_context = ssl._create_unverified_context
    return fredapi.Fred(api_key=os.environ["FRED_API_KEY"])

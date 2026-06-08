import os
from dotenv import load_dotenv

load_dotenv()


def get_wrds_connection():
    import wrds
    return wrds.Connection(wrds_username=os.environ["WRDS_USERNAME"])


def get_fred_client():
    import fredapi
    return fredapi.Fred(api_key=os.environ["FRED_API_KEY"])

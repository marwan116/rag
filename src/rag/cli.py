from typer import Typer

app = Typer()


@app.command
def fetch():
    """Fetches data from the API."""
    from rag.utils import load_config
    from rag.docs_fetcher import DocumentFetcher
    
    config = load_config()
    config["fetch_data"]
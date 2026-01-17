import typer

from cli.commands import ask, index, search
from rag.logger import setup_logging


app = typer.Typer(
    name="rag",
    help="Terminal RAG CLI Application",
    no_args_is_help=True,
    add_completion=False,
)

@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="상세 로그 출력"),
):
    """
    Terminal RAG CLI
    """
    log_level = "DEBUG" if verbose else "WARNING"
    setup_logging(log_level=log_level)


# Register commands
app.command(name="index")(index.handle_index)
app.command(name="ask")(ask.handle_ask)
app.command(name="search")(search.handle_search)


if __name__ == "__main__":
    app()

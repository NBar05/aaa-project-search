from aiohttp.web import run_app
from lib.app import create_app


def main(port: int = 8080) -> None:
    """
    Entry point for running the aiohttp web application.
    """
    app = create_app()
    run_app(app, port=port)


if __name__ == '__main__':
    main()

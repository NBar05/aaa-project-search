from pathlib import Path

import aiohttp_jinja2
import jinja2
from aiohttp.web import Application

from lib import views

from lib.engine import create_connection
from lib.models import create_model


LIB_PATH = Path('lib')


def create_app() -> Application:
    """
    Create and configure an instance of the Application class.

    :return: Configured instance of the Application class.
    """
    app = Application()

    # Setup static routes and views
    app.router.add_static('/static/', LIB_PATH / 'static')
    app.router.add_view('/', views.IndexView, name='index')

    # Setup jinja2 template engine
    aiohttp_jinja2.setup(
        app=app,
        loader=jinja2.FileSystemLoader(LIB_PATH / 'templates'),
    )

    # Create DB connection and model
    app['client'] = create_connection()
    app['model'] = create_model()

    return app


async def async_create_app() -> Application:
    return create_app()

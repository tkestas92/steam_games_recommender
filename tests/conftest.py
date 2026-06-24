"""
Pytest konfigūracija ir fixtures
"""
import os
import sys
import pytest
from dotenv import load_dotenv

# Pridedame workspace root'ą prie sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Įkeliame .env failą testams
load_dotenv()

# Nustatomos test environment variables
os.environ.setdefault("FLASK_ENV", "testing")


@pytest.fixture(scope="session")
def app():
    """Sukuria Flask aplikaciją test režimui."""
    # Importuojame app čia, po environ nustačiau
    from app.static.shap.app import app as flask_app
    
    flask_app.config["TESTING"] = True
    flask_app.config["SQLALCHEMY_ECHO"] = False
    
    return flask_app


@pytest.fixture
def client(app):
    """Flask test client."""
    return app.test_client()


@pytest.fixture
def runner(app):
    """Flask CLI runner."""
    return app.test_cli_runner()


@pytest.fixture
def session_client(client):
    """Client su sesija."""
    with client:
        return client


# Mock'ini duomenys testams
TEST_GAME_ID = 570
TEST_QUERY = "action"
TEST_INVALID_QUERY_SHORT = "a"  # Per trumpa
TEST_INVALID_QUERY_LONG = "a" * 201  # Per ilga

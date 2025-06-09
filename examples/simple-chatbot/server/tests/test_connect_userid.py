import sys
import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

# Ensure server.py and profiles.py are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
from server import app
from profiles import PROFILES

client = TestClient(app)


def mock_create_room_and_token():
    return ("https://dummy.room.url", "dummy_token")


@patch("server.create_room_and_token", side_effect=mock_create_room_and_token)
def test_connect_valid_user_id_a(mock_create):
    response = client.post("/connect?user_id=a")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "a"
    assert data["profile"]["name"] == PROFILES["a"]["name"]


@patch("server.create_room_and_token", side_effect=mock_create_room_and_token)
def test_connect_valid_user_id_b(mock_create):
    response = client.post("/connect?user_id=b")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "b"
    assert data["profile"]["name"] == PROFILES["b"]["name"]


@patch("server.create_room_and_token", side_effect=mock_create_room_and_token)
def test_connect_invalid_user_id(mock_create):
    response = client.post("/connect?user_id=invalid")
    assert response.status_code == 400
    assert "Invalid user_id" in response.text 
"""Pembacaan konfigurasi rahasia (API key, URI database)."""
from __future__ import annotations

import os

import streamlit as st


def get_secret(name: str, default: str | None = None) -> str | None:
    """Ambil nilai dari st.secrets (secrets.toml) lalu environment variable."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        # secrets.toml tidak ada: lanjut ke environment variable
        pass
    return os.environ.get(name, default)

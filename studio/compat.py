"""Kompatibilitas API Streamlit lintas versi.

Streamlit >= 1.50 memakai ``width="stretch"`` dan mendeprekasi ``use_container_width``.
"""
import streamlit as st


def _new_width_api() -> bool:
    try:
        major, minor = (int(x) for x in st.__version__.split(".")[:2])
    except Exception:
        return True
    return (major, minor) >= (1, 50)


STRETCH = {"width": "stretch"} if _new_width_api() else {"use_container_width": True}

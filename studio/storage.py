"""Penyimpanan komentar publik: MongoDB, dengan fallback file JSON lokal."""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone

import streamlit as st

from studio.config import get_secret

LOCAL_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "comments_local.json")
_lock = threading.Lock()


@st.cache_resource(ttl=60, show_spinner=False)
def _connect(uri: str, db_name: str):
    """Hubungkan ke MongoDB. Hasil (termasuk kegagalan) di-cache 60 detik
    supaya aplikasi tidak menunggu timeout di setiap interaksi."""
    try:
        from pymongo import MongoClient

        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        col = client[db_name]["comments"]
        col.create_index([("created_at", -1)])
        return col, None
    except Exception as exc:
        return None, type(exc).__name__


def backend():
    """Kembalikan (nama_backend, collection_atau_None, keterangan)."""
    uri = get_secret("MONGODB_URI")
    if not uri:
        return "local", None, "MONGODB_URI belum diatur. Komentar disimpan di file lokal."
    col, err = _connect(uri, get_secret("MONGODB_DB", "ml_studio") or "ml_studio")
    if col is None:
        return "local", None, f"Gagal terhubung ke MongoDB ({err}). Memakai file lokal sementara."
    return "mongodb", col, "Terhubung ke MongoDB."


# ---------------- file lokal ----------------
def _read_local() -> list[dict]:
    try:
        with open(LOCAL_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _write_local(rows: list[dict]) -> None:
    tmp = LOCAL_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, LOCAL_FILE)


# ---------------- API publik ----------------
def add_comment(name: str, text: str, context: str, sentiment: dict | None) -> None:
    now = datetime.now(timezone.utc)
    kind, col, _ = backend()
    if kind == "mongodb":
        col.insert_one(
            {"name": name, "text": text, "context": context,
             "sentiment": sentiment, "created_at": now}
        )
    else:
        with _lock:
            rows = _read_local()
            rows.append(
                {"id": uuid.uuid4().hex, "name": name, "text": text, "context": context,
                 "sentiment": sentiment, "created_at": now.isoformat()}
            )
            _write_local(rows)
    list_comments.clear()


@st.cache_data(ttl=20, show_spinner=False)
def list_comments(limit: int = 50) -> list[dict]:
    kind, col, _ = backend()
    out: list[dict] = []
    if kind == "mongodb":
        for d in col.find().sort("created_at", -1).limit(limit):
            created = d["created_at"]
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            out.append(
                {"id": str(d["_id"]), "name": d.get("name", "Anonim"), "text": d.get("text", ""),
                 "context": d.get("context", ""), "sentiment": d.get("sentiment"),
                 "created_at": created}
            )
    else:
        for d in sorted(_read_local(), key=lambda r: r["created_at"], reverse=True)[:limit]:
            out.append({**d, "created_at": datetime.fromisoformat(d["created_at"])})
    return out


def update_sentiment(comment_id: str, sentiment: dict) -> None:
    kind, col, _ = backend()
    if kind == "mongodb":
        from bson import ObjectId

        col.update_one({"_id": ObjectId(comment_id)}, {"$set": {"sentiment": sentiment}})
    else:
        with _lock:
            rows = _read_local()
            for r in rows:
                if r["id"] == comment_id:
                    r["sentiment"] = sentiment
            _write_local(rows)
    list_comments.clear()

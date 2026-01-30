import os
import re
import json
import sqlite3
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import streamlit as st
import numpy as np

# Optional: sentence-transformers
try:
    from sentence_transformers import SentenceTransformer

    _HAS_ST = True
except Exception:
    SentenceTransformer = None
    _HAS_ST = False

APP_TITLE = "MemoHub"
DB_PATH = os.environ.get("MEMOHUB_DB", "memohub.db")
EMBED_MODEL_NAME = os.environ.get(
    "MEMOHUB_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
SIM_THRESHOLD = float(os.environ.get("MEMOHUB_SIM_THRESHOLD", "0.86"))

DEFAULT_TYPE = "NOTE"
CATEGORIES = ["", "work", "study", "infra", "tools", "trading", "other"]


# -------------------------
# Utils
# -------------------------
def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def looks_like_url(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if t.startswith("http://") or t.startswith("https://"):
        return True
    if re.match(r"^[\w\-]+(\.[\w\-]+)+(/.*)?$", t):
        return True
    return False


def canonicalize_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", u):
        u = "https://" + u
    try:
        from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

        parts = urlsplit(u)
        scheme = parts.scheme.lower()
        netloc = parts.netloc.lower()
        netloc = re.sub(r":(80|443)$", "", netloc)

        path = parts.path or "/"
        if path != "/" and path.endswith("/"):
            path = path[:-1]

        q = []
        for k, v in parse_qsl(parts.query, keep_blank_values=True):
            lk = k.lower()
            if lk.startswith("utm_") or lk in {"fbclid", "gclid", "mc_cid", "mc_eid"}:
                continue
            q.append((k, v))
        q.sort(key=lambda x: (x[0], x[1]))
        query = urlencode(q, doseq=True)
        return urlunsplit((scheme, netloc, path, query, ""))
    except Exception:
        return u


def guess_language(content: str) -> str:
    t = (content or "").strip()
    if not t:
        return "text"
    u = t.upper()
    if u.startswith("SELECT ") or " FROM " in u or u.startswith("WITH "):
        return "sql"
    if "CREATE TABLE" in u or "INSERT INTO" in u or "UPDATE " in u or "DELETE " in u:
        return "sql"
    if t.startswith(
        ("pip ", "python ", "streamlit ", "source ", "cd ", "ls ", "export ", "conda ")
    ):
        return "bash"
    return "text"


def display_time(sn: Dict) -> str:
    # Prefer "last_used" as "updated", fallback to "added_date"
    t = (sn.get("last_used") or "").strip()
    return t if t else (sn.get("added_date") or "")


def content_fingerprint(
    title: str,
    content: str,
    description: str,
    canonical_url: str,
    category: str,
    tags: str,
) -> str:
    # light fingerprint for duplicate prevention
    import hashlib

    base = {
        "type": DEFAULT_TYPE,
        "title": normalize_text(title),
        "content": normalize_text(content),
        "description": normalize_text(description),
        "canonical_url": canonical_url,
        "category": normalize_text(category),
        "tags": normalize_text(tags),
    }
    return hashlib.sha1(
        json.dumps(base, ensure_ascii=False, sort_keys=True).encode(
            "utf-8", errors="ignore"
        )
    ).hexdigest()


# -------------------------
# DB
# -------------------------
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS snippets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            language TEXT,
            category TEXT,
            tags TEXT,
            description TEXT,
            parameters TEXT,
            embedding BLOB,
            usage_count INTEGER DEFAULT 0,
            last_used TEXT,
            added_date TEXT
        );
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_snippets_fingerprint
        ON snippets(type, title, content, IFNULL(description,''), IFNULL(category,''), IFNULL(tags,''));
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_snippets_canonical_url
        ON snippets(content)
        WHERE content LIKE 'http%' OR content LIKE '%.%';
        """
    )
    conn.commit()


def row_to_dict(row: Tuple) -> Dict:
    keys = [
        "id",
        "type",
        "title",
        "content",
        "language",
        "category",
        "tags",
        "description",
        "parameters",
        "embedding",
        "usage_count",
        "last_used",
        "added_date",
    ]
    return dict(zip(keys, row))


def fetch_snippet(conn: sqlite3.Connection, sid: int) -> Optional[Dict]:
    cur = conn.execute(
        "SELECT id,type,title,content,language,category,tags,description,parameters,embedding,usage_count,last_used,added_date FROM snippets WHERE id=?",
        (sid,),
    )
    r = cur.fetchone()
    return row_to_dict(r) if r else None


def list_snippets(
    conn: sqlite3.Connection, q: str = "", limit: int = 100
) -> List[Dict]:
    q = (q or "").strip()
    if q:
        like = f"%{q}%"
        cur = conn.execute(
            """SELECT id,type,title,content,language,category,tags,description,parameters,embedding,usage_count,last_used,added_date
               FROM snippets
               WHERE title LIKE ? OR content LIKE ? OR description LIKE ? OR tags LIKE ?
               ORDER BY COALESCE(last_used, added_date) DESC
               LIMIT ?""",
            (like, like, like, like, limit),
        )
    else:
        cur = conn.execute(
            """SELECT id,type,title,content,language,category,tags,description,parameters,embedding,usage_count,last_used,added_date
               FROM snippets
               ORDER BY COALESCE(last_used, added_date) DESC
               LIMIT ?""",
            (limit,),
        )
    return [row_to_dict(r) for r in cur.fetchall()]


def insert_snippet(conn: sqlite3.Connection, data: Dict) -> int:
    cur = conn.execute(
        """INSERT INTO snippets(type,title,content,language,category,tags,description,parameters,embedding,usage_count,last_used,added_date)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            DEFAULT_TYPE,
            data["title"],
            data["content"],
            data.get("language", ""),
            data.get("category", ""),
            data.get("tags", ""),
            data.get("description", ""),
            data.get("parameters", ""),
            data.get("embedding", None),
            0,
            None,
            now_str(),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)  # type: ignore


def update_snippet(conn: sqlite3.Connection, sid: int, data: Dict) -> None:
    conn.execute(
        """UPDATE snippets
           SET title=?, content=?, language=?, category=?, tags=?, description=?, embedding=?, last_used=?
           WHERE id=?""",
        (
            data["title"],
            data["content"],
            data.get("language", ""),
            data.get("category", ""),
            data.get("tags", ""),
            data.get("description", ""),
            data.get("embedding", None),
            now_str(),
            sid,
        ),
    )
    conn.commit()


def delete_snippet(conn: sqlite3.Connection, sid: int) -> None:
    conn.execute("DELETE FROM snippets WHERE id=?", (sid,))
    conn.commit()


# -------------------------
# Embeddings / index
# -------------------------
@st.cache_resource(show_spinner=False)
def load_embedder(model_name: str):
    if not _HAS_ST:
        return None
    return SentenceTransformer(model_name)  # type: ignore


def to_blob(vec: np.ndarray) -> bytes:
    v = vec.astype(np.float32, copy=False)
    return v.tobytes()


def from_blob(b: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32, count=dim)


@dataclass
class VectorIndex:
    ids: np.ndarray
    matrix: np.ndarray  # normalized float32 (N, D)


@st.cache_resource(show_spinner=False)
def build_vector_index(db_path: str, model_name: str) -> Optional[VectorIndex]:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cur = conn.execute("SELECT id, embedding FROM snippets WHERE embedding IS NOT NULL")
    rows = cur.fetchall()
    conn.close()
    if not rows:
        return None

    dim = 384
    emb = load_embedder(model_name) if _HAS_ST else None
    if emb is not None:
        try:
            dim = int(emb.get_sentence_embedding_dimension())  # type: ignore
        except Exception:
            pass

    ids, vecs = [], []
    for sid, blob in rows:
        if not blob:
            continue
        v = from_blob(blob, dim=dim)
        if v.size != dim:
            continue
        ids.append(sid)
        vecs.append(v)

    if not vecs:
        return None

    mat = np.vstack(vecs).astype(np.float32, copy=False)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms

    return VectorIndex(ids=np.array(ids, dtype=np.int64), matrix=mat)


def clear_vector_cache():
    build_vector_index.clear()  # type: ignore


def embed_text(text: str) -> Optional[np.ndarray]:
    if not _HAS_ST:
        return None
    emb = load_embedder(EMBED_MODEL_NAME)
    if emb is None:
        return None
    v = np.array(emb.encode(text), dtype=np.float32)
    n = np.linalg.norm(v)
    return v if n == 0 else (v / n)


def semantic_search(query: str, top_k: int = 8) -> List[Tuple[int, float]]:
    idx = build_vector_index(DB_PATH, EMBED_MODEL_NAME)
    if idx is None:
        return []
    qv = embed_text(query)
    if qv is None:
        return []
    sims = idx.matrix @ qv
    if sims.size == 0:
        return []
    k = min(top_k, sims.size)
    top = np.argpartition(-sims, kth=k - 1)[:k]  # type: ignore
    top = top[np.argsort(-sims[top])]  # type: ignore
    return [(int(idx.ids[i]), float(sims[i])) for i in top]


# -------------------------
# UI
# -------------------------
def clear_add_form():
    for k in [
        "add_title",
        "add_content",
        "add_description",
        "add_category",
        "add_tags",
    ]:
        st.session_state.pop(k, None)

    # 類似確認フローの状態も消す（あるなら）
    for k in ["pending_add", "pending_similar", "pending_hit_id"]:
        st.session_state.pop(k, None)

    # テキスト系は pop より「空を代入」が確実
    st.session_state["add_title"] = ""
    st.session_state["add_content"] = ""
    st.session_state["add_description"] = ""
    st.session_state["add_tags"] = ""
    st.session_state["add_category"] = CATEGORIES[0]


def render_content_block(content: str):
    t = (content or "").strip()
    if ("\n" not in t) and looks_like_url(t):
        can = canonicalize_url(t)
        st.markdown(f"[{t}]({can})")
        st.code(t, language="text")
        return
    st.code(t, language=guess_language(t))


def render_snippet_card(sn: Dict, conn, allow_edit: bool = True):
    with st.container():
        # --- Header ---
        col_title, col_time = st.columns([10, 3])
        with col_title:
            st.markdown(f"**{sn['title']}**")
        with col_time:
            st.markdown(
                f"<div style='text-align:right; color:#999; font-size:0.85em;'>Last Updated: {display_time(sn)}</div>",
                unsafe_allow_html=True,
            )

        # --- Content + Actions ---
        col_content, col_actions = st.columns([13, 1.6])

        with col_content:
            if sn.get("description"):
                st.caption(sn["description"])
            render_content_block(sn["content"])

        with col_actions:
            c_edit, c_del = st.columns([1, 1])
            with c_edit:
                if allow_edit:
                    if st.button(
                        "✏️",
                        key=f"edit_{sn['id']}",
                        help="編集",
                        use_container_width=True,
                    ):
                        st.session_state["view"] = "edit"
                        st.session_state["edit_id"] = int(sn["id"])
                        st.rerun()
            with c_del:
                if st.button(
                    "🗑️",
                    key=f"del_{sn['id']}",
                    help="削除",
                    use_container_width=True,
                ):
                    st.session_state["confirm_delete"] = sn["id"]

        # --- Delete confirmation ---
        if st.session_state.get("confirm_delete") == sn["id"]:
            st.warning("このメモを削除します。元に戻せません。")
            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button(
                    "削除する",
                    key=f"confirm_del_{sn['id']}",
                    type="primary",
                ):
                    delete_snippet(conn, sn["id"])
                    clear_vector_cache()
                    st.session_state.pop("confirm_delete", None)
                    st.toast("削除しました", icon="🗑️")
                    st.rerun()
            with c2:
                if st.button(
                    "キャンセル",
                    key=f"cancel_del_{sn['id']}",
                ):
                    st.session_state.pop("confirm_delete", None)

        st.divider()


def insert_draft(conn: sqlite3.Connection, draft: Dict) -> int:
    """Insert a new snippet (always new). Raises sqlite3.IntegrityError on hard duplicates."""
    emb_blob = None
    if _HAS_ST:
        text = f"{draft.get('title','')}{draft.get('description','')}{draft.get('content','')}".strip()
        v = embed_text(text)
        if v is not None:
            emb_blob = to_blob(v)
    data = dict(draft)
    data["embedding"] = emb_blob
    return insert_snippet(conn, data)


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")
    st.title("MemoHub")
    st.caption("軽量ローカルメモ/リンク保存（検索・編集に特化）")
    st.markdown("---")

    conn = get_conn()
    init_db(conn)

    st.session_state.setdefault("view", "search")
    st.session_state.setdefault("edit_id", None)
    st.session_state.setdefault("confirm_delete", None)

    # ⑤ Sidebar: fold settings
    with st.sidebar:
        with st.expander("⚙ 設定 / 統計", expanded=False):
            cur = conn.execute("SELECT COUNT(*) FROM snippets")
            total = int(cur.fetchone()[0])
            st.metric("総登録数", total)

            use_sem = st.checkbox(
                "セマンティック検索（embedding）", value=True, disabled=not _HAS_ST
            )
            if not _HAS_ST:
                st.info("sentence-transformers 未導入のため無効です。")
            st.caption(f"モデル: `{EMBED_MODEL_NAME}`")
            st.caption(f"類似しきい値: {SIM_THRESHOLD}")
            if st.button("ベクトル索引を再読込"):
                clear_vector_cache()
                st.toast("再読込しました", icon="✅")
                st.rerun()
    # store for later
    st.session_state["use_sem"] = bool(locals().get("use_sem", True))

    # EDIT view (hidden by default)
    if st.session_state["view"] == "edit":
        st.session_state.pop("confirm_delete", None)
        sid = st.session_state.get("edit_id")
        sn = fetch_snippet(conn, int(sid)) if sid else None
        if not sn:
            st.error("編集対象が見つかりません。")
            if st.button("← 戻る"):
                st.session_state["view"] = "search"
                st.session_state["edit_id"] = None
                st.rerun()
        else:
            top_l, top_r = st.columns([1, 1])
            with top_l:
                if st.button("← 検索に戻る", use_container_width=True):
                    st.session_state["view"] = "search"
                    st.session_state["edit_id"] = None
                    st.rerun()
            with top_r:
                st.markdown(
                    f"<div style='text-align:right; color:#999; font-size:0.85em;'>ID={sn['id']} / 更新: {display_time(sn)}</div>",
                    unsafe_allow_html=True,
                )

            st.subheader("編集")
            with st.form("edit_form"):
                title = st.text_input("タイトル", value=sn["title"])
                category = st.selectbox(
                    "カテゴリ",
                    CATEGORIES,
                    index=(
                        CATEGORIES.index(sn.get("category", ""))
                        if sn.get("category", "") in CATEGORIES
                        else 0
                    ),
                )
                tags = st.text_input(
                    "タグ（カンマ区切り）", value=sn.get("tags", "") or ""
                )
                content = st.text_area(
                    "内容（URLでもメモでもOK）", value=sn["content"], height=220
                )
                description = st.text_area(
                    "補足", value=sn.get("description", "") or "", height=100
                )
                ok = st.form_submit_button("更新して戻る", use_container_width=True)

            if ok:
                title2 = normalize_text(title) or "メモ"  # type: ignore
                content2 = (content or "").rstrip()
                desc2 = (description or "").strip()
                if not content2.strip():
                    st.error("内容は必須です。")
                else:
                    emb_blob = None
                    if _HAS_ST:
                        text = f"{title2}\n{desc2}\n{content2}".strip()
                        v = embed_text(text)
                        if v is not None:
                            emb_blob = to_blob(v)
                    data = dict(
                        title=title2,
                        content=content2,
                        description=desc2,
                        category=category,
                        tags=normalize_text(tags),
                        language=guess_language(content2),
                        embedding=emb_blob,
                    )
                    update_snippet(conn, int(sn["id"]), data)
                    clear_vector_cache()
                    st.toast("更新しました", icon="✅")
                    st.session_state["view"] = "search"
                    st.session_state["edit_id"] = None
                    st.rerun()

        conn.close()
        return

    tab_search, tab_add = st.tabs(["🔎 検索", "➕ 追加"])

    with tab_search:
        c1, c2 = st.columns([3, 1])
        with c1:
            q = st.text_input(
                "検索（キーワード / URL / メモ）",
                key="search_q",
                placeholder="例: FastAPI / SQL / https://... / 手順メモ",
            )
        with c2:
            topk = st.number_input(
                "意味検索 上位件数",
                min_value=3,
                max_value=20,
                value=8,
                step=1,
                disabled=not (st.session_state.get("use_sem", True) and _HAS_ST),
            )

        # ④ Header hierarchy
        st.subheader("検索結果")
        if q:
            st.caption(f"キーワード: {q}")

        rows = list_snippets(conn, q=q, limit=50)
        if rows:
            for sn in rows:
                render_snippet_card(sn, conn, allow_edit=True)
        else:
            st.info("該当なし")

        if st.session_state.get("use_sem", True) and _HAS_ST and q.strip():
            st.subheader("セマンティック検索（意味検索）")
            with st.spinner("意味検索中..."):
                sims = semantic_search(q.strip(), top_k=int(topk))
            if not sims:
                st.caption("（embedding未作成データが多い / モデル未取得 / 該当なし）")
            else:
                ids = [sid for sid, _ in sims]
                score = {sid: sc for sid, sc in sims}
                placeholders = ",".join(["?"] * len(ids))
                cur = conn.execute(
                    f"SELECT id,type,title,content,language,category,tags,description,parameters,embedding,usage_count,last_used,added_date FROM snippets WHERE id IN ({placeholders})",
                    ids,
                )
                objs = [row_to_dict(r) for r in cur.fetchall()]
                objs.sort(key=lambda x: score.get(x["id"], 0.0), reverse=True)
                for sn in objs:
                    st.markdown(f"**類似度 {int(score.get(sn['id'],0)*100)}%**")
                    render_snippet_card(sn, conn, allow_edit=True)

    with tab_add:
        st.subheader("追加（重複・類似チェック付き）")
        st.session_state.setdefault("pending_add", None)
        st.session_state.setdefault("pending_similar", None)
        st.session_state.setdefault("pending_hit_id", None)

        with st.form("add_form", clear_on_submit=False):
            title = st.text_input(
                "タイトル", key="add_title", placeholder="例: API手順 / 参考リンク"
            )
            category = st.selectbox("カテゴリ", CATEGORIES, key="add_category")
            tags = st.text_input(
                "タグ（カンマ区切り）",
                key="add_tags",
                placeholder="例: python,sql,infra",
            )
            content = st.text_area(
                "内容（URLでもメモでもOK）",
                key="add_content",
                height=180,
                placeholder="URL または メモ本文",
            )
            description = st.text_area("補足（任意）", key="add_description", height=80)
            submitted = st.form_submit_button("登録", use_container_width=True)

        if submitted:
            title2 = normalize_text(title) or "メモ"
            content2 = (content or "").rstrip()
            desc2 = (description or "").strip()
            if not content2.strip():
                st.error("内容は必須です。")
            else:
                # duplicate checks
                can = canonicalize_url(content2) if looks_like_url(content2) else ""
                fp = content_fingerprint(title2, content2, desc2, can, category, tags)

                # exact URL hit (by canonical equivalence)
                hit_id = None
                if can:
                    # best-effort: find canonical match by scanning; small N intended
                    cur = conn.execute(
                        "SELECT id, content FROM snippets WHERE content LIKE 'http%'"
                    )
                    for sid, cont in cur.fetchall():
                        if canonicalize_url(cont) == can:
                            hit_id = int(sid)
                            break

                sims = []
                best_sim = 0.0
                if _HAS_ST:
                    query_text = f"{title2}\n{desc2}\n{content2}".strip()
                    with st.spinner("類似チェック中..."):
                        sims = semantic_search(query_text, top_k=5)
                    if sims:
                        best_sim = float(sims[0][1])

                draft = dict(
                    title=title2,
                    content=content2,
                    description=desc2,
                    category=category,
                    tags=normalize_text(tags),
                    language=guess_language(content2),
                )

                # Similar/duplicate found -> confirm overwrite
                if hit_id or (sims and best_sim >= SIM_THRESHOLD):
                    st.session_state["pending_add"] = draft
                    st.session_state["pending_similar"] = sims
                    st.session_state["pending_hit_id"] = hit_id
                    st.rerun()

                # No similar -> insert directly
                emb_blob = None
                if _HAS_ST:
                    v = embed_text(f"{title2}\n{desc2}\n{content2}".strip())
                    if v is not None:
                        emb_blob = to_blob(v)
                draft["embedding"] = emb_blob  # type: ignore

                try:
                    new_id = insert_draft(conn, draft)
                    clear_vector_cache()
                    clear_add_form()
                    st.toast(f"登録しました（ID={new_id}）", icon="✅")
                    st.rerun()
                except sqlite3.IntegrityError:
                    st.warning(
                        "⚠️ 既に近い内容が登録されている可能性があります。検索で確認してください。"
                    )

        draft = st.session_state.get("pending_add")
        sims = st.session_state.get("pending_similar") or []
        hit_id = st.session_state.get("pending_hit_id")

        if draft:
            st.divider()
            st.markdown("### 類似・重複の可能性があります。確認してください。")

            overwrite_id = None
            if hit_id:
                overwrite_id = int(hit_id)
                st.info(
                    f"✅ 同一URL（正規化後）が既に登録されています（ID={overwrite_id}）"
                )

            if sims:
                best = float(sims[0][1])
                if best >= SIM_THRESHOLD:
                    st.warning(f"⚠️ 類似メモ候補があります（最上位: {int(best*100)}%）")
                with st.expander("類似候補（上位5件）", expanded=False):
                    options = []
                    for sid, sc in sims:
                        sn = fetch_snippet(conn, int(sid))
                        if sn:
                            options.append(
                                (
                                    int(sid),
                                    f"ID={sid} / {sn['title']} 類似度 {int(sc*100)}%",
                                )
                            )
                    if options:
                        overwrite_id = st.radio(
                            "上書き対象（上書きする場合）",
                            options=[o[0] for o in options],
                            format_func=lambda x: dict(options)[x],
                            index=0,
                        )
                        sn0 = fetch_snippet(conn, int(overwrite_id))  # type: ignore
                        if sn0:
                            st.caption("上書き対象プレビュー")
                            render_snippet_card(sn0, None, allow_edit=False)

            col1, col_mid, col2 = st.columns([1, 1, 1])
            with col1:
                if st.button(
                    "📝 選択候補に上書き",
                    use_container_width=True,
                    disabled=overwrite_id is None,
                ):
                    if overwrite_id is None:
                        st.error("上書き対象が選択されていません。")
                    else:
                        emb_blob = None
                        if _HAS_ST:
                            v = embed_text(
                                f"{draft['title']}\n{draft.get('description','')}\n{draft['content']}".strip()
                            )
                            if v is not None:
                                emb_blob = to_blob(v)
                        data = dict(draft)
                        data["embedding"] = emb_blob
                        update_snippet(conn, int(overwrite_id), data)
                        clear_vector_cache()
                        clear_add_form()
                        st.session_state["pending_add"] = None
                        st.session_state["pending_similar"] = None
                        st.session_state["pending_hit_id"] = None
                        st.toast(f"上書きしました（ID={overwrite_id}）", icon="✅")
                        st.rerun()
            with col_mid:
                if st.button("➕ このまま新規登録", use_container_width=True):
                    try:
                        new_id = insert_draft(conn, draft)
                        clear_vector_cache()
                        clear_add_form()
                        st.session_state["pending_add"] = None
                        st.session_state["pending_similar"] = None
                        st.session_state["pending_hit_id"] = None
                        st.toast(f"新規登録しました（ID={new_id}）", icon="✅")
                        st.rerun()
                    except sqlite3.IntegrityError:
                        st.error(
                            "完全重複（同一内容/同一URL）により登録できませんでした。内容を少し変えるか、上書きを選択してください。"
                        )

            with col2:
                if st.button("✋ キャンセル", use_container_width=True):
                    clear_add_form()
                    st.session_state["pending_add"] = None
                    st.session_state["pending_similar"] = None
                    st.session_state["pending_hit_id"] = None
                    st.toast("キャンセルしました", icon="✅")
                    st.rerun()

    conn.close()


if __name__ == "__main__":
    main()

import os
import io
import calendar
import datetime as dt
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import sqlite3
from sqlalchemy import create_engine, text

def check_pin():
    expected = st.secrets.get("APP_PIN", os.environ.get("APP_PIN"))
    if not expected:
        return True
    st.session_state.setdefault("pin_ok", False)
    if st.session_state["pin_ok"]:
        return True
    with st.form("pin_form"):
        pin = st.text_input("Enter PIN", type="password")
        ok = st.form_submit_button("Unlock")
        if ok and pin == str(expected):
            st.session_state["pin_ok"] = True
            st.success("Unlocked")
            return True
        elif ok:
            st.error("Wrong PIN")
    st.stop()

COLUMNS = ["date", "dutch", "exercise", "mindfulness", "screen_time", "notes", "ignore"]
LOCAL_DB_PATH = "habit_tracker_v2.db"
from sqlalchemy import create_engine, text

@st.cache_resource
def get_engine():
    db_url = st.secrets.get("DB_URL", os.environ.get("DB_URL"))
    try:
        if db_url:
            engine = create_engine(db_url, pool_pre_ping=True)
        else:
            engine = create_engine(
                f"sqlite:///{LOCAL_DB_PATH}",
                connect_args={"check_same_thread": False},
            )
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS habits (
                        date TEXT PRIMARY KEY,
                        dutch INTEGER NOT NULL,
                        exercise TEXT,
                        mindfulness INTEGER NOT NULL,
                        screen_time TEXT,
                        notes TEXT,
                        ignore INTEGER NOT NULL
                    )
                    """
                )
            )
        return engine
    except Exception as e:
        st.error("Could not connect to remote database. Using local SQLite instead.")
        engine = create_engine(
            f"sqlite:///{LOCAL_DB_PATH}",
            connect_args={"check_same_thread": False},
        )
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS habits (
                        date TEXT PRIMARY KEY,
                        dutch INTEGER NOT NULL,
                        exercise TEXT,
                        mindfulness INTEGER NOT NULL,
                        screen_time TEXT,
                        notes TEXT,
                        ignore INTEGER NOT NULL
                    )
                    """
                )
            )
        return engine
def init_month_df(year: int, month: int) -> pd.DataFrame:
    days = calendar.monthrange(year, month)[1]
    dates = [dt.date(year, month, d) for d in range(1, days + 1)]
    df = pd.DataFrame(
        {
            "date": dates,
            "dutch": False,
            "exercise": "",
            "mindfulness": False,
            "screen_time": "",
            "notes": "",
            "ignore": False,
        }
    )
    return df

def load_month_from_db(year: int, month: int) -> pd.DataFrame:
    base_df = init_month_df(year, month)
    engine = get_engine()
    last_day = calendar.monthrange(year, month)[1]
    start = dt.date(year, month, 1).isoformat()
    end = dt.date(year, month, last_day).isoformat()
    query = text(
        """
        SELECT date, dutch, exercise, mindfulness, screen_time, notes, ignore
        FROM habits
        WHERE date BETWEEN :start AND :end
        ORDER BY date
        """
    )
    with engine.connect() as conn:
        result = conn.execute(query, {"start": start, "end": end})
        rows = result.fetchall()
    if not rows:
        return base_df
    db_df = pd.DataFrame(rows, columns=COLUMNS)
    db_df["date"] = pd.to_datetime(db_df["date"]).dt.date
    for col in ["dutch", "mindfulness", "ignore"]:
        db_df[col] = db_df[col].astype(bool)
    db_df["exercise"] = db_df["exercise"].fillna("").astype(str)
    db_df["screen_time"] = db_df["screen_time"].fillna("").astype(str)
    db_df["notes"] = db_df["notes"].fillna("").astype(str)
    merged = base_df.merge(db_df, on="date", how="left", suffixes=("", "_db"))
    for col in COLUMNS[1:]:
        db_col = f"{col}_db"
        if db_col in merged.columns:
            merged[col] = merged[db_col].combine_first(merged[col])
            merged.drop(columns=[db_col], inplace=True)
    return merged

def save_month_to_db(df: pd.DataFrame):
    if df.empty:
        return
    engine = get_engine()
    df2 = df.copy()
    df2["date"] = df2["date"].apply(
        lambda d: d.isoformat() if isinstance(d, dt.date) else str(d)
    )
    df2["dutch"] = df2["dutch"].astype(int)
    df2["mindfulness"] = df2["mindfulness"].astype(int)
    df2["ignore"] = df2["ignore"].astype(int)
    df2["exercise"] = df2["exercise"].fillna("").astype(str)
    df2["screen_time"] = df2["screen_time"].fillna("").astype(str)
    df2["notes"] = df2["notes"].fillna("").astype(str)
    dates = sorted(df2["date"].tolist())
    start, end = dates[0], dates[-1]
    delete_q = text(
        "DELETE FROM habits WHERE date BETWEEN :start AND :end"
    )
    insert_q = text(
        """
        INSERT INTO habits (date, dutch, exercise, mindfulness, screen_time, notes, ignore)
        VALUES (:date, :dutch, :exercise, :mindfulness, :screen_time, :notes, :ignore)
        """
    )
    with engine.begin() as conn:
        conn.execute(delete_q, {"start": start, "end": end})
        conn.execute(insert_q, df2.to_dict(orient="records"))

def export_pdf(df: pd.DataFrame, title: str) -> bytes:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import mm
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        PAGE_W, PAGE_H = A4
        margin = 15 * mm
        y = PAGE_H - margin
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, y, title)
        y -= 10 * mm
        widths = [12 * mm, 24 * mm, 32 * mm, 30 * mm, 50 * mm, 70 * mm, 18 * mm]
        headers = ["Day", "Dutch", "Exercise", "Mindfulness", "Screen Time", "Notes", "Ignore"]
        row_h = 7 * mm
        x = margin
        c.setFont("Helvetica-Bold", 9)
        for w, h in zip(widths, headers):
            c.rect(x, y - row_h, w, row_h)
            c.drawString(x + 2, y - row_h + 2, h)
            x += w
        y -= row_h
        c.setFont("Helvetica", 9)
        for _, r in df.iterrows():
            if y < margin + 20 * mm:
                c.showPage()
                y = PAGE_H - margin
                c.setFont("Helvetica-Bold", 14)
                c.drawString(margin, y, title + " (cont.)")
                y -= 10 * mm
                c.setFont("Helvetica-Bold", 9)
                x = margin
                for w, h in zip(widths, headers):
                    c.rect(x, y - row_h, w, row_h)
                    c.drawString(x + 2, y - row_h + 2, h)
                    x += w
                y -= row_h
                c.setFont("Helvetica", 9)
            vals = [
                r["date"].day,
                "✔" if r["dutch"] else "✘",
                (str(r["exercise"]) or "")[:18],
                "✔" if r["mindfulness"] else "✘",
                r["screen_time"],
                (r["notes"] or "")[:60],
                "✓" if r["ignore"] else "",
            ]
            x = margin
            for w, val in zip(widths, vals):
                c.rect(x, y - row_h, w, row_h)
                c.drawString(x + 2, y - row_h + 2, str(val))
                x += w
            y -= row_h
        c.showPage()
        c.save()
        return buf.getvalue()
    except Exception:
        from PIL import Image as PILImage
        pad = 20
        row_h = 28
        header_h = 40
        cols = ["Day", "Dutch", "Exercise", "Mindfulness", "Screen Time", "Notes", "Ignore"]
        widths = [60, 100, 140, 120, 160, 260, 80]
        img_w = pad * 2 + sum(widths)
        img_h = pad * 2 + header_h + (len(df) + 1) * row_h
        img = PILImage.new("RGB", (img_w, img_h), "white")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 16)
            font_b = ImageFont.truetype("DejaVuSans.ttf", 18)
        except:
            font = ImageFont.load_default()
            font_b = font
        draw.text((pad, pad // 2), title, fill="black", font=font_b)
        x = pad
        y = pad + header_h - row_h
        for w, col in zip(widths, cols):
            draw.rectangle([x, y, x + w, y + row_h], outline="black")
            draw.text((x + 6, y + 6), col, fill="black", font=font_b)
            x += w
        y0 = pad + header_h
        for _, r in df.iterrows():
            x = pad
            vals = [
                r["date"].day,
                "✔" if r["dutch"] else "✘",
                (str(r["exercise"]) or "")[:20],
                "✔" if r["mindfulness"] else "✘",
                r["screen_time"],
                (r["notes"] or "")[:40],
                "✓" if r["ignore"] else "",
            ]
            for w, val in zip(widths, vals):
                draw.rectangle([x, y0, x + w, y0 + row_h], outline="black")
                draw.text((x + 6, y0 + 6), str(val), fill="black", font=font)
                x += w
            y0 += row_h
        out = io.BytesIO()
        img.convert("RGB").save(out, format="PDF")
        return out.getvalue()

st.set_page_config(page_title="Monthly Habit Tracker", layout="wide")

check_pin()

today = dt.date.today()
year = today.year
month = today.month

st.title(f"Monthly Habit Tracker – {calendar.month_name[month]} {year}")

key = f"df_{year}_{month}"
if key not in st.session_state:
    st.session_state[key] = load_month_from_db(year, month)

df = st.session_state[key]

cfg = {
    "date": st.column_config.DateColumn("Date", disabled=True, width="small"),
    "dutch": st.column_config.CheckboxColumn("Dutch"),
    "exercise": st.column_config.TextColumn("Exercise (e.g., run, gym, yoga)", width="medium"),
    "mindfulness": st.column_config.CheckboxColumn("Mindfulness"),
    "screen_time": st.column_config.TextColumn("Screen Time (e.g., 6h 15m)"),
    "notes": st.column_config.TextColumn("Notes"),
    "ignore": st.column_config.CheckboxColumn("Ignore"),
}

edited = st.data_editor(
    df,
    column_config=cfg,
    hide_index=True,
    use_container_width=True,
    num_rows="fixed",
    key=f"editor_{year}_{month}",
)

st.session_state[key] = edited
save_month_to_db(edited)

title = f"Monthly Habit Tracker — {calendar.month_name[month]} {year}"
pdf_bytes = export_pdf(edited, title)
st.download_button(
    "Download month as PDF",
    data=pdf_bytes,
    file_name=f"habit_{year}_{month:02d}.pdf",
    mime="application/pdf",
    use_container_width=True,
)

# 

# 

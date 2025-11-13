import os
import io
import calendar
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import sqlite3  # <- NEW

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

# ---------- NEW: SQLite helpers ----------

DB_PATH = "habit_tracker.db"

@st.cache_resource
def get_connection():
    conn = sqlite3.connect(
        DB_PATH,
        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        check_same_thread=False,
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS habits (
            date TEXT PRIMARY KEY,
            dutch INTEGER NOT NULL,
            exercise INTEGER NOT NULL,
            mindfulness INTEGER NOT NULL,
            screen_time TEXT,
            notes TEXT,
            ignore INTEGER NOT NULL
        )
        """
    )
    return conn

def init_month_df(year: int, month: int) -> pd.DataFrame:
    days = calendar.monthrange(year, month)[1]
    dates = [dt.date(year, month, d) for d in range(1, days+1)]
    df = pd.DataFrame({
        "date": dates,
        "dutch": False,
        "exercise": False,
        "mindfulness": False,
        "screen_time": "",
        "notes": "",
        "ignore": False,
    })
    return df

def load_month_from_db(year: int, month: int) -> pd.DataFrame:
    """Load a full month from DB; fallback to fresh month where missing."""
    base_df = init_month_df(year, month)
    conn = get_connection()
    last_day = calendar.monthrange(year, month)[1]
    start = dt.date(year, month, 1).isoformat()
    end = dt.date(year, month, last_day).isoformat()

    cur = conn.cursor()
    cur.execute(
        """
        SELECT date, dutch, exercise, mindfulness, screen_time, notes, ignore
        FROM habits
        WHERE date BETWEEN ? AND ?
        ORDER BY date
        """,
        (start, end),
    )
    rows = cur.fetchall()
    if not rows:
        return base_df

    db_df = pd.DataFrame(rows, columns=COLUMNS)
    db_df["date"] = pd.to_datetime(db_df["date"]).dt.date
    for col in ["dutch", "exercise", "mindfulness", "ignore"]:
        db_df[col] = db_df[col].astype(bool)

    merged = base_df.merge(db_df, on="date", how="left", suffixes=("", "_db"))
    for col in COLUMNS[1:]:
        db_col = f"{col}_db"
        if db_col in merged.columns:
            merged[col] = merged[db_col].combine_first(merged[col])
            merged.drop(columns=[db_col], inplace=True)
    return merged

def save_month_to_db(df: pd.DataFrame):
    """Upsert the whole month (simple: delete then insert)."""
    if df.empty:
        return
    conn = get_connection()
    cur = conn.cursor()

    # Determine date range of this month
    dates = sorted(df["date"].astype(str).tolist())
    start, end = dates[0], dates[-1]

    cur.execute("DELETE FROM habits WHERE date BETWEEN ? AND ?", (start, end))

    rows = []
    for _, r in df.iterrows():
        d = r["date"]
        if isinstance(d, dt.date):
            d_str = d.isoformat()
        else:
            d_str = str(d)
        rows.append(
            (
                d_str,
                int(bool(r["dutch"])),
                int(bool(r["exercise"])),
                int(bool(r["mindfulness"])),
                str(r["screen_time"] or ""),
                str(r["notes"] or ""),
                int(bool(r["ignore"])),
            )
        )

    cur.executemany(
        """
        INSERT INTO habits (date, dutch, exercise, mindfulness, screen_time, notes, ignore)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()

# ---------- existing helper functions ----------

def parse_screen_minutes(x:str) -> int:
    if x is None: return 0
    s = str(x).strip().lower()
    if not s: return 0
    try:
        if s.isdigit(): return int(s)
        h, m = 0, 0
        if "h" in s or "m" in s:
            if "h" in s:
                try: h = int(s.split("h")[0].strip())
                except: h = 0
                rest = s.split("h",1)[1]
                if "m" in rest:
                    try: m = int(rest.split("m")[0].strip())
                    except: m = 0
            elif "m" in s:
                try: m = int(s.replace("m","").strip())
                except: m = 0
            return h*60 + m
        if ":" in s:
            hh, mm = s.split(":",1)
            return int(hh)*60 + int(mm)
    except:
        return 0
    return 0

def format_minutes(mins:int) -> str:
    h = int(mins)//60; m = int(mins)%60
    return f"{h}h {m}m"

def compute_stats(df: pd.DataFrame) -> dict:
    active = df[~df["ignore"]].copy()
    if active.empty:
        return dict(dutch=0, exercise=0, mindfulness=0, n=0, avg_screen="0h 0m")
    dutch = int(active["dutch"].sum())
    exercise = int(active["exercise"].sum())
    mind = int(active["mindfulness"].sum())
    n = len(active)
    mins = active["screen_time"].apply(parse_screen_minutes).values
    avg_m = int(np.mean(mins)) if len(mins) else 0
    return dict(
        dutch=dutch, exercise=exercise, mindfulness=mind, n=n,
        dutch_pct=round(100*dutch/max(n,1)),
        exercise_pct=round(100*exercise/max(n,1)),
        mindfulness_pct=round(100*mind/max(n,1)),
        avg_screen=format_minutes(avg_m)
    )

def export_png(df: pd.DataFrame, title: str) -> bytes:
    pad = 20
    row_h = 28
    header_h = 40
    cols = ["Day","Dutch","Exercise","Mindfulness","Screen Time","Notes","Ignore"]
    widths = [60,100,100,120,160,260,80]
    img_w = pad*2 + sum(widths)
    img_h = pad*2 + header_h + (len(df)+1)*row_h
    img = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
        font_b = ImageFont.truetype("DejaVuSans.ttf", 18)
    except:
        font = ImageFont.load_default()
        font_b = font
    draw.text((pad, pad//2), title, fill="black", font=font_b)
    x = pad; y = pad + header_h - row_h
    for w, col in zip(widths, cols):
        draw.rectangle([x, y, x+w, y+row_h], outline="black")
        draw.text((x+6, y+6), col, fill="black", font=font_b)
        x += w
    y0 = pad + header_h
    for _, r in df.iterrows():
        x = pad
        vals = [
            r["date"].day,
            "✔" if r["dutch"] else "✘",
            "✔" if r["exercise"] else "✘",
            "✔" if r["mindfulness"] else "✘",
            r["screen_time"],
            (r["notes"] or "")[:40],
            "✓" if r["ignore"] else ""
        ]
        for w, val in zip(widths, vals):
            draw.rectangle([x, y0, x+w, y0+row_h], outline="black")
            draw.text((x+6, y0+6), str(val), fill="black", font=font)
            x += w
        y0 += row_h
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def export_pdf(df: pd.DataFrame, title: str) -> bytes:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import mm
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        PAGE_W, PAGE_H = A4
        margin = 15*mm
        y = PAGE_H - margin
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, y, title)
        y -= 10*mm
        widths = [12*mm, 24*mm, 24*mm, 30*mm, 50*mm, 70*mm, 18*mm]
        headers = ["Day","Dutch","Exercise","Mindfulness","Screen Time","Notes","Ignore"]
        row_h = 7*mm
        x = margin
        c.setFont("Helvetica-Bold", 9)
        for w, h in zip(widths, headers):
            c.rect(x, y-row_h, w, row_h)
            c.drawString(x+2, y-row_h+2, h)
            x += w
        y -= row_h
        c.setFont("Helvetica", 9)
        for _, r in df.iterrows():
            if y < margin + 20*mm:
                c.showPage()
                y = PAGE_H - margin
                c.setFont("Helvetica-Bold", 14)
                c.drawString(margin, y, title + " (cont.)")
                y -= 10*mm
                c.setFont("Helvetica-Bold", 9)
                x = margin
                for w, h in zip(widths, headers):
                    c.rect(x, y-row_h, w, row_h)
                    c.drawString(x+2, y-row_h+2, h)
                    x += w
                y -= row_h
                c.setFont("Helvetica", 9)
            vals = [
                r["date"].day,
                "✔" if r["dutch"] else "✘",
                "✔" if r["exercise"] else "✘",
                "✔" if r["mindfulness"] else "✘",
                r["screen_time"],
                (r["notes"] or "")[:60],
                "✓" if r["ignore"] else ""
            ]
            x = margin
            for w, val in zip(widths, vals):
                c.rect(x, y-row_h, w, row_h)
                c.drawString(x+2, y-row_h+2, str(val))
                x += w
            y -= row_h
        c.showPage(); c.save()
        return buf.getvalue()
    except Exception as e:
        png = export_png(df, title)
        from PIL import Image
        img = Image.open(io.BytesIO(png)).convert("RGB")
        out = io.BytesIO()
        img.save(out, format="PDF")
        return out.getvalue()

st.set_page_config(page_title="Monthly Habit Tracker", layout="wide")
st.title("Monthly Habit Tracker")

check_pin()

colA, colB, colC = st.columns([1,1,2])
with colA:
    year = st.number_input("Year", min_value=2000, max_value=2100, value=dt.date.today().year, step=1)
with colB:
    month = st.number_input("Month", min_value=1, max_value=12, value=dt.date.today().month, step=1)
with colC:
    st.caption("Use 'Ignore' to exclude vacation days from stats.")

key = f"df_{int(year)}_{int(month)}"

# ---------- NEW: load month from DB instead of fresh init ----------
if key not in st.session_state:
    st.session_state[key] = load_month_from_db(int(year), int(month))

df = st.session_state[key]

with st.expander("Block timeframe (vacation)"):
    days_in_month = calendar.monthrange(int(year), int(month))[1]
    start_date = st.date_input("Start", value=dt.date(int(year), int(month), 1),
                               min_value=dt.date(int(year), int(month), 1),
                               max_value=dt.date(int(year), int(month), days_in_month))
    end_date = st.date_input("End", value=dt.date(int(year), int(month), min(7, days_in_month)),
                             min_value=dt.date(int(year), int(month), 1),
                             max_value=dt.date(int(year), int(month), days_in_month))
    c1, c2 = st.columns(2)
    if c1.button("Mark Ignored"):
        mask = (df["date"] >= start_date) & (df["date"] <= end_date)
        df.loc[mask, "ignore"] = True
    if c2.button("Unmark"):
        mask = (df["date"] >= start_date) & (df["date"] <= end_date)
        df.loc[mask, "ignore"] = False

st.subheader(f"Entries for {calendar.month_name[int(month)]} {int(year)}")
cfg = {
    "date": st.column_config.DateColumn("Date", disabled=True, width="small"),
    "dutch": st.column_config.CheckboxColumn("Dutch"),
    "exercise": st.column_config.CheckboxColumn("Exercise"),
    "mindfulness": st.column_config.CheckboxColumn("Mindfulness"),
    "screen_time": st.column_config.TextColumn("Screen Time (e.g., 6h 15m)"),
    "notes": st.column_config.TextColumn("Notes"),
    "ignore": st.column_config.CheckboxColumn("Ignore"),
}
edited = st.data_editor(df, column_config=cfg, hide_index=True, use_container_width=True, num_rows="fixed")
st.session_state[key] = edited
df = edited

# ---------- NEW: Save to SQLite ----------
if st.button("💾 Save changes to database", type="primary", key="save_db"):
    save_month_to_db(df)
    st.success("Saved month to SQLite database.")

def parse_minutes_series(s: pd.Series) -> pd.Series:
    def _one(x):
        x = str(x).strip()
        if not x: return 0
        xl = x.lower()
        if xl.isdigit(): return int(xl)
        if ":" in xl:
            hh, mm = xl.split(":",1)
            return int(hh)*60 + int(mm)
        h = 0; m = 0
        if "h" in xl:
            try: h = int(xl.split("h")[0].strip())
            except: h = 0
            rest = xl.split("h",1)[1]
            if "m" in rest:
                try: m = int(rest.split("m")[0].strip())
                except: m = 0
        elif "m" in xl:
            try: m = int(xl.replace("m","").strip())
            except: m = 0
        return h*60 + m
    return s.apply(_one)

st.subheader("Stats (ignores excluded)")
active = df[~df["ignore"]].copy()
n = len(active)
d_done = int(active["dutch"].sum()) if n else 0
e_done = int(active["exercise"].sum()) if n else 0
m_done = int(active["mindfulness"].sum()) if n else 0
mins = parse_minutes_series(active["screen_time"]) if n else pd.Series([0])
avg_m = int(mins.mean()) if n else 0
def pct(val): return int(round(100*val/max(n,1)))
c1, c2, c3, c4 = st.columns(4)
c1.metric("Dutch", f"{d_done} / {n}", f"{pct(d_done)}%")
c2.metric("Exercise", f"{e_done} / {n}", f"{pct(e_done)}%")
c3.metric("Mindfulness", f"{m_done} / {n}", f"{pct(m_done)}%")
c4.metric("Avg Screen", f"{avg_m//60}h {avg_m%60}m")

st.subheader("Export")
title = f"Monthly Habit Tracker — {calendar.month_name[int(month)]} {int(year)}"
colx, coly = st.columns(2)
with colx:
    if st.button("Export as PNG"):
        png_bytes = export_png(df, title)
        st.download_button("Download PNG", data=png_bytes, file_name=f"habit_{int(year)}_{int(month):02d}.png", mime="image/png")
with coly:
    if st.button("Export as PDF"):
        pdf_bytes = export_pdf(df, title)
        st.download_button("Download PDF", data=pdf_bytes, file_name=f"habit_{int(year)}_{int(month):02d}.pdf", mime="application/pdf")

# ---------- NEW: Weekly PDF autogenerator ----------
st.subheader("Weekly PDF export (ISO weeks)")
weeks = sorted({d.isocalendar()[1] for d in df["date"]})
if weeks:
    selected_week = st.selectbox(
        "Select ISO week from this month",
        weeks,
        format_func=lambda w: f"Week {w}"
    )
    week_df = df[df["date"].apply(lambda d: d.isocalendar()[1] == selected_week)]

    if st.button("Export selected week as PDF", key="export_week_pdf"):
        week_title = (
            f"Weekly Habit Tracker — Week {selected_week} "
            f"({calendar.month_name[int(month)]} {int(year)})"
        )
        week_pdf = export_pdf(week_df, week_title)
        st.download_button(
            "Download Weekly PDF",
            data=week_pdf,
            file_name=f"habit_week{selected_week}_{int(year)}_{int(month):02d}.pdf",
            mime="application/pdf",
            key="download_week_pdf"
        )
else:
    st.info("No dates found for this month; nothing to export by week.")

st.caption(
    "Locally, data is stored in a SQLite database file `habit_tracker.db`. "
    "On Streamlit Community Cloud, local files may be cleared when the app "
    "restarts — use an external database if you need long-term online storage."
)

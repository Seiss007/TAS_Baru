"""CSS dan potongan HTML bertema web hiburan (gelap, aksen merah dan emas)."""
from __future__ import annotations

from html import escape

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@1,500&display=swap');

:root{
  --bg:#0B0B10; --surface:#14141C; --surface2:#1C1C27; --line:#2A2A38;
  --text:#F5F5F7; --muted:#9A9AB0; --red:#E50914; --gold:#F5C518; --teal:#2EC4B6;
}

.stApp{
  background:
    radial-gradient(1100px 560px at 85% -8%, rgba(229,9,20,.20), transparent 60%),
    radial-gradient(900px 480px at -8% 12%, rgba(245,197,24,.07), transparent 55%),
    var(--bg);
  color:var(--text);
  font-family:'Inter',sans-serif;
}
.stApp p,.stApp label,.stApp li,.stMarkdown,.stCaption{font-family:'Inter',sans-serif;}
h1,h2,h3,h4{font-family:'Bebas Neue','Inter',sans-serif !important; letter-spacing:.04em; font-weight:400 !important; color:var(--text);}
h3{font-size:1.9rem !important;} h4{font-size:1.5rem !important;}
#MainMenu,footer{visibility:hidden;}
header[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:2.2rem; max-width:1280px;}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"]{background:#0F0F15 !important; border-right:1px solid var(--line);}
.brand{font-family:'Bebas Neue',sans-serif; font-size:2.4rem; letter-spacing:.06em; color:var(--text); line-height:1;}
.brand span{color:var(--red);}
.brand-sub{color:var(--muted); font-size:.72rem; letter-spacing:.22em; text-transform:uppercase; margin:.2rem 0 1rem;}

/* ---------- Hero ---------- */
.hero{
  position:relative; overflow:hidden; padding:2.6rem 2.4rem 2.2rem; margin-bottom:1.6rem;
  border:1px solid var(--line); border-radius:18px;
  background:
    linear-gradient(100deg, rgba(11,11,16,.96) 20%, rgba(11,11,16,.55) 100%),
    radial-gradient(600px 300px at 90% 0%, rgba(229,9,20,.55), transparent 65%),
    radial-gradient(500px 260px at 60% 120%, rgba(245,197,24,.25), transparent 60%),
    var(--surface);
}
.hero-eyebrow{color:var(--gold); font-size:.74rem; font-weight:600; letter-spacing:.32em; text-transform:uppercase;}
.hero-title{
  font-family:'Bebas Neue',sans-serif; font-size:clamp(3.4rem,8vw,6.2rem); line-height:.92;
  letter-spacing:.03em; margin:.5rem 0 .6rem; color:#fff;
}
.hero-title em{font-style:normal; color:var(--red);}
.hero-sub{font-family:'Playfair Display',serif; font-size:1.2rem; color:#D4D4E0; max-width:640px; margin-bottom:1.2rem;}
.chips{display:flex; flex-wrap:wrap; gap:.5rem; margin-bottom:1.3rem;}
.chip{padding:.28rem .8rem; border:1px solid rgba(255,255,255,.18); border-radius:999px; font-size:.76rem; font-weight:500; color:#E8E8F0; background:rgba(255,255,255,.05);}
.chip.hot{border-color:rgba(229,9,20,.7); background:rgba(229,9,20,.16);}
.chip.gold{border-color:rgba(245,197,24,.6); color:var(--gold);}
.stats{display:flex; gap:2.4rem; flex-wrap:wrap;}
.stat b{display:block; font-family:'Bebas Neue',sans-serif; font-size:2.3rem; line-height:1; color:#fff;}
.stat span{font-size:.7rem; letter-spacing:.2em; text-transform:uppercase; color:var(--muted);}

/* ---------- Judul seksi ---------- */
.section-title{font-family:'Bebas Neue',sans-serif; font-size:1.8rem; letter-spacing:.05em; margin:1.4rem 0 .8rem; padding-left:.8rem; border-left:4px solid var(--red); line-height:1.1;}
.now-playing{display:flex; align-items:center; gap:.7rem; flex-wrap:wrap; margin:.2rem 0 1rem;}
.now-playing .np-label{font-size:.7rem; letter-spacing:.28em; color:var(--gold); text-transform:uppercase; font-weight:600;}
.now-playing .np-title{font-family:'Bebas Neue',sans-serif; font-size:2rem; letter-spacing:.04em;}

/* ---------- Kartu poster ---------- */
.poster-grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:1rem; margin:.6rem 0 1.4rem;}
.poster{position:relative; min-height:200px; padding:1.2rem; border-radius:14px; border:1px solid var(--line); overflow:hidden; transition:transform .25s ease, border-color .25s ease;}
.poster:hover{transform:translateY(-4px); border-color:rgba(255,255,255,.35);}
.poster .num{font-family:'Bebas Neue',sans-serif; font-size:4.6rem; line-height:.9; color:rgba(255,255,255,.18);}
.poster h5{font-family:'Bebas Neue',sans-serif; font-size:1.7rem; letter-spacing:.05em; margin:.3rem 0 .3rem; color:#fff; font-weight:400;}
.poster p{font-size:.85rem; color:#D8D8E4; margin:0;}
.p1{background:linear-gradient(160deg,#3b0a0d,#14141C 70%);} .p2{background:linear-gradient(160deg,#3a2f06,#14141C 70%);}
.p3{background:linear-gradient(160deg,#07302d,#14141C 70%);} .p4{background:linear-gradient(160deg,#231b52,#14141C 70%);}

/* ---------- Widget ---------- */
div[data-testid="stMetric"]{background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:14px 16px;}
div[data-testid="stMetric"] label,div[data-testid="stMetricLabel"] p{color:var(--muted) !important; font-size:.72rem !important; letter-spacing:.14em; text-transform:uppercase; font-weight:600;}
div[data-testid="stMetricValue"]{font-family:'Bebas Neue',sans-serif !important; font-size:2.2rem !important; letter-spacing:.03em; color:#fff !important;}

.stButton>button,.stFormSubmitButton>button{
  background:linear-gradient(135deg,#E50914,#B20710) !important; color:#fff !important; border:none !important;
  border-radius:6px; font-weight:700; letter-spacing:.1em; text-transform:uppercase; font-size:.82rem;
  padding:.6rem 1.2rem; transition:transform .15s ease, box-shadow .15s ease;
}
.stButton>button:hover,.stFormSubmitButton>button:hover{transform:translateY(-1px); box-shadow:0 8px 24px rgba(229,9,20,.35);}
.stButton>button:disabled{opacity:.5;}

.stTabs [data-baseweb="tab-list"]{gap:1.4rem; border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{color:var(--muted); font-weight:600; letter-spacing:.12em; text-transform:uppercase; font-size:.8rem; padding:.7rem .2rem; background:transparent;}
.stTabs [aria-selected="true"]{color:#fff !important;}
.stTabs [data-baseweb="tab-highlight"]{background:var(--red) !important; height:3px;}

div[data-baseweb="select"]>div,.stTextInput input,.stTextArea textarea,.stNumberInput input{background:var(--surface2) !important; border-color:var(--line) !important; color:var(--text) !important;}
div[data-testid="stForm"]{background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:1.2rem;}
div[data-testid="stExpander"]{background:var(--surface); border:1px solid var(--line); border-radius:12px;}
hr{border-color:var(--line) !important;}

/* ---------- Komentar ---------- */
.comment{background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:1rem 1.1rem; margin-bottom:.8rem;}
.comment:hover{border-color:#3a3a4d;}
.c-head{display:flex; align-items:center; gap:.75rem; margin-bottom:.55rem;}
.avatar{width:38px; height:38px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-family:'Bebas Neue',sans-serif; font-size:1.3rem; background:linear-gradient(135deg,#E50914,#F5C518); color:#0B0B10; flex:none;}
.c-name{font-weight:600; font-size:.95rem;}
.c-meta{font-size:.74rem; color:var(--muted);}
.c-body{font-size:.93rem; line-height:1.55; color:#E4E4EE; word-wrap:break-word;}
.badge{margin-left:auto; padding:.18rem .65rem; border-radius:999px; font-size:.68rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase;}
.badge.pos{background:rgba(46,196,182,.15); color:var(--teal); border:1px solid rgba(46,196,182,.5);}
.badge.neu{background:rgba(154,154,176,.12); color:#B9B9CC; border:1px solid rgba(154,154,176,.4);}
.badge.neg{background:rgba(229,9,20,.14); color:#FF5A62; border:1px solid rgba(229,9,20,.55);}
.status{display:flex; align-items:center; gap:.5rem; font-size:.8rem; color:var(--muted); margin-bottom:.4rem;}
.dot{width:8px; height:8px; border-radius:50%; background:var(--teal); box-shadow:0 0 0 0 rgba(46,196,182,.7); animation:pulse 1.8s infinite;}
.dot.off{background:#7a7a8c; animation:none;} .dot.warn{background:var(--gold); animation:none;}
@keyframes pulse{70%{box-shadow:0 0 0 8px rgba(46,196,182,0);}100%{box-shadow:0 0 0 0 rgba(46,196,182,0);}}
.live{display:inline-flex; align-items:center; gap:.4rem; color:#FF5A62; font-size:.7rem; font-weight:700; letter-spacing:.2em;}
.live i{width:8px; height:8px; border-radius:50%; background:var(--red); animation:pulse 1.4s infinite;}
</style>
"""


def hero_html(n_datasets: int, n_realtime: int) -> str:
    chips = [
        ("Regresi", ""), ("Klasifikasi", ""), ("K-Means Clustering", "hot"),
        ("Deep Learning · PyTorch", "hot"), ("UCI Repository", "gold"),
        ("Real-time Data", "gold"), ("Groq LLM", ""), ("MongoDB", ""),
    ]
    chip_html = "".join(f'<span class="chip {c}">{escape(t)}</span>' for t, c in chips)
    return f"""
<div class="hero">
  <div class="hero-eyebrow">Machine Learning Lab</div>
  <div class="hero-title">ML <em>STUDIO</em></div>
  <div class="hero-sub">Latih model, temukan pola tersembunyi, lalu diskusikan hasilnya bersama komunitas.</div>
  <div class="chips">{chip_html}</div>
  <div class="stats">
    <div class="stat"><b>{n_datasets}</b><span>Dataset</span></div>
    <div class="stat"><b>4</b><span>Mode analisis</span></div>
    <div class="stat"><b>{n_realtime}</b><span>Sumber real-time</span></div>
    <div class="stat"><b>3</b><span>Algoritma clustering</span></div>
  </div>
</div>
"""


def poster_grid_html() -> str:
    items = [
        ("p1", "01", "Regresi", "Prediksi nilai kontinu: harga rumah, suhu udara, magnitudo gempa."),
        ("p2", "02", "Klasifikasi", "Kenali kelas: penyakit jantung, varietas gandum, keaslian uang."),
        ("p3", "03", "Clustering", "K-Means, Agglomerative, dan DBSCAN dengan peta PCA dan silhouette."),
        ("p4", "04", "Deep Learning", "MLP PyTorch dengan kurva loss yang diperbarui setiap epoch."),
    ]
    cards = "".join(
        f'<div class="poster {c}"><div class="num">{n}</div><h5>{escape(t)}</h5><p>{escape(d)}</p></div>'
        for c, n, t, d in items
    )
    return f'<div class="poster-grid">{cards}</div>'


def section_title(text: str) -> str:
    return f'<div class="section-title">{escape(text)}</div>'

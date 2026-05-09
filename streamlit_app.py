import os, sys, subprocess, tempfile, time, traceback
import streamlit as st
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Tennis AI Analysis", page_icon="🎾",
                   layout="wide", initial_sidebar_state="expanded")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);min-height:100vh;}
#MainMenu,footer{visibility:hidden;}
/* Keep header visible so sidebar collapse/expand arrow shows */
header{visibility:visible !important;background:transparent !important;}
header [data-testid="stHeader"]{background:transparent !important;}
.block-container{padding-top:1rem;padding-bottom:2rem;}
.hero-banner{background:linear-gradient(135deg,rgba(0,212,255,.15),rgba(144,0,255,.15));border:1px solid rgba(0,212,255,.3);border-radius:20px;padding:2.5rem;text-align:center;margin-bottom:2rem;}
.hero-title{font-size:3rem;font-weight:800;background:linear-gradient(90deg,#00d4ff,#9000ff,#ff6b6b);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin:0;}
.hero-subtitle{color:rgba(255,255,255,.7);font-size:1.1rem;margin-top:.5rem;font-weight:300;}
.stat-card{background:linear-gradient(135deg,rgba(255,255,255,.08),rgba(255,255,255,.03));border:1px solid rgba(255,255,255,.15);border-radius:16px;padding:1.5rem;text-align:center;backdrop-filter:blur(10px);}
.stat-value{font-size:2rem;font-weight:700;color:#00d4ff;display:block;}
.stat-label{font-size:.8rem;color:rgba(255,255,255,.5);text-transform:uppercase;letter-spacing:1px;margin-top:.25rem;}
.player-card{background:linear-gradient(135deg,rgba(255,255,255,.07),rgba(255,255,255,.02));border-radius:16px;padding:1.5rem;border:1px solid rgba(255,255,255,.12);}
.player-name{font-size:1.2rem;font-weight:700;margin-bottom:1rem;padding-bottom:.5rem;border-bottom:2px solid;}
.p1-accent{border-color:#00d4ff;color:#00d4ff;} .p2-accent{border-color:#ff6b6b;color:#ff6b6b;}
.metric-row{display:flex;justify-content:space-between;align-items:center;padding:.4rem 0;border-bottom:1px solid rgba(255,255,255,.05);}
.metric-key{color:rgba(255,255,255,.6);font-size:.85rem;} .metric-val{color:white;font-weight:600;font-size:.95rem;}
.section-header{font-size:1.4rem;font-weight:700;color:white;margin:1.5rem 0 1rem 0;}
.stButton>button{background:linear-gradient(135deg,#00d4ff,#9000ff)!important;color:white!important;border:none!important;border-radius:12px!important;font-size:1.1rem!important;font-weight:600!important;padding:.75rem 2rem!important;width:100%!important;box-shadow:0 4px 20px rgba(0,212,255,.3)!important;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1a1a2e,#16213e)!important;border-right:1px solid rgba(255,255,255,.1);}
[data-testid="stSidebar"] label{color:#ffffff !important;}
[data-testid="stSidebar"] .stRadio label p{color:#ffffff !important;font-size:.92rem !important;font-weight:500 !important;}
[data-testid="stSidebar"] .stRadio>div{gap:.4rem;}
/* Target only text nodes — NOT divs (which breaks toggle button) */
[data-testid="stSidebar"] .stMarkdown p{color:#e0e0e0 !important;}
[data-testid="stSidebar"] .stMarkdown span{color:#e0e0e0 !important;}
/* Sidebar collapse/expand arrow — white and always visible */
[data-testid="collapsedControl"]{background:rgba(255,255,255,.15) !important;border-radius:0 8px 8px 0 !important;border:1px solid rgba(255,255,255,.4) !important;}
[data-testid="collapsedControl"]:hover{background:rgba(255,255,255,.3) !important;}
[data-testid="collapsedControl"] svg{fill:#ffffff !important;stroke:#ffffff !important;}
[data-testid="collapsedControl"] button{color:#ffffff !important;}
[data-testid="collapsedControl"] span{color:#ffffff !important;}
.stProgress>div>div>div>div{background:linear-gradient(90deg,#00d4ff,#9000ff)!important;}
.badge{display:inline-block;background:linear-gradient(135deg,#00d4ff22,#9000ff22);border:1px solid rgba(0,212,255,.4);border-radius:20px;padding:.2rem .8rem;font-size:.75rem;color:#00d4ff;font-weight:600;}
.warn-box{background:rgba(255,170,0,.1);border:1px solid rgba(255,170,0,.3);border-radius:10px;padding:.75rem 1rem;color:#ffaa00;font-size:.85rem;}
</style>
""", unsafe_allow_html=True)

PYTHON = sys.executable
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(PROJECT_DIR, "pipeline_runner.py")

def safe(val):
    try:
        v = float(val)
        return f"{v:.1f}" if not np.isnan(v) else "—"
    except Exception:
        return "—"

def run_pipeline_subprocess(input_path, progress_bar, status_text, mode="auto"):
    output_dir = tempfile.mkdtemp()
    cmd = [PYTHON, RUNNER, input_path, output_dir, mode]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding='utf-8', errors='replace',
                            cwd=PROJECT_DIR)
    out_video, stats_file, used_stubs = None, None, False

    for line in proc.stdout:
        line = line.strip()
        if line.startswith("PROGRESS:"):
            parts = line.split(":", 2)
            pct  = int(parts[1])
            msg  = parts[2] if len(parts) > 2 else ""
            progress_bar.progress(pct)
            status_text.markdown(
                f"<div style='color:#00d4ff;font-size:1rem;font-weight:600;padding:.4rem 0;'>⚙️ {msg}</div>",
                unsafe_allow_html=True
            )
        elif line.startswith("OUTPUT_VIDEO:"):
            out_video = line.split(":", 1)[1]
        elif line.startswith("STATS_FILE:"):
            stats_file = line.split(":", 1)[1]
        elif line.startswith("USED_STUBS:"):
            used_stubs = line.split(":", 1)[1].strip().lower() == "true"

    proc.wait()
    if proc.returncode != 0:
        stderr = proc.stderr.read()
        raise RuntimeError(stderr)

    df = pd.read_csv(stats_file) if stats_file and os.path.exists(stats_file) else pd.DataFrame()
    return out_video, df, used_stubs


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0;'>
        <div style='font-size:3rem;'>🎾</div>
        <div style='font-size:1.1rem;font-weight:700;color:white;'>Tennis AI</div>
        <div style='font-size:.75rem;color:rgba(255,255,255,.5);'>Analysis System</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='color:rgba(255,255,255,.7);font-size:.85rem;font-weight:600;letter-spacing:1px;'>⚙️ MODELS</div>", unsafe_allow_html=True)
    models_ok = os.path.exists("models/yolo5_last.pt") and os.path.exists("models/keypoints_model.pth")
    if models_ok:
        st.markdown("<div style='color:#ffffff;font-size:.9rem;padding:.3rem 0;'>🟢 <strong>Ball Tracker</strong> — YOLOv5</div>", unsafe_allow_html=True)
        st.markdown("<div style='color:#ffffff;font-size:.9rem;padding:.3rem 0;'>🟢 <strong>Player Tracker</strong> — YOLOv8x</div>", unsafe_allow_html=True)
        st.markdown("<div style='color:#ffffff;font-size:.9rem;padding:.3rem 0;'>🟢 <strong>Court Detector</strong> — KeypointsCNN</div>", unsafe_allow_html=True)
    else:
        st.error("⚠️ Model files missing in `models/` folder")
    st.markdown("---")
    st.markdown("<div style='color:rgba(255,255,255,.7);font-size:.85rem;font-weight:600;letter-spacing:1px;'>🔧 INFERENCE MODE</div>", unsafe_allow_html=True)
    inference_mode = st.radio(
        "Mode",
        ["🚀 Auto (smart stub)", "🔬 Full Inference (any video)"],
        help="Auto uses pre-computed stubs if the video matches. Full Inference runs YOLO on every frame — slower but works for any video.",
        label_visibility="collapsed"
    )
    if inference_mode.startswith("🚀"):
        st.markdown("<div style='color:rgba(0,212,255,.8);font-size:.75rem;'>⚡ Fast mode — uses cached detections if video matches</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='color:rgba(255,107,107,.8);font-size:.75rem;'>🐢 Slow but works for any tennis video (~10-20 min)</div>", unsafe_allow_html=True)

    for icon, txt in [("1️⃣","Upload tennis match video"),("2️⃣","AI detects players & ball"),
                       ("3️⃣","Court lines mapped"),("4️⃣","Stats computed"),("5️⃣","Download output video")]:
        st.markdown(f"<div style='color:#e0e0e0;font-size:.83rem;padding:.25rem 0;font-weight:500;'>{icon} {txt}</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='font-size:.75rem;color:rgba(255,255,255,.3);text-align:center;'>BTech Capstone · Tennis Analysis · YOLOv8</div>", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1 class="hero-title">🎾 Tennis AI Analysis</h1>
    <p class="hero-subtitle">Upload a match video · AI detects players, ball & court · Get live statistics</p>
    <span class="badge">YOLOv8 · YOLOv5 · CNN Court Detection</span>
</div>""", unsafe_allow_html=True)

# ── Upload ─────────────────────────────────────────────────────────────────────
col_up, col_tip = st.columns([2,1])
with col_up:
    st.markdown('<div class="section-header">📤 Upload Match Video</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload your tennis match video",
                                     type=["mp4","avi","mov"],
                                     help="10–30 sec clips work best.",
                                     label_visibility="collapsed")
with col_tip:
    st.markdown('<div class="section-header">📌 Tips</div>', unsafe_allow_html=True)
    st.markdown('<div class="warn-box">⏱️ <strong>Best results with:</strong><br>• 10–30 sec clips<br>• Full court visible<br>• MP4 / AVI format<br>• Good lighting</div>', unsafe_allow_html=True)

# ── Preview & Run ──────────────────────────────────────────────────────────────
if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read()); tfile.flush()
    input_path = tfile.name

    st.markdown('<div class="section-header">🎬 Input Preview</div>', unsafe_allow_html=True)
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur    = total_frames/fps if fps>0 else 0
    cap.release()

    c1,c2,c3,c4 = st.columns(4)
    for col, val, lbl in [(c1,total_frames,"Total Frames"),(c2,f"{fps:.0f}","FPS"),
                           (c3,f"{dur:.1f}s","Duration"),(c4,f"{width}×{height}","Resolution")]:
        with col:
            st.markdown(f'<div class="stat-card"><span class="stat-value">{val}</span><div class="stat-label">{lbl}</div></div>', unsafe_allow_html=True)
    st.video(input_path)
    st.markdown("<br>", unsafe_allow_html=True)

    if not models_ok:
        st.error("❌ Cannot process: model files missing from `models/` folder.")
    else:
        if st.button("🚀  Analyse Match", use_container_width=True):
            st.markdown('<div class="section-header">⚙️ Processing Pipeline</div>', unsafe_allow_html=True)
            progress_bar = st.progress(0)
            status_text  = st.empty()
            try:
                t0 = time.time()
                mode_flag = "auto" if inference_mode.startswith("🚀") else "full"
                out_video, stats_df, used_stubs = run_pipeline_subprocess(
                    input_path, progress_bar, status_text, mode=mode_flag)
                elapsed = time.time() - t0
                status_text.markdown(
                    f"<div style='color:#00ff88;font-size:1rem;font-weight:700;padding:.4rem 0;'>✅ Done! Processed in {elapsed:.1f} seconds.</div>",
                    unsafe_allow_html=True
                )
                if used_stubs:
                    st.info("📦 **Stub mode** — pre-computed detections used. Court detection & stats ran live.")

                # Output video
                st.markdown('<div class="section-header">🎥 Annotated Output</div>', unsafe_allow_html=True)
                if out_video and os.path.exists(out_video):
                    with open(out_video,"rb") as f: vbytes = f.read()
                    st.video(vbytes)
                    st.download_button("⬇️  Download Annotated Video", vbytes,
                                       "tennis_analysis_output.mp4", "video/mp4",
                                       use_container_width=True)

                # Stats dashboard
                if not stats_df.empty:
                    st.markdown('<div class="section-header">📊 Match Statistics</div>', unsafe_allow_html=True)
                    last = stats_df.iloc[-1]
                    s1,s2,s3,s4 = st.columns(4)
                    with s1: st.markdown(f'<div class="stat-card"><span class="stat-value">{int(last.get("player_1_number_of_shots",0)+last.get("player_2_number_of_shots",0))}</span><div class="stat-label">Total Shots</div></div>', unsafe_allow_html=True)
                    with s2: st.markdown(f'<div class="stat-card"><span class="stat-value" style="color:#00d4ff">{int(last.get("player_1_number_of_shots",0))}</span><div class="stat-label">Player 1 Shots</div></div>', unsafe_allow_html=True)
                    with s3: st.markdown(f'<div class="stat-card"><span class="stat-value" style="color:#ff6b6b">{int(last.get("player_2_number_of_shots",0))}</span><div class="stat-label">Player 2 Shots</div></div>', unsafe_allow_html=True)
                    with s4: st.markdown(f'<div class="stat-card"><span class="stat-value">{safe(last.get("player_1_average_shot_speed",0))}</span><div class="stat-label">Avg Ball Speed km/h</div></div>', unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    pc1,pc2 = st.columns(2)
                    for col, pid, acc in [(pc1,1,"p1-accent"),(pc2,2,"p2-accent")]:
                        emoji = "🔵" if pid==1 else "🔴"
                        with col:
                            st.markdown(f"""
                            <div class="player-card">
                                <div class="player-name {acc}">{emoji} Player {pid}</div>
                                <div class="metric-row"><span class="metric-key">Shots</span><span class="metric-val">{int(last.get(f'player_{pid}_number_of_shots',0))}</span></div>
                                <div class="metric-row"><span class="metric-key">Last Shot Speed</span><span class="metric-val">{safe(last.get(f'player_{pid}_last_shot_speed',0))} km/h</span></div>
                                <div class="metric-row"><span class="metric-key">Avg Shot Speed</span><span class="metric-val">{safe(last.get(f'player_{pid}_average_shot_speed',0))} km/h</span></div>
                                <div class="metric-row"><span class="metric-key">Last Move Speed</span><span class="metric-val">{safe(last.get(f'player_{pid}_last_player_speed',0))} km/h</span></div>
                                <div class="metric-row"><span class="metric-key">Avg Move Speed</span><span class="metric-val">{safe(last.get(f'player_{pid}_average_player_speed',0))} km/h</span></div>
                            </div>""", unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # === PHASE 1: Speed Graph & CSV Download ===
                    st.markdown('<div class="section-header">📈 Player Speed Over Time</div>', unsafe_allow_html=True)
                    if 'player_1_last_player_speed' in stats_df.columns and 'player_2_last_player_speed' in stats_df.columns:
                        chart_df = stats_df[['frame_num', 'player_1_last_player_speed', 'player_2_last_player_speed']].set_index('frame_num')
                        chart_df.columns = ['Player 1 Speed (km/h)', 'Player 2 Speed (km/h)']
                        st.line_chart(chart_df, color=["#00d4ff", "#ff6b6b"])

                    # === PHASE 2: Heatmap & Dominance ===
                    st.markdown('<div class="section-header">🔥 Court Dominance & Heatmap</div>', unsafe_allow_html=True)
                    hm_col1, hm_col2 = st.columns([1, 1])
                    
                    with hm_col1:
                        st.markdown("<br><h3 style='color:white;'>Court Coverage (Work Rate)</h3>", unsafe_allow_html=True)
                        p1_dist = stats_df['player_1_total_player_speed'].max() if 'player_1_total_player_speed' in stats_df.columns else 0
                        p2_dist = stats_df['player_2_total_player_speed'].max() if 'player_2_total_player_speed' in stats_df.columns else 0
                        total_dist = p1_dist + p2_dist if (p1_dist + p2_dist) > 0 else 1
                        
                        p1_pct = int((p1_dist/total_dist)*100)
                        p2_pct = int((p2_dist/total_dist)*100)
                        
                        st.markdown(f"<span style='color:#e0e0e0;'>**Player 1 (🔵):** {p1_pct}%</span>", unsafe_allow_html=True)
                        st.progress(float(p1_dist/total_dist))
                        st.markdown(f"<span style='color:#e0e0e0;'>**Player 2 (🔴):** {p2_pct}%</span>", unsafe_allow_html=True)
                        st.progress(float(p2_dist/total_dist))
                        
                        st.markdown("<p style='color:gray;font-size:0.8rem;margin-top:20px;'>Calculated based on total distance traversed across the court.</p>", unsafe_allow_html=True)
                        
                    with hm_col2:
                        if 'player_1_x' in stats_df.columns:
                            fig, ax = plt.subplots(figsize=(4, 6))
                            # Draw simplified court boundaries
                            ax.plot([0, 250, 250, 0, 0], [0, 0, 500, 500, 0], color='white', lw=2) # Outline
                            ax.plot([0, 250], [250, 250], color='white', lw=2, linestyle='--') # Net
                            
                            # Filter out (0,0) fallback values
                            p1_data = stats_df[(stats_df['player_1_x'] > 0) & (stats_df['player_1_y'] > 0)]
                            p2_data = stats_df[(stats_df['player_2_x'] > 0) & (stats_df['player_2_y'] > 0)]
                            
                            if len(p1_data) > 0 and len(p2_data) > 0:
                                # Normalize coordinates to fit the 0-250 and 0-500 mock court
                                min_x = min(p1_data['player_1_x'].min(), p2_data['player_2_x'].min())
                                min_y = min(p1_data['player_1_y'].min(), p2_data['player_2_y'].min())
                                
                                p1_x_norm = p1_data['player_1_x'] - min_x
                                p1_y_norm = p1_data['player_1_y'] - min_y
                                p2_x_norm = p2_data['player_2_x'] - min_x
                                p2_y_norm = p2_data['player_2_y'] - min_y

                                sns.kdeplot(x=p1_x_norm, y=p1_y_norm, fill=True, cmap="Blues", alpha=0.6, ax=ax, thresh=0.1)
                                sns.kdeplot(x=p2_x_norm, y=p2_y_norm, fill=True, cmap="Reds", alpha=0.6, ax=ax, thresh=0.1)
                            
                            ax.set_xlim(-50, 300)
                            ax.set_ylim(-50, 550)
                            ax.set_facecolor('#1a1a2e')
                            fig.patch.set_facecolor('#1a1a2e')
                            ax.axis('off')
                            st.pyplot(fig)
                        else:
                            st.warning("⚠️ Heatmap data missing. Please re-run the pipeline in 'Any Video' mode.")

                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # CSV Download Button
                    csv = stats_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Raw Stats Data (CSV)",
                        data=csv,
                        file_name='tennis_match_stats.csv',
                        mime='text/csv',
                        use_container_width=True
                    )

                    with st.expander("📋 Raw Statistics Table"):
                        st.dataframe(stats_df, use_container_width=True, height=300)

            except Exception as e:
                st.error(f"❌ **Processing failed:** {str(e)}")
                with st.expander("🔍 Full Error Details"):
                    st.code(traceback.format_exc(), language="python")
                st.info("💡 Make sure model files exist and the video is a valid tennis match clip.")
else:
    st.markdown("""
    <div style="text-align:center;padding:3rem;color:rgba(255,255,255,.3);">
        <div style="font-size:5rem;">🎾</div>
        <div style="font-size:1.1rem;margin-top:1rem;">Upload a tennis match video to get started</div>
        <div style="font-size:.85rem;margin-top:.5rem;">Supports MP4 · AVI · MOV</div>
    </div>""", unsafe_allow_html=True)

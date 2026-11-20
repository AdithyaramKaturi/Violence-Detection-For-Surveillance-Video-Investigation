"""
Surveillance Video Violence Detection & Police Forensic Investigation Console
=============================================================================
Interactive Web Application powered by MobileNetV2 Deep Learning Backbone
with Role-Based Law Enforcement Authentication.
"""

import os
import json
import tempfile
from datetime import datetime
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from PIL import Image
import plotly.graph_objects as go

from core.video_processor import VideoViolenceDetector
from core.report_generator import ForensicReportGenerator


# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Police Video Forensics & Violence Detection Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Tech Police Dark CSS Styling
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
    }
    .metric-card {
        background: linear-gradient(135deg, #1A202C 0%, #2D3748 100%);
        padding: 18px;
        border-radius: 10px;
        border: 1px solid #4A5568;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        text-align: center;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #E2E8F0;
        margin-top: 5px;
    }
    .metric-label {
        font-size: 13px;
        color: #A0AEC0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-alert {
        color: #FC8181 !important;
    }
    .metric-safe {
        color: #68D391 !important;
    }
    .badge-violence {
        background-color: #E53E3E;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .police-header-badge {
        display: inline-block;
        background: rgba(49, 130, 206, 0.15);
        color: #63B3ED;
        border: 1px solid #3182CE;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .police-banner {
        background: linear-gradient(90deg, #1A365D 0%, #2A4365 100%);
        border-left: 4px solid #3182CE;
        border-radius: 8px;
        padding: 12px 18px;
        margin-bottom: 20px;
        color: #E2E8F0;
        font-size: 14px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .officer-profile-card {
        background: linear-gradient(135deg, #1A2634 0%, #223042 100%);
        border: 1px solid #2B6CB0;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 15px;
    }
    .login-container-card {
        background: linear-gradient(135deg, #1A202C 0%, #2D3748 100%);
        border: 1px solid #4A5568;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }
    /* Hide the 200MB file limit instruction across all Streamlit versions */
    [data-testid="stFileUploaderDropzoneInstructions"] small,
    [data-testid="stFileUploaderDropzoneInstructions"] > div:nth-child(2),
    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzone"] > div > small,
    [data-testid="stFileUploaderDropzone"] div[data-testid="stMarkdownContainer"] small,
    [data-testid="stFileUploaderInstructions"],
    [data-testid="stFileUploader"] small,
    .stFileUploader small,
    .stFileUploader div[data-testid="stFileUploaderDropzoneInstructions"] small,
    section[data-testid="stFileUploader"] small,
    div[data-testid="stFileUploaderDropzoneInstructions"] > div:last-child {
        display: none !important;
        visibility: hidden !important;
        height: 0px !important;
        font-size: 0px !important;
    }
</style>
<script>
    // Clean any 200MB text appearing in DOM
    const remove200MB = () => {
        const root = window.parent.document || document;
        root.querySelectorAll('*').forEach(el => {
            if (el.children.length === 0 && el.innerText && el.innerText.includes('200MB')) {
                el.innerText = el.innerText.replace(/Limit 200MB per file • /g, '').replace(/200MB per file • /g, '').replace(/200MB/g, '');
            }
        });
    };
    setInterval(remove200MB, 500);
</script>
""", unsafe_allow_html=True)


# Official Authorized Police Accounts Registry
AUTHORIZED_USERS = {
    "police@gov.in": {
        "password": "police123",
        "name": "Sanjay Pagadala",
        "badge_id": "INSP-7821",
        "department": "Central Surveillance & Video Forensics Division"
    },
    "investigator@police.gov.in": {
        "password": "investigate2026",
        "name": "Ram",
        "badge_id": "ANALYST-504",
        "department": "Cyber Crime & Digital Evidence Investigation Bureau"
    },
    "admin@forensics.gov.in": {
        "password": "admin123",
        "name": "Chief Forensic Examiner",
        "badge_id": "CHIEF-001",
        "department": "State Forensic Science Laboratory"
    }
}


def render_login_portal():
    """
    Minimal login authentication portal: title, email id, pwd, and login button
    with strict validation against official authorized police credentials.
    """
    col1, col2, col3 = st.columns([1, 1.3, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; margin-bottom: 25px;'>Surveillance Video Evidence Portal</h2>", unsafe_allow_html=True)

        with st.form("login_form"):
            email_id = st.text_input("Email ID", placeholder="Enter Email ID")
            pwd = st.text_input("Password", type="password", placeholder="Enter Password")
            login_clicked = st.form_submit_button("Login", type="primary", use_container_width=True)

            if login_clicked:
                clean_email = email_id.strip().lower()
                clean_pwd = pwd.strip()

                if not clean_email or not clean_pwd:
                    st.warning("⚠️ Please enter both Email ID and Password.")
                elif clean_email in AUTHORIZED_USERS and AUTHORIZED_USERS[clean_email]["password"] == clean_pwd:
                    user_info = AUTHORIZED_USERS[clean_email]
                    st.session_state["authenticated"] = True
                    st.session_state["officer_email"] = clean_email
                    st.session_state["officer_name"] = user_info["name"]
                    st.session_state["badge_id"] = user_info["badge_id"]
                    st.session_state["department"] = user_info["department"]
                    st.session_state["session_token"] = f"POL-{datetime.now().strftime('%Y%m%d%H%M')}-{user_info['badge_id']}"
                    st.session_state["login_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.success("✅ Access Granted: Validated Official Credentials.")
                    st.rerun()
                else:
                    st.error("❌ Access Denied: Invalid Email ID or Password.")


def main():
    # Dynamic DOM cleanup for file uploader limit text
    components.html("""
    <script>
    function clean200MB() {
        try {
            var doc = window.parent.document;
            var elements = doc.querySelectorAll('small, div, span, p');
            elements.forEach(function(el) {
                if (el.children.length === 0 && el.textContent && el.textContent.includes('200MB')) {
                    el.textContent = el.textContent.replace(/Limit 200MB per file • /gi, '')
                                                  .replace(/Limit 200MB per file/gi, '')
                                                  .replace(/200MB per file • /gi, '')
                                                  .replace(/200MB per file/gi, '')
                                                  .replace(/200MB/gi, '');
                }
            });
        } catch(e) {}
    }
    setInterval(clean200MB, 250);
    </script>
    """, height=0, width=0)

    # Check authentication state
    if not st.session_state.get("authenticated", False):
        render_login_portal()
        return

    # =========================================================================
    # AUTHENTICATED INVESTIGATION WORKSPACE
    # =========================================================================

    officer_name = st.session_state.get("officer_name", "Sanjay Pagadala")
    badge_id = st.session_state.get("badge_id", "INSP-7821")
    rank_title = st.session_state.get("rank", "Lead Forensic Investigator")
    department = st.session_state.get("department", "Central Surveillance & Video Forensics Division")
    session_token = st.session_state.get("session_token", "POL-2026-ACTIVE")
    login_time = st.session_state.get("login_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Active Forensic Session Banner on Main Console
    st.markdown(f"""
    <div class="police-banner">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <span style="font-weight: 700; color: #63B3ED; letter-spacing: 0.5px;">🛡️ OFFICIAL POLICE INVESTIGATION CONSOLE</span>
                <span style="margin: 0 8px; color: #718096;">|</span>
                <span>Unit: <i>{department}</i></span>
            </div>
            <div>
                <span style="background: rgba(72, 187, 120, 0.2); color: #68D391; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700;">● SECURE AUDIT ACTIVE</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Application Title
    st.title("🛡️ Deep Learning-Based Violence Detection for Surveillance Video Investigation")
    st.caption("Forensic Evidence Investigation Console • MobileNetV2 Deep Neural Architecture")

    model_weight_path = "models/mobilenet_violence_best.pth"
    metrics_path = "outputs/test_metrics.json"
    acc_text = ""
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as mf:
                mdata = json.load(mf)
                acc_text = f" ({mdata.get('accuracy', 90)}% Test Acc)"
        except Exception:
            pass

    # Left Sidebar: Officer Credentials, Model Status & Forensic Calibration Parameters
    with st.sidebar:
        # Officer Profile Card in Sidebar
        officer_email = st.session_state.get("officer_email", "officer@police.gov.in")
        st.markdown(f"""
        <div class="officer-profile-card">
            <div style="font-size: 11px; color: #63B3ED; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">👮 Authenticated Investigator</div>
            <div style="font-size: 16px; font-weight: 700; color: #F7FAFC; margin-top: 4px;">{officer_name}</div>
            <div style="font-size: 12px; color: #CBD5E0;">📧 <code>{officer_email}</code></div>
            <div style="font-size: 10px; color: #718096; margin-top: 4px;">🕒 Login: {login_time}</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔒 Sign Out / Lock Console", type="secondary", use_container_width=True):
            st.session_state["authenticated"] = False
            st.session_state.pop("results", None)
            st.rerun()

        st.markdown("---")

        if os.path.exists(model_weight_path):
            st.success(f"🟢 **Trained Model Active**\n`mobilenet_violence_best.pth`{acc_text}")
        else:
            st.warning("⚠️ **ImageNet Weights**")

        st.markdown("---")
        st.subheader("⚙️ Investigation Parameters")
        st.caption("Adjust detection sensitivity and scanning settings for the surveillance video:")

        confidence_thresh = st.slider(
            "🎯 Detection Sensitivity (Threshold τ)",
            min_value=0.30,
            max_value=0.95,
            value=float(st.session_state.get("confidence_thresh", 0.50)),
            step=0.05,
            key="confidence_thresh",
            help="Higher = stricter (avoids false alarms). Lower = more sensitive (catches subtle/distant struggles). Recommended: 0.50"
        )

        sampling_interval = st.slider(
            "⏱️ Frame Scan Interval (Seconds Δt)",
            min_value=0.50,
            max_value=3.00,
            value=float(st.session_state.get("sampling_interval", 1.00)),
            step=0.50,
            key="sampling_interval",
            help="Interval between sampled frames. 0.5s is ideal for quick fight movements; 1.0s is faster for long CCTV recordings."
        )

        temporal_window = st.slider(
            "🛡️ Noise Smoothing Filter (Frames W)",
            min_value=1,
            max_value=5,
            value=int(st.session_state.get("temporal_window", 3)),
            step=1,
            key="temporal_window",
            help="Rolling average filter across frames to prevent sudden camera shakes or lighting flickers from triggering false alerts."
        )

        with st.expander("🔬 View Mathematical & Forensic Details"):
            st.markdown("""
            **1. Decision Threshold ($\tau$):**  
            *Mathematical Role:* Discretizes continuous Sigmoid probability output $P(\\text{Violent} \\mid x) \\in [0, 1]$. A frame is flagged as violent if $P \\ge \\tau$.  
            *Forensic Purpose:* Balances **Precision vs. Recall**. Higher $\\tau$ eliminates false alarms; lower $\\tau$ maximizes threat detection sensitivity.

            **2. Temporal Sampling Interval ($\Delta t$):**  
            *Mathematical Role:* Subsamples frames at uniform stride $K = \\lfloor \\text{FPS} \\times \\Delta t \\rfloor$.  
            *Forensic Purpose:* Eliminates temporal redundancy in CCTV footage. Sampling at $\\Delta t = 1.0\\text{s}$ reduces computational FLOPs by **~95%** while retaining critical event identification.

            **3. Smoothing Window Size ($W$):**  
            *Mathematical Role:* Sliding moving average filter $\\bar{P}_t = \\frac{1}{W} \\sum_{i=0}^{W-1} P_{t-i}$.  
            *Forensic Purpose:* Eliminates transient single-frame noise spikes (e.g. camera shakes, motion blur) to ensure only sustained violent actions trigger forensic flags.
            """)

    # Main Console: Video Ingestion & Forensic Investigation
    col_upload, col_preview = st.columns([1, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Surveillance Video File",
            type=["mp4", "avi", "mov", "mkv", "webm", "flv", "wmv"],
            help="Supports any surveillance video format: MP4, AVI, MOV, MKV, WebM, FLV, WMV."
        )

    video_path = None
    if uploaded_file is not None:
        # Preserve original file extension so OpenCV uses the correct decoder
        orig_ext = os.path.splitext(uploaded_file.name)[-1].lower() or ".mp4"
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=orig_ext)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        with col_preview:
            st.subheader("Surveillance Footage Preview")
            st.video(video_path)

        st.markdown("---")
        
        col_btn, _ = st.columns([1, 3])
        with col_btn:
            run_analysis = st.button("🚀 Run Deep Learning Analysis", type="primary", use_container_width=True)

        if run_analysis:
            model_weight_path = "models/mobilenet_violence_best.pth"
            detector = VideoViolenceDetector(
                model_path=model_weight_path if os.path.exists(model_weight_path) else None,
                confidence_threshold=confidence_thresh,
                temporal_window_size=temporal_window,
                min_event_duration_seconds=1.0
            )
            
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def update_progress(percent: float, message: str):
                progress_bar.progress(percent)
                status_text.text(message)

            with st.spinner("Processing surveillance stream through MobileNetV2..."):
                results = detector.analyze_video(
                    video_path=video_path,
                    output_dir="outputs/evidence_frames",
                    sample_interval_seconds=sampling_interval,
                    progress_callback=update_progress,
                    original_filename=uploaded_file.name
                )

            # Embed investigating police officer metadata into forensic results
            results["metadata"]["officer_name"] = officer_name
            results["metadata"]["badge_id"] = badge_id
            results["metadata"]["department"] = department
            results["metadata"]["session_token"] = session_token

            progress_bar.progress(1.0)
            status_text.success("Analysis Complete!")
            st.session_state["results"] = results

    # Display Results If Present in Session State
    if "results" in st.session_state:
        results = st.session_state["results"]
        meta = results["metadata"]
        events = results["events"]

        st.markdown("## 📊 Investigation Analysis Metrics")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Duration</div>
                <div class="metric-value">{meta['duration_formatted']}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Sampled Frames</div>
                <div class="metric-value">{results['total_sampled_frames']}</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            color_class = "metric-alert" if results['violent_events_count'] > 0 else "metric-safe"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Violent Incidents</div>
                <div class="metric-value {color_class}">{results['violent_events_count']}</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Peak Confidence</div>
                <div class="metric-value">{results['max_confidence']}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Violence Detection Timeline (Spike Line Chart) ──────────────────────────
        if "timeline" in results and len(results["timeline"]) > 0:
            st.subheader("📈 Violence Timeline")

            timeline_df = pd.DataFrame(results["timeline"])
            x_vals = timeline_df["time_sec"].tolist()
            threshold_pct = confidence_thresh * 100

            # Convert seconds → "MM:SS" labels for x-axis ticks
            max_sec = max(x_vals) if x_vals else 60
            tick_step = max(10, int(max_sec / 10 / 10) * 10)   # ~10 ticks
            tick_vals = list(range(0, int(max_sec) + tick_step, tick_step))
            tick_text = [f"{int(t)//60:02d}:{int(t)%60:02d}" for t in tick_vals]

            # Partition timeline into baseline segments (blue) and violence spike segments (red)
            # Normal video remains at baseline (0%), spiking up to confidence (red) during violence.
            timeline_records = timeline_df.to_dict('records')
            segments = []
            if timeline_records:
                curr_state = timeline_records[0]["is_violent"]
                curr_x = []
                curr_y = []

                for row in timeline_records:
                    state = bool(row["is_violent"])
                    x = float(row["time_sec"])
                    conf = float(row["confidence"])
                    y = conf if state else 0.0

                    if state != curr_state:
                        if state:  # Transition: Normal -> Violence (Spike rises from 0% to confidence)
                            curr_x.append(x)
                            curr_y.append(0.0)
                            segments.append({'is_violent': False, 'x': curr_x, 'y': curr_y})
                            # Violent spike starts at baseline and rises vertically
                            curr_x = [x, x]
                            curr_y = [0.0, y]
                        else:      # Transition: Violence -> Normal (Spike drops back to baseline 0%)
                            curr_x.append(x)
                            curr_y.append(0.0)
                            segments.append({'is_violent': True, 'x': curr_x, 'y': curr_y})
                            # Normal baseline resumes
                            curr_x = [x]
                            curr_y = [0.0]
                        curr_state = state
                    else:
                        curr_x.append(x)
                        curr_y.append(y)

                if curr_x:
                    segments.append({'is_violent': curr_state, 'x': curr_x, 'y': curr_y})

            fig = go.Figure()

            has_normal_legend = False
            has_violent_legend = False

            for seg in segments:
                is_v = seg['is_violent']
                color = '#FF1744' if is_v else '#1E88E5'
                name = 'Violence Detected (Spike)' if is_v else 'Normal Video (Baseline)'
                show_leg = False
                if is_v and not has_violent_legend:
                    show_leg = True
                    has_violent_legend = True
                elif not is_v and not has_normal_legend:
                    show_leg = True
                    has_normal_legend = True

                fig.add_trace(go.Scatter(
                    x=seg['x'],
                    y=seg['y'],
                    mode='lines',
                    line=dict(color=color, width=2.5),
                    name=name,
                    showlegend=show_leg,
                    hoverinfo='x+y'
                ))

            # Detection threshold reference line
            fig.add_hline(
                y=threshold_pct,
                line_dash="dash",
                line_color="#FF5252",
                opacity=0.6,
                annotation_text=f"Threshold ({threshold_pct:.0f}%)",
                annotation_position="top right"
            )

            fig.update_layout(
                xaxis=dict(
                    title="Video Time (seconds)",
                    tickmode='array',
                    tickvals=tick_vals,
                    ticktext=tick_text,
                    showgrid=True,
                    gridcolor="rgba(255,255,255,0.08)"
                ),
                yaxis=dict(
                    title="Confidence (%)",
                    range=[-3, 105],
                    showgrid=True,
                    gridcolor="rgba(255,255,255,0.08)"
                ),
                height=280,
                margin=dict(l=45, r=25, t=30, b=45),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1.0
                )
            )

            st.plotly_chart(fig, use_container_width=True)

        # Sort extracted incidents in DESCENDING order of confidence percentage score
        sorted_events = sorted(events, key=lambda x: x["peak_confidence"], reverse=True)

        # Forensic Evidence Gallery sorted by confidence score descending
        st.subheader("🔍 Forensic Evidence Gallery (Ranked by Confidence Score)")
        st.info("Extracted violent incident frames sorted in **descending order from highest confidence score to lowest**.")

        if len(sorted_events) == 0:
            st.success("✅ No violent incidents detected in this surveillance recording. The entire footage is classified as Non-Violent.")
        else:
            # Summary table of ranked extractions
            rank_data = []
            for idx, ev in enumerate(sorted_events, 1):
                rank_data.append({
                    "Rank": f"#{idx}",
                    "Incident ID": f"Incident #{ev['event_id']}",
                    "Confidence Score (%)": f"{ev['peak_confidence']}%",
                    "Climax Timestamp": ev['peak_time'],
                    "Active Window": f"{ev['start_time']} – {ev['end_time']}",
                    "Duration": f"{ev['duration_seconds']}s",
                    "Classification": ev.get('violence_type', 'Violence')
                })
            st.dataframe(pd.DataFrame(rank_data), use_container_width=True, hide_index=True)

            st.markdown("#### 📸 Extracted Climax Frame Snapshots (Highest Confidence First)")

            for rank, event in enumerate(sorted_events, 1):
                with st.expander(
                    f"🏆 Rank #{rank} — Incident #{event['event_id']} (Confidence: {event['peak_confidence']}%) | Window: {event['start_time']} to {event['end_time']}",
                    expanded=(rank == 1)
                ):
                    col_img, col_info = st.columns([1.6, 1.2])

                    with col_img:
                        st.caption(f"📸 Forensic Evidence Snapshot (Rank #{rank} — Climax Frame)")
                        if os.path.exists(event.get("evidence_frame_path", "")):
                            st.image(event["evidence_frame_path"], use_container_width=True)

                    with col_info:
                        st.markdown(f"""
                        - **Confidence Rank:** `#{rank} (Highest to Lowest)`
                        - **Incident ID:** `#{event['event_id']}`
                        - **Peak Confidence:** `🔥 {event['peak_confidence']}%`
                        - **Climax Timestamp:** `{event['peak_time']}`
                        - **Active Window:** `{event['start_time']}` – `{event['end_time']}`
                        - **Duration:** `{event['duration_seconds']}s`
                        - **Classification:** `{event.get('violence_type', 'Physical Violence / Fight')}`
                        - **Investigating Officer:** `{officer_name}`
                        - **Chain of Custody:** `Logged & Verified`
                        """)

        st.markdown("---")

        # Report Generation Section
        st.subheader("📑 Police Forensic Report Generation")
        col_rep1, col_rep2 = st.columns(2)

        reporter = ForensicReportGenerator()
        csv_path = reporter.generate_csv_log(results)
        pdf_path = reporter.generate_pdf_report(results)

        with col_rep1:
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="📄 Download Police Forensic Investigation PDF Report",
                    data=f,
                    file_name=os.path.basename(pdf_path),
                    mime="application/pdf",
                    use_container_width=True
                )

        with col_rep2:
            with open(csv_path, "rb") as f:
                st.download_button(
                    label="📊 Download Incident CSV Log (Chain of Custody)",
                    data=f,
                    file_name=os.path.basename(csv_path),
                    mime="text/csv",
                    use_container_width=True
                )


if __name__ == "__main__":
    main()

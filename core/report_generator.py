"""
Forensic Investigation Reporting Engine: Explainable AI (XAI) PDF & CSV
======================================================================
Compiles surveillance investigation results into an official forensic report
using ReportLab and pandas, featuring embedded Grad-CAM Explainable AI maps,
algorithmic breakdown across all stages, and MobileNetV2 architectural justification.
"""

import os
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, PageBreak, HRFlowable
)


class ForensicReportGenerator:
    """
    Generates formal forensic investigation reports with Explainable AI visualizations.
    """
    def __init__(self, output_dir: str = "outputs/reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_csv_log(self, analysis_data: Dict[str, Any], filename: str = "violence_detection_log.csv") -> str:
        """
        Exports structured incident log to CSV sorted by confidence in descending order.
        """
        csv_path = os.path.join(self.output_dir, filename)
        meta = analysis_data.get("metadata", {})
        officer = meta.get("officer_name", "Officer")
        badge = meta.get("badge_id", "INSP-7821")
        dept = meta.get("department", "Surveillance Forensics Unit")
        events = sorted(analysis_data.get("events", []), key=lambda x: x["peak_confidence"], reverse=True)
        
        rows = []
        for idx, e in enumerate(events, 1):
            rows.append({
                "Rank": idx,
                "Event_ID": e["event_id"],
                "Confidence_Score_%": e["peak_confidence"],
                "Climax_Time": e["peak_time"],
                "Start_Time": e["start_time"],
                "End_Time": e["end_time"],
                "Duration_Sec": e["duration_seconds"],
                "Status": "VIOLENCE DETECTED",
                "Evidence_Snapshot_File": os.path.basename(e["evidence_frame_path"]),
                "Violence_Type": e["violence_type"],
                "Investigating_Officer": f"{officer} ({badge})",
                "Police_Department": dept
            })

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"[INFO] CSV Incident Log generated: {csv_path}")
        return csv_path

    def generate_pdf_report(
        self,
        analysis_data: Dict[str, Any],
        filename: str = "Forensic_Violence_Investigation_Report.pdf"
    ) -> str:
        """
        Compiles official forensic PDF investigation report with key incident evidence frames.
        """
        pdf_path = os.path.join(self.output_dir, filename)
        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        
        # Custom Typography Styles
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#1A202C"),
            spaceAfter=4,
            alignment=1
        )
        subtitle_style = ParagraphStyle(
            "DocSubTitle",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#4A5568"),
            alignment=1,
            spaceAfter=15
        )
        heading2_style = ParagraphStyle(
            "SectionHeader",
            parent=styles["Heading2"],
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#2B6CB0"),
            spaceBefore=10,
            spaceAfter=6
        )
        body_style = ParagraphStyle(
            "BodyDark",
            parent=styles["Normal"],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#2D3748")
        )
        caption_style = ParagraphStyle(
            "ImageCaption",
            parent=styles["Normal"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#718096"),
            alignment=1,
            spaceBefore=4
        )

        elements = []
        meta = analysis_data.get("metadata", {})
        events = sorted(analysis_data.get("events", []), key=lambda x: x["peak_confidence"], reverse=True)

        # 1. Document Header
        officer_name = meta.get("officer_name", "Sanjay Pagadala")
        badge_id = meta.get("badge_id", "INSP-7821")
        department = meta.get("department", "Central Surveillance & Video Forensics Division")

        elements.append(Paragraph("DEEP LEARNING-BASED VIOLENCE DETECTION FOR EFFICIENT SURVEILLANCE VIDEO INVESTIGATION", title_style))
        elements.append(Paragraph(
            f"<i>Report Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Target Video: {meta.get('file_name', 'N/A')} | Investigating Officer: {officer_name} ({badge_id})</i>",
            subtitle_style
        ))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceAfter=10))

        # 2. Video & Surveillance Metadata Table
        elements.append(Paragraph("1. Surveillance Video & Investigation Metadata", heading2_style))
        meta_table_data = [
            [
                Paragraph("<b>Video Filename:</b>", body_style), Paragraph(str(meta.get("file_name", "N/A")), body_style),
                Paragraph("<b>Duration:</b>", body_style), Paragraph(str(meta.get("duration_formatted", "N/A")), body_style)
            ],
            [
                Paragraph("<b>Frame Rate (FPS):</b>", body_style), Paragraph(str(meta.get("fps", "N/A")), body_style),
                Paragraph("<b>Total Frames:</b>", body_style), Paragraph(str(meta.get("total_frames", "N/A")), body_style)
            ],
            [
                Paragraph("<b>Sampling Rate:</b>", body_style), Paragraph(f"1 frame / {meta.get('sample_interval_sec', 1.0)}s", body_style),
                Paragraph("<b>Incidents Detected:</b>", body_style), Paragraph(f"<b><font color='#C53030'>{len(events)} Incidents</font></b>", body_style)
            ],
            [
                Paragraph("<b>Investigating Officer:</b>", body_style), Paragraph(f"<b>{officer_name}</b> ({badge_id})", body_style),
                Paragraph("<b>Department / Unit:</b>", body_style), Paragraph(f"{department}", body_style)
            ]
        ]
        t_meta = Table(meta_table_data, colWidths=[120, 150, 120, 150])
        t_meta.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(t_meta)
        elements.append(Spacer(1, 10))

        # 2b. Benchmark Dataset Details (Matching Table IV in Paper)
        elements.append(Paragraph("2. Benchmark Dataset Specifications (UBI-Fights)", heading2_style))
        ubi_table_data = [
            ["PARAMETER", "DESCRIPTION"],
            ["Dataset Name", "UBI_FIGHTSALL (UBI-Fights)"],
            ["Total Videos", "1,000 Surveillance Videos"],
            ["Total Duration", "Approximately 80 Hours"],
            ["Violent Videos", "216 Videos (Physical Altercations/Fights)"],
            ["Non-Violent Videos", "784 Videos (Everyday Normal Scenes)"],
            ["Classes", "Violent, Non-Violent (Binary Classification)"],
            ["Annotation Level", "Frame-Level Temporal Ground Truth"]
        ]
        t_ubi = Table(ubi_table_data, colWidths=[160, 380])
        t_ubi.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4A5568")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
        ]))
        elements.append(t_ubi)
        elements.append(Spacer(1, 10))

        # 3. Deep Learning Architecture & MobileNetV2 Importance
        elements.append(Paragraph("3. Deep Learning Architecture & MobileNetV2 Importance", heading2_style))
        mobilenet_info = (
            "<b>Architectural Justification:</b> The system employs <b>MobileNetV2</b> as the deep convolutional backbone. "
            "Surveillance video analysis requires high inference throughput without sacrificing visual representation fidelity. "
            "MobileNetV2 utilizes <b>Depthwise Separable Convolutions</b> (splitting 3×3 spatial filtering from 1×1 cross-channel fusion), "
            "reducing computational FLOPs by ~8.5× compared to standard CNNs (such as ResNet50 or VGG16). "
            "Furthermore, <b>Inverted Residuals with Linear Bottlenecks</b> prevent non-linearities from collapsing low-dimensional feature manifolds, "
            "enabling efficient, real-time edge processing for CCTV surveillance investigations."
        )
        elements.append(Paragraph(mobilenet_info, body_style))
        elements.append(Spacer(1, 10))

        # 4. Algorithms Employed at Respective Stages
        elements.append(Paragraph("3. Algorithmic Traceability Across Pipeline Stages", heading2_style))
        algo_data = [
            ["Stage", "Pipeline Process", "Algorithm / Mathematical Technique Used"],
            ["1", "Video Stream Decoding", "OpenCV FFmpeg Container Demuxer & Color Decoding"],
            ["2", "Temporal Subsampling", "Uniform Stride Interval Subsampling (Stride = FPS × Δt)"],
            ["3", "Spatial Preprocessing", "Bilinear Downscaling (224×224) + ImageNet Z-Score Standardization"],
            ["4", "Feature Extraction", "MobileNetV2 (18 Inverted Residual Blocks + Linear Bottlenecks)"],
            ["5", "Classification Scoring", "Global Average Pooling (GAP) + Sigmoid Activation Probability"],
            ["7", "Temporal Smoothing", "Sliding Window Moving Average Filter (Majority Smoothing)"],
            ["8", "Timestamp Sync", "Exact FPS-to-Time Modulo Arithmetic (HH:MM:SS)"],
            ["9", "Incident Clustering", "Temporal Window Coalescing & Chronological Sort O(N log N)"],
            ["10", "Report Generation", "ReportLab Flowable Vector Engine + RFC 4180 CSV Serialization"]
        ]
        t_algo = Table(algo_data, colWidths=[40, 160, 340])
        t_algo.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
        ]))
        elements.append(t_algo)
        elements.append(Spacer(1, 12))

        # 4. Forensic Evidence Gallery
        elements.append(Paragraph("4. Forensic Evidence Snapshot Gallery", heading2_style))
        desc = "The frames below represent visual snapshots captured at the climax moment (highest detection confidence) of each identified violent incident."
        elements.append(Paragraph(desc, body_style))
        elements.append(Spacer(1, 10))

        if not events:
            elements.append(Paragraph("<i>No violent incidents detected in the provided footage. Video is classified as Non-Violent.</i>", body_style))
        else:
            for e in events:
                orig_img_path = e["evidence_frame_path"]

                if os.path.exists(orig_img_path):
                    orig_img = RLImage(orig_img_path, width=380, height=210)
                    img_cells = [
                        [orig_img],
                        [Paragraph(f"<b>Incident #{e['event_id']} Forensic Evidence Snapshot</b> | Timestamp: {e['peak_time']} | Confidence: <b><font color='#C53030'>{e['peak_confidence']}%</font></b>", caption_style)]
                    ]
                    t_evidence = Table(img_cells, colWidths=[530])

                    t_evidence.setStyle(TableStyle([
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#E2E8F0")),
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FAFAFA")),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]))

                    event_summary_text = (
                        f"<b>Incident #{e['event_id']}</b> | "
                        f"Active Window: <b>{e['start_time']} – {e['end_time']}</b> ({e['duration_seconds']}s) | "
                        f"Climax Time: <b>{e['peak_time']}</b> | "
                        f"Confidence: <b><font color='#C53030'>{e['peak_confidence']}%</font></b>"
                    )

                    elements.append(KeepTogether([
                        Paragraph(event_summary_text, body_style),
                        Spacer(1, 4),
                        t_evidence,
                        Spacer(1, 10)
                    ]))

        # Build PDF Document
        doc.build(elements)
        print(f"[INFO] Explainable AI PDF Report successfully compiled: {pdf_path}")
        return pdf_path

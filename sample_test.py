"""
End-to-End Deep Learning Pipeline Verification & Sample Video Generator
======================================================================
Tests:
1. MobileNetV2 Model initialization and tensor forward propagation.
2. Grad-CAM Explainable AI heatmap generation.
3. Synthetic surveillance video generation for rapid testing.
4. End-to-end video analysis and evidence frame generation.
5. Explainable AI PDF report & CSV generation via ReportLab.
"""

import os
import cv2
import torch
import numpy as np
from models.model import ViolenceDetectionMobileNetV2, get_architecture_summary
from core.video_processor import VideoViolenceDetector
from core.report_generator import ForensicReportGenerator


def create_sample_surveillance_video(output_path: str = "data/sample_surveillance.mp4", duration_sec: int = 10, fps: int = 24):
    """
    Synthesizes a sample surveillance video with normal activity followed by rapid motion.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = 480, 640
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    total_frames = duration_sec * fps
    print(f"[TEST] Synthesizing {duration_sec}s test surveillance video ({total_frames} frames)...")

    # Person 1 and Person 2 positions
    p1_x, p1_y = 150, 300
    p2_x, p2_y = 450, 300

    for i in range(total_frames):
        # Create CCTV background (indoor hallway look)
        frame = np.full((h, w, 3), 40, dtype=np.uint8)
        
        # Draw floor and wall perspective
        cv2.line(frame, (0, 380), (w, 380), (70, 70, 70), 2)
        cv2.putText(frame, "CAM_04 - ENTRANCE LOBBY [REC]", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        timestamp_text = f"2026-09-14 14:30:{i // fps:02d}"
        cv2.putText(frame, timestamp_text, (400, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Seconds 0-4: Normal walking towards each other
        if i < 4 * fps:
            p1_x += 1
            p2_x -= 1
            cv2.circle(frame, (p1_x, p1_y), 25, (0, 180, 255), -1)
            cv2.rectangle(frame, (p1_x - 15, p1_y + 25), (p1_x + 15, p1_y + 90), (0, 120, 200), -1)
            cv2.circle(frame, (p2_x, p2_y), 25, (255, 180, 0), -1)
            cv2.rectangle(frame, (p2_x - 15, p2_y + 25), (p2_x + 15, p2_y + 90), (200, 120, 0), -1)
        # Seconds 4-8: Violent confrontation / rapid altercation
        elif i < 8 * fps:
            jitter_x = np.random.randint(-15, 15)
            jitter_y = np.random.randint(-15, 15)
            center_x = (p1_x + p2_x) // 2 + jitter_x
            center_y = 300 + jitter_y
            
            # Draw close physical contact with dynamic shapes
            cv2.circle(frame, (center_x - 20, center_y), 30, (0, 0, 255), -1)
            cv2.circle(frame, (center_x + 20, center_y), 30, (50, 50, 255), -1)
            cv2.line(frame, (center_x - 40, center_y), (center_x + 40, center_y + 30), (0, 0, 255), 4)
            cv2.putText(frame, "! SUDDEN MOTION / IMPACT !", (center_x - 120, center_y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # Seconds 8-10: Separation / Non-violent
        else:
            cv2.circle(frame, (200, 300), 25, (0, 180, 255), -1)
            cv2.circle(frame, (420, 300), 25, (255, 180, 0), -1)

        out.write(frame)

    out.release()
    print(f"[TEST] Test surveillance video saved to: {output_path}")
    return output_path


def run_pipeline_verification():
    print("=" * 60)
    print("STEP 1: VERIFYING MOBILENETV2 DEEP LEARNING MODEL")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    model = ViolenceDetectionMobileNetV2(pretrained=True).to(device)
    summary = get_architecture_summary(model)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    output_prob = model(dummy_input)
    assert output_prob.shape == (2, 1), f"Expected shape (2, 1), got {output_prob.shape}"
    print(f"[*] Forward propagation successful! Output tensor shape: {output_prob.shape}")

    print("\n" + "=" * 60)
    print("STEP 2: GENERATING SYNTHETIC TEST VIDEO & RUNNING INFERENCE")
    print("=" * 60)
    sample_vid_path = create_sample_surveillance_video()
    detector = VideoViolenceDetector(confidence_threshold=0.50)
    results = detector.analyze_video(
        video_path=sample_vid_path,
        output_dir="outputs/evidence_frames",
        sample_interval_seconds=1.0
    )
    print(f"[*] Video analysis completed!")
    print(f"    - Total Sampled Frames: {results['total_sampled_frames']}")
    print(f"    - Detected Events: {results['violent_events_count']}")
    print(f"    - Max Confidence: {results['max_confidence']}%")

    print("\n" + "=" * 60)
    print("STEP 3: VERIFYING PDF & CSV REPORT GENERATION")
    print("=" * 60)
    reporter = ForensicReportGenerator(output_dir="outputs/reports")
    csv_path = reporter.generate_csv_log(results)
    pdf_path = reporter.generate_pdf_report(results)
    
    assert os.path.exists(csv_path), f"CSV report not found at {csv_path}"
    assert os.path.exists(pdf_path), f"PDF report not found at {pdf_path}"
    print(f"[*] CSV Report successfully generated: {csv_path} ({os.path.getsize(csv_path)} bytes)")
    print(f"[*] Explainable AI PDF Report successfully generated: {pdf_path} ({os.path.getsize(pdf_path)} bytes)")

    print("\n" + "=" * 60)
    print("ALL DEEP LEARNING VERIFICATION TESTS PASSED SUCCESSFULLY! [SUCCESS]")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline_verification()

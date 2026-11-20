from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--figures-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def pct(value: float, digits: int = 2) -> str:
    return f"{100 * float(value):.{digits}f}%"


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "BodyJustified",
            parent=styles["BodyText"],
            fontName="Times-Roman",
            fontSize=9.4,
            leading=13.1,
            alignment=TA_JUSTIFY,
            spaceAfter=5.5,
        )
    )
    styles.add(
        ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading1"],
            fontName="Times-Bold",
            fontSize=13.5,
            leading=16,
            spaceBefore=8,
            spaceAfter=5,
            textColor=colors.HexColor("#172B4D"),
        )
    )
    styles.add(
        ParagraphStyle(
            "SubHeading",
            parent=styles["Heading2"],
            fontName="Times-Bold",
            fontSize=10.5,
            leading=13,
            spaceBefore=6,
            spaceAfter=3,
            textColor=colors.HexColor("#243B53"),
        )
    )
    styles.add(
        ParagraphStyle(
            "PaperTitle",
            parent=styles["Title"],
            fontName="Times-Bold",
            fontSize=20,
            leading=24,
            alignment=TA_CENTER,
            spaceAfter=8,
            textColor=colors.HexColor("#102A43"),
        )
    )
    styles.add(
        ParagraphStyle(
            "Subtitle",
            parent=styles["Normal"],
            fontName="Times-Roman",
            fontSize=9.5,
            leading=12,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#52606D"),
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            "Caption",
            parent=styles["Normal"],
            fontName="Times-Italic",
            fontSize=8.1,
            leading=10.3,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#3E4C59"),
            spaceBefore=3,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            "TableText",
            parent=styles["Normal"],
            fontName="Times-Roman",
            fontSize=7.6,
            leading=9.1,
        )
    )
    styles.add(
        ParagraphStyle(
            "Abstract",
            parent=styles["BodyText"],
            fontName="Times-Roman",
            fontSize=9.0,
            leading=12.2,
            alignment=TA_JUSTIFY,
            leftIndent=0.65 * cm,
            rightIndent=0.65 * cm,
        )
    )
    return styles


def data_table(rows, widths):
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9E2EC")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#102A43")),
                ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEADING", (0, 0), (-1, -1), 9.5),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                ("TOPPADDING", (0, 0), (-1, 0), 6),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9FB3C8")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ]
        )
    )
    return table


def add_figure(story, path: Path, width_cm: float, caption: str, styles) -> None:
    image = Image(str(path))
    original_width, original_height = image.imageWidth, image.imageHeight
    draw_width = width_cm * cm
    draw_height = draw_width * original_height / original_width
    image.drawWidth = draw_width
    image.drawHeight = draw_height
    story.append(KeepTogether([image, Paragraph(caption, styles["Caption"])]))


def page_number(canvas, document) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#BCCCDC"))
    canvas.setLineWidth(0.4)
    canvas.line(1.8 * cm, 1.45 * cm, A4[0] - 1.8 * cm, 1.45 * cm)
    canvas.setFont("Times-Roman", 8)
    canvas.setFillColor(colors.HexColor("#52606D"))
    canvas.drawString(1.8 * cm, 1.05 * cm, "Computer Vision Final Project")
    canvas.drawRightString(
        A4[0] - 1.8 * cm, 1.05 * cm, f"Page {document.page}"
    )
    canvas.restoreState()


def build(args: argparse.Namespace) -> Path:
    results = Path(args.results_dir)
    figures = Path(args.figures_dir)
    summary = pd.read_csv(results / "aggregate_all_conditions.summary.csv")
    bootstrap = pd.read_csv(results / "bootstrap_all_conditions.csv")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = make_styles()
    story = []

    story.append(
        Paragraph(
            "Error-Flip-Aware Visual Adaptation for GenD under "
            "Sparse Local Degradation",
            styles["PaperTitle"],
        )
    )
    story.append(
        Paragraph(
            "Computer Vision Final Project - Track 2 Research-Oriented Project"
            "<br/>Experiment date: November 2026",
            styles["Subtitle"],
        )
    )
    story.append(
        Paragraph(
            "<b>Abstract.</b> Short-video deepfake detectors aggregate evidence "
            "from a small number of face frames. We study a failure mode in which "
            "a fixed number of frames are degraded but the positions of those "
            "frames change the final decision. Starting from the frozen GenD "
            "detector, we train a lightweight residual adapter on its 1024-"
            "dimensional CLIP visual features. On a fixed source-group split of "
            "Celeb-DF v2, a clean-only adapter control raises average local-"
            "degradation accuracy from 83.12% to 88.89%. Uniform training over "
            "all six two-of-four blur masks raises it further to 90.38%. The "
            "independent mask-training estimate is +1.50 points with a source-"
            "group 95% interval of [-0.15, 3.43]. Most of the apparent gain is "
            "therefore generic visual adaptation, while sparse mask supervision "
            "has a modest and statistically inconclusive increment.",
            styles["Abstract"],
        )
    )
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("1. Introduction", styles["SectionHeading"]))
    story.append(
        Paragraph(
            "Deepfake videos are frequently compressed, resized, and blurred "
            "during online distribution. Degradation need not affect every "
            "sampled frame: in a four-frame detector, only one or two frames may "
            "lose detail. Earlier analysis of 518 Celeb-DF v2 videos found 61 "
            "videos whose binary label changed across different two-of-four blur "
            "masks although the number and strength of blurred frames were held "
            "fixed. This creates an error-flip problem: can a lightweight "
            "trainable visual extension recover decisions that GenD loses under "
            "sparse local degradation?",
            styles["BodyJustified"],
        )
    )
    story.append(
        Paragraph(
            "We implement a residual adapter after frozen GenD features and "
            "compare clean-only fine-tuning with three degradation objectives "
            "under the same adapter and data split. Uniform training uses all "
            "six local masks; error weighting emphasizes GenD flip masks; and "
            "hard-mask training backpropagates through the currently hardest "
            "eligible mask. All methods use the same split, "
            "adapter, classifier, six mask forwards, inference aggregation, and "
            "threshold. We evaluate three random seeds and two unseen degradation "
            "types. The main experimental finding is that adaptation helps, while "
            "harder-position emphasis does not consistently add value. The "
            "clean-only control reveals that most of the gain over GenD comes "
            "from generic visual fine-tuning.",
            styles["BodyJustified"],
        )
    )
    story.append(
        Paragraph(
            "<b>Contributions.</b> (1) A parameter-efficient visual adapter for "
            "sparse local-degradation robustness on frozen GenD features. "
            "(2) A clean-only attribution control and a comparison of uniform, "
            "error-weighted, and hard-mask training. (3) Video-level error "
            "recovery, worst-mask, and unseen-degradation analysis, including a "
            "transparent negative result for error-focused training.",
            styles["BodyJustified"],
        )
    )

    story.append(Paragraph("2. Related Work", styles["SectionHeading"]))
    story.append(
        Paragraph(
            "GenD generalizes across deepfake benchmarks through parameter-"
            "efficient adaptation of a pretrained CLIP vision encoder and an "
            "L2-normalized feature manifold [1]. Studies of frame aggregation "
            "show that frame-to-video decision design can materially change "
            "detection behavior [2]. QAD learns collaborative representations "
            "across image qualities [3], while DPL uses progressive cross-quality "
            "learning [4]. DCPT enforces clean/degraded paired consistency [5], "
            "and attack-aware deepfake detection studies worst-of-K transform "
            "training and test-time defenses [6].",
            styles["BodyJustified"],
        )
    )
    story.append(
        Paragraph(
            "These works establish that quality variation, robust augmentation, "
            "and degraded-pair consistency are not new ideas. The narrower "
            "question here is whether sparse frame-position selection changes "
            "the benefit obtained from a lightweight visual adapter. We therefore "
            "do not claim that worst-case transforms or the relationship between "
            "frame position and video decisions are first discovered here.",
            styles["BodyJustified"],
        )
    )

    story.append(Paragraph("3. Method", styles["SectionHeading"]))
    story.append(Paragraph("3.1 Frozen GenD features", styles["SubHeading"]))
    story.append(
        Paragraph(
            "Each video is represented by four aligned face crops. GenD maps every "
            "crop to a 1024-dimensional feature and applies a two-class linear "
            "head. We cache clean features, blur-k9 features, and their original "
            "logits. The CLIP encoder remains frozen during all adapter training.",
            styles["BodyJustified"],
        )
    )
    story.append(Paragraph("3.2 Residual visual adapter", styles["SubHeading"]))
    story.append(
        Paragraph(
            "For frame feature z, the adapter computes z' = Normalize(z + "
            "tanh(alpha) W2 GELU(W1 z)). The bottleneck is 128 dimensions and "
            "dropout is 0.1. The residual scale starts at zero, so the initial "
            "model exactly reproduces GenD. The output is re-normalized because "
            "the published GenD classifier expects L2-normalized features. Frame "
            "probabilities are averaged into the video probability and the "
            "decision threshold remains fixed at 0.5.",
            styles["BodyJustified"],
        )
    )
    story.append(Paragraph("3.3 Training objectives", styles["SubHeading"]))
    story.append(
        Paragraph(
            "The clean-only control trains the same adapter using only clean "
            "videos, with the combined clean plus degraded loss weight placed on "
            "the clean objective. It never reads degraded features. The other "
            "variants evaluate all six masks containing exactly two blurred "
            "frames. Uniform training averages the six binary cross-entropy "
            "losses. Error-weighted "
            "training assigns weight three to masks for which GenD is correct on "
            "the clean video but wrong under that mask. Hard-mask training selects "
            "the currently highest-loss eligible mask per video. Every method also "
            "uses a clean-video loss with weight 0.5 and a degraded loss weight of "
            "1.0. Early stopping selects models only on validation loss.",
            styles["BodyJustified"],
        )
    )

    story.append(Paragraph("4. Experimental Setup", styles["SectionHeading"]))
    story.append(
        Paragraph(
            "The supplied Celeb-DF v2 face bank contains 518 videos (178 real, "
            "340 fake) and four aligned 256x256 face crops per video. The original "
            "and blur-k9 archives contain the same 2,072 paths. A deterministic "
            "group split produces 362 training videos (124 real, 238 fake), 78 "
            "validation videos (27 real, 51 fake), and 78 test videos. All twelve "
            "runs use the same split manifest hash. No video crosses splits, and "
            "videos generated from the same source identity pair remain together.",
            styles["BodyJustified"],
        )
    )
    story.append(
        Paragraph(
            "Training uses blur k9. Evaluation additionally uses unseen JPEG "
            "quality 30 and bicubic resize to 25%. The experiment is run with "
            "seeds 2026, 2027, and 2028 for each method. AdamW uses learning rate "
            "0.001, weight decay 0.0001, batch size 32, a maximum of 60 epochs, "
            "and patience 12. Metrics include clean accuracy, average and "
            "worst-mask accuracy, AUC, flip rate, recovery of GenD mask-level "
            "errors, and newly introduced errors. The 16 masks are repeated "
            "measurements of the same videos, so confidence intervals are "
            "bootstrapped over the 52 held-out source groups.",
            styles["BodyJustified"],
        )
    )

    story.append(Paragraph("5. Results", styles["SectionHeading"]))
    story.append(Paragraph("5.1 Main comparison", styles["SubHeading"]))
    rows = [[
        "Condition",
        "GenD",
        "Clean-only",
        "Uniform",
        "Error-weighted",
        "Hard-mask",
    ]]
    for condition, label in (
        ("blur_k9", "Blur k9 (trained)"),
        ("jpeg_q30", "JPEG q30 (unseen)"),
        ("resize_25", "Resize 25% (unseen)"),
    ):
        base = summary[(summary.condition == condition)].iloc[0]
        method_values = []
        for method in ("clean_only", "uniform", "flip_weighted", "flip_hard"):
            row = summary[
                (summary.condition == condition) & (summary.method == method)
            ].iloc[0]
            method_values.append(
                f"{pct(row.adapter_local_accuracy_mean)} "
                f"+/- {pct(row.adapter_local_accuracy_std)}"
            )
        rows.append([label, pct(base.base_local_accuracy_mean), *method_values])
    story.append(
        KeepTogether(
            [
                data_table(
                    rows,
                    [3.0 * cm, 2.5 * cm, 2.7 * cm, 2.9 * cm, 3.0 * cm, 2.8 * cm],
                ),
                Paragraph(
                    "Table 1. Test accuracy averaged over the six two-of-four "
                    "degradation masks. Values are mean +/- standard deviation "
                    "over three model seeds. The clean input and test split are "
                    "identical across methods.",
                    styles["Caption"],
                ),
            ]
        )
    )
    story.append(
        Paragraph(
            "On trained blur, clean-only fine-tuning gains 5.77 points over "
            "GenD and mask training adds 1.50 points. The independent mask-"
            "training interval is [-0.15, 3.43]. For unseen JPEG, clean-only "
            "gains 3.85 points and mask training adds 0.71. For unseen resize, "
            "the corresponding increments are 2.71 and 0.14. The mask-training "
            "effect is directionally positive but not statistically conclusive "
            "with 52 test groups.",
            styles["BodyJustified"],
        )
    )
    add_figure(
        story,
        figures / "local_accuracy_by_condition.png",
        15.5,
        "Figure 1. Local-degradation accuracy. The clean-only adapter captures "
        "most of the gain over GenD; mask training adds a smaller increment.",
        styles,
    )

    story.append(Paragraph("5.2 Error recovery and new errors", styles["SubHeading"]))
    recovery_rows = [["Condition", "Method", "Recovery", "New errors", "Net masks"]]
    for condition, label in (("blur_k9", "Blur"), ("jpeg_q30", "JPEG"), ("resize_25", "Resize")):
        for method in ("clean_only", "uniform", "flip_weighted", "flip_hard"):
            row = summary[
                (summary.condition == condition) & (summary.method == method)
            ].iloc[0]
            recovery_rows.append(
                [
                    label,
                    {
                        "clean_only": "Clean-only",
                        "uniform": "Uniform",
                        "flip_weighted": "Error-weighted",
                        "flip_hard": "Hard-mask",
                    }[method],
                    pct(row.recovery_rate_mean),
                    pct(row.harm_rate_mean),
                    f"{row.net_recovered_mask_events_mean:.2f}",
                ]
            )
    story.append(
        KeepTogether(
            [
                data_table(
                    recovery_rows,
                    [2.6 * cm, 3.9 * cm, 2.7 * cm, 2.7 * cm, 2.6 * cm],
                ),
                Paragraph(
                    "Table 2. Test error inversion. Recovery is the fraction of "
                    "GenD error mask events corrected by the adapter; new errors "
                    "are originally correct mask events made wrong. Counts are "
                    "averaged over seeds.",
                    styles["Caption"],
                ),
            ]
        )
    )
    add_figure(
        story,
        figures / "recovery_and_harm.png",
        15.5,
        "Figure 2. Error recovery versus newly introduced errors. All adapters "
        "recover considerably more blur errors than they create, while uniform "
        "training has the best net balance.",
        styles,
    )

    story.append(Paragraph("5.3 Is harder-position training better?", styles["SubHeading"]))
    story.append(
        Paragraph(
            "Error-weighted training is 0.07 points below uniform on the trained "
            "blur condition (95% interval [-0.32, 0.15]), 0.07 above uniform on "
            "JPEG (interval [0.00, 0.23]), and equal on resize. These differences "
            "do not establish a stable advantage. Hard-mask training is 0.71 "
            "points below uniform on blur with interval [-1.34, -0.16]. Averaging "
            "all local positions is therefore more stable than selecting a single "
            "high-loss position.",
            styles["BodyJustified"],
        )
    )
    add_figure(
        story,
        figures / "position_sensitivity.png",
        15.5,
        "Figure 3. Accuracy for each of the six two-frame blur masks. GenD is "
        "sensitive to which frames are degraded. Every adapter reduces this "
        "position sensitivity, but hard selection is consistently below uniform "
        "training.",
        styles,
    )
    story.append(
        Paragraph(
            "Clean accuracy is 85.90% for GenD, 91.88% for clean-only adaptation, "
            "93.59% for uniform mask training, 93.59% for error weighting, and "
            "92.74% for hard-mask training. The clean-only result confirms that "
            "most of the improvement is a generic adaptation effect.",
            styles["BodyJustified"],
        )
    )

    story.append(Paragraph("5.4 Clean performance and stability", styles["SubHeading"]))
    clean_rows = [["Method", "Clean", "Blur local", "JPEG local", "Resize local"]]
    clean_rows.append(["GenD", "85.90%", "83.12%", "88.03%", "88.46%"])
    for method, label in (
        ("clean_only", "Clean-only"),
        ("uniform", "Uniform"),
        ("flip_weighted", "Error-weighted"),
        ("flip_hard", "Hard-mask"),
    ):
        selected = summary[summary.method == method]
        values = []
        for condition in ("blur_k9", "jpeg_q30", "resize_25"):
            row = selected[selected.condition == condition].iloc[0]
            values.append(pct(row.adapter_local_accuracy_mean))
        first = selected.iloc[0]
        clean_rows.append(
            [
                label,
                f"{pct(first.adapter_clean_accuracy_mean)} "
                f"+/- {pct(first.adapter_clean_accuracy_std)}",
                *values,
            ]
        )
    story.append(
        KeepTogether(
            [
                data_table(
                    clean_rows,
                    [3.4 * cm, 3.5 * cm, 3.0 * cm, 3.0 * cm, 3.0 * cm],
                ),
                Paragraph(
                    "Table 3. Clean accuracy and mean local accuracy by test "
                    "condition. Local values aggregate the six two-frame masks; "
                    "adapter means and standard deviations use three seeds.",
                    styles["Caption"],
                ),
            ]
        )
    )
    story.append(
        Paragraph(
            "The clean-only adapter improves clean test accuracy by 5.98 points "
            "over GenD, and uniform mask training adds another 1.71 points. This "
            "demonstrates that the residual branch and retrained head alter the "
            "visual decision function rather than merely adjusting a scalar at "
            "test time. The gain over GenD therefore cannot be attributed mainly "
            "to local-mask robustness.",
            styles["BodyJustified"],
        )
    )
    story.append(
        Paragraph(
            "The adapter contains 265,347 trainable parameters, including the "
            "two-class head. The 304M-parameter CLIP encoder remains frozen. The "
            "complete 518-video clean/blur feature cache requires 4,144 image "
            "encodes and occupies about 16 MB as a compressed NPZ. After caching, "
            "one training step evaluates six mask variants of a batch but updates "
            "only the small adapter/head, so the experiment is practical on a "
            "single consumer GPU.",
            styles["BodyJustified"],
        )
    )

    story.append(Paragraph("5.5 Interpretation of the negative result", styles["SubHeading"]))
    story.append(
        Paragraph(
            "Several properties of hard-mask training may explain its lower "
            "stability. The selected mask can change as the adapter updates, "
            "creating a moving target. The hardest current mask may emphasize "
            "label noise rather than a stable forgery cue. It also discards the "
            "regularizing signal from the remaining five masks. Error weighting "
            "retains those masks and is therefore much closer to uniform training, "
            "but its additional preference is too weak to create an advantage. "
            "These explanations are hypotheses rather than established mechanisms.",
            styles["BodyJustified"],
        )
    )
    story.append(
        Paragraph(
            "The result separates three questions that should not be merged. "
            "First, does adapting frozen visual features improve robustness? Yes. "
            "Second, does local-mask training add value beyond clean-only fine-"
            "tuning? The estimate is positive on blur but not statistically "
            "conclusive. Third, does explicit error-position selection add value "
            "beyond ordinary local augmentation? The current evidence does not. "
            "Reporting only the improvement over GenD would overstate both "
            "contributions.",
            styles["BodyJustified"],
        )
    )

    story.append(Paragraph("6. Discussion and Limitations", styles["SectionHeading"]))
    story.append(
        Paragraph(
            "The strongest supported intervention is visual adaptation itself. "
            "Clean-only fine-tuning explains 5.77 of the 7.26 points gained over "
            "GenD on trained blur. Local-mask training contributes 1.50 further "
            "points, but its interval crosses zero. The experiment also does not "
            "support error weighting or hard-position emphasis as superior "
            "training rules. A performance improvement over GenD alone would be "
            "insufficient evidence for either mechanism.",
            styles["BodyJustified"],
        )
    )
    story.append(
        Paragraph(
            "The test split contains only 78 videos and 52 source groups. Masks "
            "are repeated measurements rather than independent videos. Celeb-DF "
            "videos were used in earlier exploratory analysis, and the current "
            "test is therefore a held-out group split rather than a completely "
            "blind external benchmark. The source grouping keeps frames and "
            "generation-pair files together but does not prove complete identity "
            "isolation. GenD pretraining overlap must also be checked. Finally, "
            "the study uses four fixed face crops and controlled degradations; it "
            "does not fully model online transcoding or face-detection failure.",
            styles["BodyJustified"],
        )
    )
    story.append(
        Paragraph(
            "The supplied UADFV download link timed out from both the local and "
            "server environments. Existing UADFV score tables were not reused as an "
            "adapter result because they do not contain the new visual features.",
            styles["BodyJustified"],
        )
    )

    story.append(Paragraph("7. Conclusion", styles["SectionHeading"]))
    story.append(
        Paragraph(
            "A lightweight residual adapter on frozen GenD features substantially "
            "improves performance on the fixed test split. The clean-only control "
            "shows that most of this gain is generic visual fine-tuning. Local-"
            "mask training adds a modest and statistically inconclusive increment, "
            "while error-flip and hardest-position emphasis do not improve over "
            "uniform training across all two-frame masks.",
            styles["BodyJustified"],
        )
    )

    story.append(Paragraph("Reproducibility", styles["SectionHeading"]))
    story.append(
        Paragraph(
            "The repository stores the GenD SHA256, fixed split manifests, "
            "feature-cache metadata, three-seed checkpoints, per-video and "
            "per-mask predictions, aggregation scripts, bootstrap output, and "
            "figures. The server workspace is "
            "/root/autodl-tmp/CV_ErrorFlip_GenD. The implementation starts from "
            "the public GenD code at https://github.com/yermandy/GenD.",
            styles["BodyJustified"],
        )
    )

    story.append(Paragraph("Appendix A. Artifact and Evaluation Details", styles["SectionHeading"]))
    artifact_rows = [
        ["Artifact", "Contents"],
        ["Feature cache", "518 x 4 clean and degraded frame features; GenD logits; model and ZIP hashes"],
        ["Run directory", "Best checkpoint, validation history, fixed split manifest, run metadata"],
        ["Evaluation", "Per-video scores for all 16 masks and JSON aggregate metrics"],
        ["Statistics", "Mean and standard deviation over three seeds; source-group bootstrap intervals"],
        ["Figures", "Condition comparison, position sensitivity, and recovery/harm plots"],
    ]
    story.append(
        KeepTogether(
            [
                data_table(artifact_rows, [3.5 * cm, 12.5 * cm]),
                Paragraph(
                    "Table 4. Main reproducibility artifacts. Checkpoint files are "
                    "stored per method and seed; the original GenD model is not "
                    "modified.",
                    styles["Caption"],
                ),
            ]
        )
    )
    story.append(
        Paragraph(
            "The feature cache stores each video's four-frame clean and degraded "
            "feature tensors, the original per-frame logits, the publication "
            "classification head, video labels, source groups, and the source "
            "archive hashes. Training loads only this cache and does not rerun the "
            "vision encoder. Evaluation reconstructs every binary mask from the "
            "cached clean/degraded pair and then averages frame probabilities in "
            "the same way as the original GenD baseline.",
            styles["BodyJustified"],
        )
    )
    story.append(
        Paragraph(
            "The split is generated once from the fixed configuration seed and "
            "written to every run. A SHA256 comparison of all twelve manifests "
            "produced one unique digest, verifying that the model seeds changed "
            "while the data partition did not. Early stopping uses only the 78 "
            "validation videos. The 78 test videos are read only during final "
            "evaluation. Source-group bootstrap resamples 52 held-out groups with "
            "replacement and retains all videos from a selected group.",
            styles["BodyJustified"],
        )
    )
    story.append(
        Paragraph(
            "Unseen JPEG and resize archives are generated from the same selected "
            "face paths. Their feature caches repeat the clean features and replace "
            "only the degraded side. Because a fresh feature-cache job is run for "
            "each condition, small numerical differences in CLIP preprocessing can "
            "change a threshold-adjacent historical prediction. All reported "
            "conditions use the cached baseline from the same execution as the "
            "adapter evaluation, preserving internal comparison fairness.",
            styles["BodyJustified"],
        )
    )

    story.append(Paragraph("References", styles["SectionHeading"]))
    references = [
        "[1] A. Yermakov, J. Cech, J. Matas, and M. Fritz. Deepfake Detection "
        "that Generalizes Across Benchmarks. WACV 2026. arXiv:2508.06248.",
        "[2] P. Charitidis, G. Kordopatis-Zilos, S. Papadopoulos, and I. "
        "Kompatsiaris. Investigating the Impact of Pre-processing and Prediction "
        "Aggregation on the DeepFake Detection Task. arXiv:2006.07084, 2020.",
        "[3] B. M. Le and S. S. Woo. Quality-Agnostic Deepfake Detection with "
        "Intra-model Collaborative Learning. ICCV 2023. arXiv:2309.05911.",
        "[4] D. Zhang, Y. Li, J. Zhou, and Y. Li. DPL: Cross-quality DeepFake "
        "Detection via Dual Progressive Learning. ACCV 2024. arXiv:2410.07633.",
        "[5] Z. Yang, Y. Hou, and X. Yang. Degradation-Consistent Paired Training "
        "for Robust AI-Generated Image Detection. arXiv:2604.10102, 2026.",
        "[6] N. Fatima, H. F. Khan, and M. Behzad. Attack-Aware Deepfake Detection "
        "under Counter-Forensic Manipulations. arXiv:2512.22303, 2025.",
    ]
    for reference in references:
        story.append(Paragraph(reference, styles["BodyJustified"]))

    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=1.85 * cm,
        leftMargin=1.85 * cm,
        topMargin=1.75 * cm,
        bottomMargin=1.8 * cm,
        title="Error-Flip-Aware Visual Adaptation for GenD",
        author="Computer Vision Final Project",
    )
    document.build(story, onFirstPage=page_number, onLaterPages=page_number)
    return output


def main() -> None:
    path = build(parse_args())
    print(path)


if __name__ == "__main__":
    main()

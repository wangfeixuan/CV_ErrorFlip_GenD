from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures_report_20261122"
DOCX_PATH = REPORTS / "CV_ErrorFlip_GenD_Report_20261122.docx"

NAVY = "#1F4E79"
TEAL = "#2A9D8F"
ORANGE = "#E76F51"
GRAY = "#6B7280"
LIGHT_BLUE = "#EAF2F8"
LIGHT_GRAY = "#F2F4F7"
MID_GRAY = "#D9D9D9"
BLACK = "#000000"


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metric(
    rows: list[dict],
    condition: str,
    method: str,
    column: str,
) -> float:
    for row in rows:
        if row["condition"] == condition and row["method"] == method:
            return float(row[column])
    raise KeyError((condition, method, column))


def pct(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}%"


def signed_pp(value: float, digits: int = 2) -> str:
    return f"{100 * value:+.{digits}f} pp"


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shade = tc_pr.find(qn("w:shd"))
    if shade is None:
        shade = OxmlElement("w:shd")
        tc_pr.append(shade)
    shade.set(qn("w:fill"), fill.lstrip("#"))


def set_cell_margins(cell, top=70, start=90, bottom=70, end=90) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (
        ("top", top),
        ("start", start),
        ("bottom", bottom),
        ("end", end),
    ):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table) -> None:
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = f"w:{edge}"
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "6")
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), MID_GRAY.lstrip("#"))


def repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)


def prevent_row_split(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    tr_pr.append(cant_split)


def set_run_font(run, name: str = "Arial", size: float | None = None) -> None:
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    if size is not None:
        run.font.size = Pt(size)


def set_repeat_tab_stops(paragraph, left: float, right: float) -> None:
    paragraph.paragraph_format.tab_stops.add_tab_stop(Inches(left))
    paragraph.paragraph_format.tab_stops.add_tab_stop(Inches(right))


def add_page_field(paragraph) -> None:
    run = paragraph.add_run()
    fld_char_1 = OxmlElement("w:fldChar")
    fld_char_1.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = " PAGE "
    fld_char_2 = OxmlElement("w:fldChar")
    fld_char_2.set(qn("w:fldCharType"), "end")
    run._r.append(fld_char_1)
    run._r.append(instr_text)
    run._r.append(fld_char_2)


def set_document_styles(doc: Document) -> None:
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    normal.paragraph_format.space_after = Pt(4)
    normal.paragraph_format.line_spacing = 1.08
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    for name, size, before, after in (
        ("Title", 21, 0, 10),
        ("Heading 1", 14.5, 12, 5),
        ("Heading 2", 11.5, 8, 3),
    ):
        style = styles[name]
        style.font.name = "Arial"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    subtitle = styles.add_style("Report Subtitle", 1)
    subtitle.font.name = "Arial"
    subtitle.font.size = Pt(11.5)
    subtitle.font.color.rgb = RGBColor(0, 0, 0)
    subtitle.paragraph_format.space_after = Pt(12)

    caption = styles.add_style("Report Caption", 1)
    caption.font.name = "Arial"
    caption.font.size = Pt(8.5)
    caption.font.italic = True
    caption.font.color.rgb = RGBColor(55, 65, 81)
    caption.paragraph_format.space_before = Pt(3)
    caption.paragraph_format.space_after = Pt(7)
    caption.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT

    abstract = styles.add_style("Report Abstract", 1)
    abstract.font.name = "Arial"
    abstract.font.size = Pt(10)
    abstract.paragraph_format.line_spacing = 1.1
    abstract.paragraph_format.space_after = Pt(5)
    abstract.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    code = styles.add_style("Report Code", 1)
    code.font.name = "Consolas"
    code.font.size = Pt(8.7)
    code.paragraph_format.space_before = Pt(2)
    code.paragraph_format.space_after = Pt(5)
    code.paragraph_format.left_indent = Inches(0.25)
    code.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    remove_paragraph_borders(styles["Title"]._element)


def remove_paragraph_borders(element) -> None:
    paragraph_properties = element.find(qn("w:pPr"))
    if paragraph_properties is None:
        return
    for border in list(paragraph_properties.findall(qn("w:pBdr"))):
        paragraph_properties.remove(border)


def configure_page(doc: Document) -> None:
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.68)
    section.bottom_margin = Inches(0.67)
    section.left_margin = Inches(0.78)
    section.right_margin = Inches(0.78)
    section.header_distance = Inches(0.32)
    section.footer_distance = Inches(0.33)
    section.different_first_page_header_footer = True

    header = section.header.paragraphs[0]
    header.text = ""
    header.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = header.add_run("Computer Vision Final Project  |  Domain-Balanced Uniform Adapter")
    set_run_font(run, size=8)
    run.font.color.rgb = RGBColor(90, 99, 112)

    footer = section.footer.paragraphs[0]
    footer.text = ""
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = footer.add_run("Page ")
    set_run_font(run, size=8)
    run.font.color.rgb = RGBColor(90, 99, 112)
    add_page_field(footer)


def add_body(doc: Document, text: str, *, bold_lead: str | None = None) -> None:
    paragraph = doc.add_paragraph(style="Normal")
    if bold_lead and text.startswith(bold_lead):
        lead = paragraph.add_run(bold_lead)
        lead.bold = True
        paragraph.add_run(text[len(bold_lead) :])
    else:
        paragraph.add_run(text)


def add_bullet(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph(style="List Bullet")
    paragraph.paragraph_format.left_indent = Inches(0.24)
    paragraph.paragraph_format.first_line_indent = Inches(-0.14)
    paragraph.paragraph_format.space_after = Pt(2.5)
    paragraph.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    paragraph.add_run(text)


def add_caption(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph(style="Report Caption")
    paragraph.add_run(text)


def add_table(
    doc: Document,
    caption: str,
    headers: list[str],
    rows: list[list[str]],
    widths: list[float],
    *,
    font_size: float = 8.7,
) -> None:
    caption_paragraph = doc.add_paragraph(style="Report Caption")
    caption_paragraph.paragraph_format.keep_with_next = True
    caption_paragraph.add_run(caption)
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_borders(table)
    header_row = table.rows[0]
    repeat_table_header(header_row)
    prevent_row_split(header_row)
    for index, (cell, header, width) in enumerate(
        zip(header_row.cells, headers, widths)
    ):
        cell.width = Inches(width)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        set_cell_margins(cell)
        set_cell_shading(cell, NAVY)
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run(header)
        set_run_font(run, size=font_size)
        run.bold = True
        run.font.color.rgb = RGBColor(255, 255, 255)

    for row_index, row_values in enumerate(rows):
        row = table.add_row()
        prevent_row_split(row)
        fill = "#FFFFFF" if row_index % 2 == 0 else LIGHT_GRAY
        for column_index, (cell, value, width) in enumerate(
            zip(row.cells, row_values, widths)
        ):
            cell.width = Inches(width)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)
            set_cell_shading(cell, fill)
            paragraph = cell.paragraphs[0]
            paragraph.alignment = (
                WD_ALIGN_PARAGRAPH.LEFT
                if column_index == 0
                else WD_ALIGN_PARAGRAPH.CENTER
            )
            run = paragraph.add_run(value)
            set_run_font(run, size=font_size)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def add_picture(
    doc: Document,
    path: Path,
    alt_text: str,
    width: float = 6.75,
) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(4)
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.paragraph_format.keep_with_next = True
    inline_shape = paragraph.add_run().add_picture(str(path), width=Inches(width))
    inline_shape._inline.docPr.set("descr", alt_text)
    inline_shape._inline.docPr.set("title", alt_text)


def generate_pipeline_figure(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.15, 2.55), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.02, 0.58, 0.17, 0.26, "Celeb-DF\nand DFDC", "#EAF2F8"),
        (0.22, 0.58, 0.18, 0.26, "Face detection\nand alignment", "#E8F5F2"),
        (0.43, 0.58, 0.17, 0.26, "Four frames\nper video", "#FFF3E8"),
        (0.63, 0.58, 0.17, 0.26, "Six local\nblur masks", "#F3F0FA"),
        (0.83, 0.58, 0.15, 0.26, "Frozen GenD\nCLIP encoder", "#EEF2F5"),
        (0.17, 0.12, 0.20, 0.25, "Domain-balanced\nsampling", "#EAF2F8"),
        (0.41, 0.12, 0.20, 0.25, "Residual Adapter\nand GenD head", "#E8F5F2"),
        (0.65, 0.12, 0.20, 0.25, "Domain-balanced\nvalidation", "#FFF3E8"),
    ]
    for x, y, w, h, label, fill in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=fill,
            edgecolor="#AAB7C4",
            linewidth=1.0,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=8.3,
            color="#111827",
            fontweight="semibold",
        )
    arrows = [
        ((0.19, 0.71), (0.22, 0.71)),
        ((0.40, 0.71), (0.43, 0.71)),
        ((0.60, 0.71), (0.63, 0.71)),
        ((0.80, 0.71), (0.83, 0.71)),
        ((0.11, 0.58), (0.19, 0.37)),
        ((0.37, 0.245), (0.41, 0.245)),
        ((0.51, 0.58), (0.51, 0.37)),
        ((0.88, 0.58), (0.61, 0.37)),
        ((0.61, 0.245), (0.65, 0.245)),
    ]
    for start, end in arrows:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=1.1,
                color="#4B5563",
            )
        )
    ax.text(
        0.59,
        0.015,
        "Final model: Domain-Balanced Uniform Adapter (DBUA)",
        ha="center",
        va="bottom",
        fontsize=8.6,
        color=NAVY,
        fontweight="bold",
    )
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_domain_figure(
    path: Path,
    same_domain: dict[str, dict[str, float]],
    zero_shot: dict[str, dict[str, float]],
) -> None:
    conditions = ["Clean", "Blur", "JPEG", "Resize"]
    keys = ["clean", "blur", "jpeg", "resize"]
    x = np.arange(len(conditions))
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.25), dpi=300)
    colors = [GRAY, TEAL, NAVY]
    labels = ["Frozen GenD", "Celeb-DF-only", "DBUA"]
    for series_index, label in enumerate(labels):
        values = [
            same_domain[label][key]
            for key in keys
        ]
        bars = axes[0].bar(
            x + (series_index - 1) * width,
            values,
            width,
            label=label,
            color=colors[series_index],
        )
        for bar, value in zip(bars, values):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{100 * value:.1f}",
                ha="center",
                va="bottom",
                fontsize=6.8,
                rotation=90,
            )
    axes[0].set_title(
        "Held-out Celeb-DF, 78 videos",
        fontsize=9.2,
        fontweight="bold",
        pad=18,
    )
    axes[0].set_ylim(0.70, 1.01)
    axes[0].set_xticks(x, conditions)
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(
        frameon=False,
        fontsize=7.1,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
    )
    axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.7)
    axes[0].set_axisbelow(True)

    for series_index, label in enumerate(["Frozen GenD", "DFDC-only"]):
        values = [zero_shot[label][key] for key in keys]
        bars = axes[1].bar(
            x + (series_index - 0.5) * 0.34,
            values,
            width=0.34,
            label=label,
            color=[GRAY, ORANGE][series_index],
        )
        for bar, value in zip(bars, values):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{100 * value:.1f}",
                ha="center",
                va="bottom",
                fontsize=6.8,
                rotation=90,
            )
    axes[1].set_title(
        "Zero-shot transfer to full 518-video list",
        fontsize=9.2,
        fontweight="bold",
        pad=18,
    )
    axes[1].set_ylim(0.70, 1.01)
    axes[1].set_xticks(x, conditions)
    axes[1].legend(
        frameon=False,
        fontsize=7.1,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
    )
    axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.7)
    axes[1].set_axisbelow(True)
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(labelsize=7.4)
    fig.tight_layout(w_pad=1.8)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_ci_figure(path: Path, intervals: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.2), dpi=300)
    y = np.arange(len(intervals))
    for index, item in enumerate(intervals):
        color = TEAL if item["group"] == "Adapter vs GenD" else NAVY
        ax.plot(
            [item["lower"], item["upper"]],
            [index, index],
            color=color,
            linewidth=2.0,
            solid_capstyle="round",
        )
        ax.scatter(
            item["point"],
            index,
            s=28,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        ax.text(
            item["upper"] + 0.006,
            index,
            f"{100 * item['point']:+.1f} pp",
            va="center",
            fontsize=7.2,
            color="#111827",
        )
    ax.axvline(0, color="#9CA3AF", linewidth=1, linestyle=(0, (3, 2)))
    ax.set_yticks(y)
    ax.set_yticklabels([item["label"] for item in intervals], fontsize=7.7)
    ax.set_xlabel("Accuracy difference (percentage points)", fontsize=8)
    ax.set_xlim(-0.065, 0.155)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=7.2)
    handles = [
        plt.Line2D([0], [0], color=TEAL, lw=2, label="DBUA vs GenD"),
        plt.Line2D([0], [0], color=NAVY, lw=2, label="Mask vs Clean-only"),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=7.2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def extract_intervals(
    bootstrap_rows: list[dict],
) -> list[dict]:
    mapping = {
        ("blur_k9", "adapter_vs_base_local"): "Blur vs GenD",
        ("jpeg_q30", "adapter_vs_base_local"): "JPEG vs GenD",
        ("resize_25", "adapter_vs_base_local"): "Resize vs GenD",
        ("blur_k9", "adapter_vs_base_clean"): "Clean vs GenD",
        ("blur_k9", "mask_training_vs_clean_only_local"): "Blur mask vs Clean-only",
        ("jpeg_q30", "mask_training_vs_clean_only_local"): "JPEG mask vs Clean-only",
        ("resize_25", "mask_training_vs_clean_only_local"): "Resize mask vs Clean-only",
    }
    intervals = []
    for row in bootstrap_rows:
        if row["method"] != "multidomain_uniform":
            continue
        key = (row["condition"], row["comparison"])
        if key not in mapping:
            continue
        group = (
            "Adapter vs GenD"
            if row["comparison"] in {"adapter_vs_base_local", "adapter_vs_base_clean"}
            else "Mask vs Clean-only"
        )
        intervals.append(
            {
                "label": mapping[key],
                "group": group,
                "point": float(row["difference"]),
                "lower": float(row["ci95_lower"]),
                "upper": float(row["ci95_upper"]),
            }
        )
    order = [
        "Clean vs GenD",
        "Blur vs GenD",
        "JPEG vs GenD",
        "Resize vs GenD",
        "Blur mask vs Clean-only",
        "JPEG mask vs Clean-only",
        "Resize mask vs Clean-only",
    ]
    return sorted(intervals, key=lambda item: order.index(item["label"]))


def build_report() -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    cdf_methods = load_csv(ROOT / "results" / "aggregate_all_conditions.summary.csv")
    dfdc_only = load_csv(
        ROOT / "results_dfdc" / "aggregate_celebdf.summary.csv"
    )
    multi_cdf = load_csv(
        ROOT / "results_multidomain" / "aggregate_celebdf.summary.csv"
    )
    multi_dfdc = load_csv(
        ROOT / "results_multidomain" / "aggregate_dfdc.summary.csv"
    )
    bootstrap = load_csv(
        ROOT / "results_multidomain" / "bootstrap_celebdf.csv"
    )

    condition_keys = [
        ("Clean", "clean"),
        ("Blur", "blur"),
        ("JPEG", "jpeg"),
        ("Resize", "resize"),
    ]
    gen_values = {
        "clean": metric(
            multi_cdf, "blur_k9", "multidomain_uniform", "base_clean_accuracy_mean"
        ),
        "blur": metric(
            multi_cdf, "blur_k9", "multidomain_uniform", "base_local_accuracy_mean"
        ),
        "jpeg": metric(
            multi_cdf, "jpeg_q30", "multidomain_uniform", "base_local_accuracy_mean"
        ),
        "resize": metric(
            multi_cdf, "resize_25", "multidomain_uniform", "base_local_accuracy_mean"
        ),
    }
    cdf_only_values = {
        "clean": metric(
            cdf_methods, "blur_k9", "uniform", "adapter_clean_accuracy_mean"
        ),
        "blur": metric(
            cdf_methods, "blur_k9", "uniform", "adapter_local_accuracy_mean"
        ),
        "jpeg": metric(
            cdf_methods, "jpeg_q30", "uniform", "adapter_local_accuracy_mean"
        ),
        "resize": metric(
            cdf_methods, "resize_25", "uniform", "adapter_local_accuracy_mean"
        ),
    }
    dua_values = {
        "clean": metric(
            multi_cdf, "blur_k9", "multidomain_uniform", "adapter_clean_accuracy_mean"
        ),
        "blur": metric(
            multi_cdf, "blur_k9", "multidomain_uniform", "adapter_local_accuracy_mean"
        ),
        "jpeg": metric(
            multi_cdf, "jpeg_q30", "multidomain_uniform", "adapter_local_accuracy_mean"
        ),
        "resize": metric(
            multi_cdf, "resize_25", "multidomain_uniform", "adapter_local_accuracy_mean"
        ),
    }
    dfa_zero_values = {
        "clean": metric(
            dfdc_only, "blur_k9", "uniform", "adapter_clean_accuracy_mean"
        ),
        "blur": metric(
            dfdc_only, "blur_k9", "uniform", "adapter_local_accuracy_mean"
        ),
        "jpeg": metric(
            dfdc_only, "jpeg_q30", "uniform", "adapter_local_accuracy_mean"
        ),
        "resize": metric(
            dfdc_only, "resize_25", "uniform", "adapter_local_accuracy_mean"
        ),
    }

    same_domain = {
        "Frozen GenD": gen_values,
        "Celeb-DF-only": cdf_only_values,
        "DBUA": dua_values,
    }
    zero_shot = {
        "Frozen GenD": gen_values,
        "DFDC-only": dfa_zero_values,
    }
    pipeline_path = FIGURES / "figure_1_pipeline.png"
    domains_path = FIGURES / "figure_2_domains.png"
    ci_path = FIGURES / "figure_3_ci.png"
    generate_pipeline_figure(pipeline_path)
    generate_domain_figure(domains_path, same_domain, zero_shot)
    intervals = extract_intervals(bootstrap)
    generate_ci_figure(ci_path, intervals)

    doc = Document()
    configure_page(doc)
    set_document_styles(doc)
    properties = doc.core_properties
    properties.title = (
        "Domain-Balanced Uniform Adapter for Robust Deepfake Detection "
        "under Sparse Local Degradation"
    )
    properties.subject = "Computer Vision Final Project Track 2"
    properties.author = "Computer Vision Final Project"
    properties.comments = "Editable report generated from recorded experiment outputs."

    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(28)
    remove_paragraph_borders(title._p)
    title.add_run(
        "Domain-Balanced Uniform Adapter for Robust Deepfake Detection "
        "under Sparse Local Degradation"
    )
    subtitle = doc.add_paragraph(style="Report Subtitle")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.add_run("Computer Vision Final Project Track 2")
    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.paragraph_format.space_after = Pt(20)
    meta_run = meta.add_run("Final report  |  November 2026")
    set_run_font(meta_run, size=9.5)
    meta_run.font.color.rgb = RGBColor(75, 85, 99)

    abstract_label = doc.add_paragraph()
    abstract_label.paragraph_format.space_before = Pt(10)
    abstract_label.paragraph_format.space_after = Pt(3)
    abstract_run = abstract_label.add_run("Abstract")
    set_run_font(abstract_run, size=11,)
    abstract_run.bold = True
    doc.add_paragraph(
        "Short-video deepfake detectors combine evidence from only a few face "
        "frames, so changing which frames are degraded can change the final "
        "video decision. We study this failure mode on the frozen GenD detector "
        "and train a small residual adapter over its 1024-dimensional CLIP "
        "features. Our final method, the Domain-Balanced Uniform Adapter "
        "(DBUA), samples real and fake videos from Celeb-DF and DFDC in every "
        "batch and optimizes all six two-of-four blur masks. On a fixed "
        "source-group Celeb-DF test split, DBUA raises degraded-condition "
        "accuracy from 86.54% for frozen GenD to 90.12%, while improving clean "
        "accuracy from 85.90% to 91.03%. A DFDC-only adapter transfers poorly "
        "to Celeb-DF. Joint training recovers most of that loss but does not "
        "outperform a Celeb-DF-only adapter. We therefore present the result as "
        "a data-composition and negative-transfer study, not as evidence that "
        "adding data alone is a methodological contribution. Paired "
        "source-group bootstrap intervals, worst-mask accuracy, and explicit "
        "negative ablations are reported.",
        style="Report Abstract",
    )
    keyword = doc.add_paragraph(style="Report Abstract")
    keyword.add_run("Keywords: ").bold = True
    keyword.add_run(
        "deepfake detection; local degradation; residual adapter; domain "
        "generalization; robustness; GenD; Celeb-DF; DFDC"
    )
    doc.add_page_break()

    doc.add_paragraph("1 Introduction", style="Heading 1")
    add_body(
        doc,
        "Deepfake detectors are commonly evaluated on clean frames, while "
        "online video is blurred, compressed, resized, and mixed across "
        "heterogeneous capture conditions. A video-level detector also has to "
        "combine a small set of frame scores. These two properties interact: "
        "degrading only some frames can reverse the aggregate decision even "
        "when the remaining frames still contain useful evidence. We treat that "
        "position-dependent reversal as the central robustness problem.",
    )
    add_body(
        doc,
        "Prior studies show that preprocessing, quality degradation, and "
        "compression can change deepfake detector behavior (Charitidis et al., "
        "2020; Le & Woo, 2023; Yang et al., 2026).",
    )
    add_body(
        doc,
        "The baseline is GenD, a published deepfake detector that operates on "
        "CLIP ViT-L/14 visual features. We keep the 304M-parameter visual "
        "encoder frozen and train a residual adapter over its cached features. "
        "This design makes repeated mask experiments practical on one consumer "
        "GPU and isolates the effect of adaptation from encoder fine-tuning "
        "(Yermakov et al., 2026).",
    )
    add_body(doc, "The project asks three questions:")
    add_bullet(
        doc,
        "Does lightweight adaptation of frozen GenD features improve robustness "
        "to sparse local blur?",
    )
    add_bullet(
        doc,
        "Does mask-specific training add value beyond a clean-only adapter with "
        "the same trainable parameters and loss weight?",
    )
    add_bullet(
        doc,
        "Does training with a second dataset improve cross-dataset behavior, and "
        "what failure modes appear under domain shift?",
    )
    add_body(
        doc,
        "The report separates these questions. It does not attribute every gain "
        "over GenD to the mask objective, and it does not call additional data a "
        "new method. The controlled comparisons, negative transfer analysis, "
        "and failure cases are part of the result.",
    )
    doc.add_paragraph("1.1 Related work", style="Heading 2")
    add_body(
        doc,
        "Deepfake detection research has developed from artifact-specific "
        "classifiers toward models intended to transfer across datasets and "
        "generation methods. GenD uses CLIP visual features and a compact "
        "classification head to improve cross-benchmark behavior (Yermakov et "
        "al., 2026). Celeb-DF and DFDC provide complementary evaluation "
        "settings: Celeb-DF emphasizes challenging synthesis artifacts in "
        "celebrity videos, while DFDC contains a much larger and more diverse "
        "collection of real and manipulated clips (Dolhansky et al., 2020; Li "
        "et al., 2020).",
    )
    add_body(
        doc,
        "A second line of work studies how preprocessing and low-level quality "
        "change detector decisions. Compression, resizing, and blur can remove "
        "the traces learned by a detector, and aggregation choices can change "
        "the final video score (Charitidis et al., 2020). Quality-aware and "
        "degradation-consistent training methods attempt to reduce this "
        "sensitivity by modeling quality conditions explicitly (Le & Woo, "
        "2023; Yang et al., 2026).",
    )
    add_body(
        doc,
        "Our study differs from global augmentation by changing only a sparse "
        "subset of frame positions. This exposes the interaction between "
        "per-frame degradation and video-level aggregation. We also separate "
        "target-domain adaptation from zero-shot transfer, which is necessary "
        "for interpreting whether an additional dataset improves robustness or "
        "only changes the calibration of the trained head.",
    )
    add_picture(
        doc,
        pipeline_path,
        "Diagram of the frozen GenD, local mask training, and domain-balanced adapter pipeline.",
    )
    add_caption(
        doc,
        "Figure 1. Training and evaluation pipeline. GenD remains frozen; the "
        "residual adapter is trained with domain-balanced samples and six local "
        "blur masks.",
    )

    doc.add_paragraph("2 Method", style="Heading 1")
    doc.add_paragraph("2.1 Residual feature adaptation", style="Heading 2")
    add_body(
        doc,
        "For each of four aligned face frames, GenD produces an L2-normalized "
        "feature vector z with dimension 1024. The residual adapter projects z "
        "to a 64-dimensional bottleneck, applies GELU and dropout, projects "
        "back to 1024 dimensions, and combines the result with the original "
        "feature. The adapted feature is h = normalize(z + tanh(a) R(z)), where "
        "a is a learned scale initialized to zero and R is the residual branch. "
        "The original GenD two-class head is retained and updated with the "
        "adapter. The trainable part contains approximately 0.13M parameters.",
    )
    doc.add_paragraph("2.2 Sparse local degradation", style="Heading 2")
    add_body(
        doc,
        "We represent a video by four frame features. A binary four-position "
        "mask selects which frames use blur features. The primary setting uses "
        "exactly two blurred frames, giving six masks. Uniform training averages "
        "the video-level binary cross-entropy over all six masks. This objective "
        "does not assume that a particular frame position is more informative.",
    )
    add_body(
        doc,
        "The training objective combines a clean-video loss and the mean masked "
        "loss: L = 0.5 L_clean + 1.0 L_mask. The clean term preserves the "
        "original decision function, while the mask term exposes the adapter to "
        "all frame positions. The final method excludes prediction-consistency, "
        "feature-restoration, and identity penalties because these variants "
        "produced weaker results.",
    )
    doc.add_paragraph("2.3 Domain-balanced sampling", style="Heading 2")
    add_body(
        doc,
        "DBUA uses both Celeb-DF and DFDC during training. Each micro-batch "
        "contains 16 samples from each domain; within each domain, eight samples "
        "are real and eight are fake. This addresses the DFDC class imbalance "
        "without discarding its manipulation diversity. Validation keeps the "
        "two domains separate, computes a class-balanced loss for each, and "
        "early-stops on their mean. The test splits are never read during "
        "training or model selection.",
    )

    doc.add_paragraph("3 Experimental protocol", style="Heading 1")
    add_body(
        doc,
        "Celeb-DF and DFDC are partitioned by source group. For Celeb-DF, all "
        "videos derived from the same identity pair remain in one split. For "
        "DFDC, a fake video and the real source named by its original field "
        "remain in one split. This prevents the same source from appearing in "
        "both training and test partitions (Dolhansky et al., 2020; Li et al., "
        "2020).",
    )
    add_table(
        doc,
        "Table 1. Video counts in the fixed domain-aware split.",
        ["Domain", "Train", "Validation", "Test", "Test groups"],
        [
            ["Celeb-DF", "362 (124 real / 238 fake)", "78 (27 / 51)", "78 (27 / 51)", "52"],
            ["DFDC part 0", "934 (58 real / 876 fake)", "200 (16 / 184)", "200 (12 / 188)", "12"],
        ],
        [1.08, 1.78, 1.36, 1.28, 0.85],
    )
    add_body(
        doc,
        "GenD and all adapters use the same face frames, degradation protocol, "
        "classification threshold, and source-group partition. JPEG uses "
        "Pillow quality 30 with 4:2:0 subsampling. Resize uses bicubic "
        "downscaling to 25% of width and height. Blur uses a 9 x 9 Gaussian "
        "kernel. Four aligned 256 x 256 crops represent each video.",
    )
    add_body(
        doc,
        "We report clean accuracy, mean local accuracy over six two-frame "
        "masks, worst-mask accuracy, AUC, recovered and harmed mask events, and "
        "feature similarity where relevant. Confidence intervals use 10,000 "
        "bootstrap resamples over source groups. Masks are repeated "
        "measurements of the same videos and are never treated as independent "
        "video samples.",
    )
    add_table(
        doc,
        "Table 2. Compared models and their roles in the analysis.",
        ["Model", "Training data", "Validation", "Purpose"],
        [
            ["Frozen GenD", "None", "None", "Published baseline"],
            ["Celeb-DF-only", "Celeb-DF train", "Celeb-DF val", "Target-domain control"],
            ["DFDC-only", "DFDC train", "DFDC val", "Zero-shot transfer"],
            ["DBUA", "Celeb-DF + DFDC train", "Both validation splits", "Final model"],
            ["DBUA + identity", "Same as DBUA", "Same as DBUA", "Regularization ablation"],
        ],
        [1.15, 1.62, 1.55, 1.95],
        font_size=8.4,
    )

    doc.add_paragraph("4 Results", style="Heading 1")
    doc.add_paragraph("4.1 Single-domain and dual-domain accuracy", style="Heading 2")
    add_body(
        doc,
        "On the held-out Celeb-DF split, DBUA improves all four conditions "
        "over frozen GenD. The target-domain-only adapter is still stronger, so "
        "the DFDC data do not provide an independent gain on this target test.",
    )
    result_rows = []
    for label, key in condition_keys:
        result_rows.append(
            [
                label,
                pct(gen_values[key]),
                pct(cdf_only_values[key]),
                pct(dua_values[key]),
                signed_pp(dua_values[key] - cdf_only_values[key]),
            ]
        )
    add_table(
        doc,
        "Table 3. Single-domain versus dual-domain accuracy on the fixed "
        "Celeb-DF 78-video test split.",
        ["Condition", "GenD", "Celeb-DF-only", "DBUA", "DBUA minus single-domain"],
        result_rows,
        [0.92, 0.92, 1.30, 1.05, 1.48],
    )
    add_picture(
        doc,
        domains_path,
        "Two-panel bar chart comparing held-out Celeb-DF and zero-shot DFDC transfer accuracy.",
    )
    add_caption(
        doc,
        "Figure 2. Left: the same 78-video Celeb-DF split separates the "
        "target-domain control from DBUA. Right: DFDC-only training causes "
        "clear negative transfer on the full 518-video list.",
    )
    doc.add_paragraph("4.2 Negative transfer from DFDC only", style="Heading 2")
    add_body(
        doc,
        "DFDC-only adaptation learns its training domain well but does not "
        "transfer to Celeb-DF. Relative to frozen GenD, clean accuracy falls "
        f"from {pct(gen_values['clean'])} to {pct(dfa_zero_values['clean'])}. "
        f"Blur falls from {pct(gen_values['blur'])} to "
        f"{pct(dfa_zero_values['blur'])}, JPEG from {pct(gen_values['jpeg'])} "
        f"to {pct(dfa_zero_values['jpeg'])}, and resize from "
        f"{pct(gen_values['resize'])} to {pct(dfa_zero_values['resize'])}. "
        "The direction is not explained by a single threshold because clean "
        "AUC also declines.",
    )
    add_body(
        doc,
        "The likely cause is the combination of domain shift and a weak real-"
        "video partition. DFDC part 0 contains only 86 real source groups, and "
        "the adapter updates the final GenD head on features from a different "
        "face-crop and generation distribution. The classification head moves "
        "away from the calibration required by Celeb-DF even though training "
        "and validation losses improve.",
    )
    doc.add_paragraph("4.3 Joint training and uncertainty", style="Heading 2")
    add_body(
        doc,
        "DBUA recovers the DFDC-only loss and improves over GenD, but the "
        "bootstrap intervals are wide. The Celeb-DF test split contains 78 "
        "videos and 52 source groups, so each resample can move several "
        "percentage points. The intervals below describe source-group "
        "uncertainty and should not be interpreted as confidence intervals over "
        "independent masks.",
    )
    add_picture(
        doc,
        ci_path,
        "Forest plot of paired source-group bootstrap intervals for accuracy differences.",
    )
    add_caption(
        doc,
        "Figure 3. Paired source-group bootstrap intervals. The vertical line "
        "marks zero difference.",
    )
    ci_rows = []
    for item in intervals:
        ci_rows.append(
            [
                item["label"],
                signed_pp(item["point"], 2),
                f"[{100 * item['lower']:+.2f}, {100 * item['upper']:+.2f}]",
            ]
        )
    add_table(
        doc,
        "Table 4. Paired source-group 95% bootstrap intervals on Celeb-DF.",
        ["Comparison", "Mean difference", "95% interval (pp)"],
        ci_rows,
        [2.25, 1.20, 1.65],
    )
    add_body(
        doc,
        "The point estimate for mask training is positive under blur and JPEG "
        "and nearly neutral under resize. Only the JPEG interval excludes zero. "
        "The supported claim is therefore a consistent directional signal with "
        "limited statistical resolution, not a universal improvement.",
    )
    doc.add_paragraph("4.4 Identity regularization", style="Heading 2")
    add_body(
        doc,
        "A cosine near-identity penalty of 0.05 was added to constrain the "
        "adapter from moving too far from the frozen GenD representation. It "
        "does not help. DBUA without the penalty has higher clean accuracy and "
        "higher blur and JPEG accuracy, while resize differs by less than one "
        "tenth of a percentage point.",
    )
    identity_rows = []
    for label, key in condition_keys:
        uniform_value = dua_values[key]
        identity_value = (
            metric(
                multi_cdf,
                "blur_k9",
                "multidomain_uniform_identity",
                "adapter_clean_accuracy_mean",
            )
            if key == "clean"
            else metric(
                multi_cdf,
                f"{key}_k9" if key == "blur" else f"{key}_q30" if key == "jpeg" else "resize_25",
                "multidomain_uniform_identity",
                "adapter_local_accuracy_mean",
            )
        )
        identity_rows.append(
            [
                label,
                pct(uniform_value),
                pct(identity_value),
                signed_pp(identity_value - uniform_value),
            ]
        )
    add_table(
        doc,
        "Table 5. Identity-regularization ablation on Celeb-DF.",
        ["Condition", "DBUA", "DBUA + identity", "Effect"],
        identity_rows,
        [1.10, 1.15, 1.45, 1.10],
        font_size=8.2,
    )
    add_body(
        doc,
        "The clean BCE and residual-scale parameter already anchor the model "
        "near the baseline decision surface. A second cosine penalty adds a "
        "representation constraint without adding label information, so it can "
        "restrict useful corrections while retaining domain-specific nuisance "
        "directions. This explanation is a hypothesis; the measured result is "
        "the consistently weaker ablation.",
    )

    doc.add_paragraph("5 Failure analysis and limitations", style="Heading 1")
    doc.add_paragraph("5.1 Why more data is not a method claim", style="Heading 2")
    add_body(
        doc,
        "Adding DFDC only recovers the model after DFDC-only adaptation has "
        "failed. It does not beat the Celeb-DF-only adapter. The correct "
        "comparison is therefore single-domain versus dual-domain on the same "
        "test split, not dual-domain versus GenD. Using the weaker GenD baseline "
        "alone would overstate the contribution of extra data.",
    )
    add_body(
        doc,
        "The methodological contribution remains the residual visual adapter "
        "and the controlled local-mask training question. Domain-balanced "
        "training is an auxiliary robustness experiment. This distinction is "
        "kept explicit in the abstract and conclusion.",
    )
    doc.add_paragraph("5.2 Statistical and dataset limits", style="Heading 2")
    add_body(
        doc,
        "Celeb-DF and DFDC are repartitioned from public video lists or shards "
        "rather than used as completely untouched official blind test sets. "
        "The 78-video Celeb-DF test partition is held out by source group, but "
        "the broader video list was used during exploratory development. The "
        "result is an internal group-held-out evaluation, not a fresh external "
        "benchmark.",
    )
    add_body(
        doc,
        "The DFDC shard is a third-party Kaggle mirror with unknown dataset "
        "license metadata; its videos remain subject to the original DFDC "
        "terms. DFDC part 0 is highly imbalanced and contains only 86 real "
        "source groups. Four fixed face crops also do not reproduce complete "
        "online transcoding or face-detection failures.",
    )
    add_body(
        doc,
        "UADFV was considered as an additional check during early development, "
        "but its download failed and no completed UADFV experiment is reported "
        "here. The DFDC and cross-domain results replace that unfinished branch "
        "in the final analysis.",
    )

    doc.add_paragraph("6 Conclusion", style="Heading 1")
    add_body(
        doc,
        "A small residual adapter over frozen GenD features improves robustness "
        "to sparse local degradation. On the fixed Celeb-DF test split, the "
        "Domain-Balanced Uniform Adapter raises clean accuracy from 85.90% to "
        "91.03% and mean degraded accuracy from 86.54% to 90.12%. The result is "
        "stronger than DFDC-only adaptation but remains below the Celeb-DF-only "
        "control, so DFDC data are not credited with an independent gain.",
    )
    add_body(
        doc,
        "Uniform mask training provides a positive point estimate over "
        "clean-only adaptation on all three degradation types, with the JPEG "
        "interval excluding zero. Identity regularization does not improve the "
        "model. The project therefore supports a restrained conclusion: "
        "lightweight feature adaptation is useful, local-mask training offers a "
        "small and condition-dependent effect, and domain composition can "
        "change robustness without solving cross-dataset transfer.",
    )

    doc.add_paragraph("Reproducibility statement", style="Heading 1")
    add_body(
        doc,
        "The repository records the GenD model hash, fixed manifests, feature "
        "cache metadata, three-seed checkpoints, per-video and per-mask "
        "predictions, aggregation scripts, and bootstrap outputs. The "
        "submission package excludes raw videos and large model weights. "
        "Training uses cached 1024-dimensional features; the frozen CLIP "
        "encoder is not retrained.",
    )
    add_body(
        doc,
        "The final evaluation file has 24 DFDC-only models, 12 multi-domain "
        "models, three seeds per method, and 72 multi-domain evaluation files. "
        "Local tests cover mask enumeration, source-group isolation, "
        "multi-label group splitting, metadata parsing, adapter initialization, "
        "metrics, and balanced sampling.",
    )

    doc.add_paragraph("References", style="Heading 1")
    apa_references = [
        [
            ("Charitidis, P., Kordopatis-Zilos, G., Papadopoulos, S., & "
             "Kompatsiaris, I. (2020). ", False),
            ("Investigating the impact of pre-processing and prediction "
             "aggregation on the deepfake detection task", True),
            (". arXiv. https://arxiv.org/abs/2006.07084", False),
        ],
        [
            ("Dolhansky, B., Bitton, J., Pflaum, B., Lu, J., Howes, R., Wang, "
             "M., & Canton Ferrer, C. (2020). ", False),
            ("The DeepFake Detection Challenge (DFDC) dataset", True),
            (". arXiv. https://arxiv.org/abs/2006.07397", False),
        ],
        [
            ("Le, B. M., & Woo, S. S. (2023). Quality-agnostic deepfake "
             "detection with intra-model collaborative learning. In ", False),
            ("Proceedings of the IEEE/CVF International Conference on Computer "
             "Vision", True),
            (". https://arxiv.org/abs/2309.05911", False),
        ],
        [
            ("Li, Y., Yang, X., Sun, P., Qi, H., & Lyu, S. (2020). Celeb-DF: "
             "A large-scale challenging dataset for deepfake forensics. In ", False),
            ("Proceedings of the IEEE/CVF Conference on Computer Vision and "
             "Pattern Recognition", True),
            (" (pp. 3207-3216). https://doi.org/10.1109/CVPR42600.2020.00327", False),
        ],
        [
            ("Yang, Z., Hou, Y., & Yang, X. (2026). ", False),
            ("Degradation-consistent paired training for robust AI-generated "
             "image detection", True),
            (". arXiv. https://arxiv.org/abs/2604.10102", False),
        ],
        [
            ("Yermakov, A., Cech, J., Matas, J., & Fritz, M. (2026). "
             "Deepfake detection that generalizes across benchmarks. In ", False),
            ("Proceedings of the IEEE/CVF Winter Conference on Applications of "
             "Computer Vision", True),
            (". https://arxiv.org/abs/2508.06248", False),
        ],
    ]
    for reference_runs in apa_references:
        paragraph = doc.add_paragraph(style="Normal")
        paragraph.paragraph_format.left_indent = Inches(0.20)
        paragraph.paragraph_format.first_line_indent = Inches(-0.20)
        paragraph.paragraph_format.space_after = Pt(3)
        paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
        for text, italic in reference_runs:
            run = paragraph.add_run(text)
            run.italic = italic

    doc.add_paragraph("Appendix A  Artifact layout", style="Heading 1")
    add_table(
        doc,
        "Table A1. Reproducibility artifacts included without raw data.",
        ["Artifact", "Contents"],
        [
            ["Models", "Fixed manifests, feature-cache metadata, three-seed checkpoints"],
            ["Evaluation", "Per-video scores for 16 masks and aggregate JSON files"],
            ["Statistics", "Source-group bootstrap intervals and seed summaries"],
            ["Code", "Face import, degradation, feature extraction, training, evaluation"],
            ["Report", "Editable DOCX, final PDF, figures, and result summaries"],
        ],
        [1.40, 4.85],
    )

    doc.save(DOCX_PATH)
    return DOCX_PATH


def main() -> None:
    path = build_report()
    print(path)


if __name__ == "__main__":
    main()

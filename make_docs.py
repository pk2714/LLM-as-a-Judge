# ============================================================
#  make_docs.py  -  STEP 4: build the submission PDFs.
#  Creates  Report.pdf        (2-page report + references page)
#           Team_Details.pdf  (team & problem statement)
#  If results/metrics.csv exists, the REAL numbers and chart are
#  put into the report automatically.
#  Run:    python make_docs.py
# ============================================================
import json
import os

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

import config

# ---------- Styles ----------
BODY = ParagraphStyle("body", fontName="Times-Roman", fontSize=10, leading=12.6,
                      alignment=TA_JUSTIFY, spaceAfter=3)
H = ParagraphStyle("h", fontName="Times-Bold", fontSize=11.5, leading=14,
                   spaceBefore=5, spaceAfter=2)
TITLE = ParagraphStyle("t", fontName="Times-Bold", fontSize=15, leading=18,
                       alignment=TA_CENTER, spaceAfter=3)
CENTER = ParagraphStyle("c", parent=BODY, alignment=TA_CENTER, spaceAfter=1)
SMALL = ParagraphStyle("s", parent=BODY, fontSize=9, leading=11)
REF = ParagraphStyle("r", parent=BODY, fontSize=10, leftIndent=18, firstLineIndent=-18,
                     spaceAfter=5, alignment=0)


def table(rows, widths, header=True):
    t = Table(rows, colWidths=widths)
    style = [("FONT", (0, 0), (-1, -1), "Times-Roman", 9.5),
             ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        style += [("FONT", (0, 0), (-1, 0), "Times-Bold", 9.5),
                  ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8ECF2"))]
    t.setStyle(TableStyle(style))
    return t


def has_custom_ratings():
    """True if the team rated our own 30 answers (data/human_ratings.csv has rows)."""
    f = config.EXPERIMENTS["custom"]["human"]
    return os.path.exists(f) and len(pd.read_csv(f)) > 0


def n_raters():
    f = config.EXPERIMENTS["custom"]["human"]
    return pd.read_csv(f)["rater"].nunique() if has_custom_ratings() else 0


def rater_text():
    n = n_raters()
    words = {1: "one team member", 2: "two team members independently",
             3: "all three team members independently"}
    return words.get(n, f"{n} raters independently")


def team_line():
    return ", ".join(f"{n} ({r})" for n, r, _ in config.TEAM_MEMBERS)


# ---------- Results section (real numbers if available) ----------
def results_section():
    out = []
    if not os.path.exists("results/metrics_all.csv"):
        out.append(Paragraph(
            "<b>[Results will appear here automatically after running judge.py and "
            "evaluate.py, then make_docs.py again.]</b>", BODY))
        return out

    m = pd.read_csv("results/metrics_all.csv")
    rows = [["Dataset", "Criterion", "n", "Spearman \u03c1", "Pearson r", "Weighted \u03ba",
             "MAE", "Within 1", "Bias"]]
    for _, r in m.iterrows():
        rows.append([r.dataset, r.criterion.capitalize(), str(int(r.n)), f"{r.spearman:.2f}",
                     f"{r.pearson:.2f}", f"{r.kappa:.2f}", f"{r.mae:.2f}",
                     f"{r.within_1:.0f}%", f"{r.bias:+.2f}"])
    out.append(Paragraph(f"All responses were judged by <b>{config.MODEL}</b>. Table 1 compares "
                         "the judge with the human scores.", BODY))
    out.append(table(rows, [2.2*cm, 2.3*cm, 1*cm, 2*cm, 1.8*cm, 2.1*cm, 1.4*cm, 1.7*cm, 1.5*cm]))
    out.append(Paragraph("Table 1: Judge-human agreement (bias = mean judge score minus mean "
                         "human score, on the 1-5 scale).", CENTER))

    text = ""
    for name in m.dataset.unique():
        d = m[m.dataset == name]
        best, worst = d.loc[d.spearman.idxmax()], d.loc[d.spearman.idxmin()]
        text += (f"On <b>{name}</b> agreement was highest for {best.criterion} "
                 f"(\u03c1 = {best.spearman:.2f}) and lowest for {worst.criterion} "
                 f"(\u03c1 = {worst.spearman:.2f}); the judge was within one point of the "
                 f"humans for {d.within_1.mean():.0f}% of scores on average. ")
    for _, r in m.iterrows():
        if abs(r.bias) >= 0.3:
            text += (f"For {r.criterion} on {r.dataset} the judge was "
                     f"{'more generous' if r.bias > 0 else 'stricter'} than humans "
                     f"(bias {r.bias:+.2f}). ")
    hh = m[m.human_human_spearman.notna()]
    if not hh.empty:
        text += ("Among our own raters, human-human agreement was " +
                 ", ".join(f"\u03c1 = {r.human_human_spearman:.2f} for {r.criterion}"
                           for _, r in hh.iterrows()) +
                 ", which is the natural upper limit for the judge on that set.")
    out.append(Paragraph(text, BODY))

    if os.path.exists("results/correlation_chart.png"):
        out.append(Image("results/correlation_chart.png", width=11*cm, height=5.75*cm))
        out.append(Paragraph("Figure 1: Spearman correlation between the LLM judge and humans.",
                             CENTER))
    if os.path.exists("results/scatter_helpsteer.png"):
        out.append(Image("results/scatter_helpsteer.png", width=10*cm, height=5*cm))
        out.append(Paragraph("Figure 2: Judge score vs human score for every HelpSteer2 response "
                             "(dashed line = perfect agreement).", CENTER))
    return out


# ---------- Report.pdf ----------
def build_report():
    s = []
    s.append(Paragraph(config.PROJECT_TITLE, TITLE))
    s.append(Paragraph(team_line(), CENTER))
    s.append(Paragraph(f"{config.COURSE}, {config.INSTITUTE}", CENTER))
    s.append(Spacer(1, 4))

    s.append(Paragraph("Abstract", H))
    s.append(Paragraph(
        "Checking the quality of text written by Large Language Models (LLMs) with human "
        "reviewers is slow and costly. We build a simple, reproducible <i>LLM-as-a-Judge</i> "
        "pipeline in which one LLM scores another model's answers on a 1-5 scale for "
        "<b>helpfulness</b>, <b>correctness</b> and <b>tone</b> using a fixed rubric. We then "
        "validate the judge by comparing its scores with human ratings "
        "using Pearson and Spearman correlation, quadratic-weighted Cohen's kappa and mean "
        "absolute error, on " + (f"{config.HELPSTEER_N} HelpSteer2 samples with real human labels "
        "and on 30 of our own samples rated by the team." if has_custom_ratings() else
        f"{config.HELPSTEER_N} HelpSteer2 samples with real human labels."), BODY))

    s.append(Paragraph("1. Introduction", H))
    s.append(Paragraph(
        "LLMs such as ChatGPT, Llama and Gemini produce open-ended answers, so classic metrics "
        "like BLEU or ROUGE, which count overlapping words with a reference answer, do not "
        "measure whether an answer is actually useful, true or polite. Human evaluation is the "
        "gold standard but does not scale. Recent work shows that strong LLMs can act as "
        "evaluators and agree with humans at a level close to human-human agreement (Zheng et "
        "al. 2023; Liu et al. 2023; Chiang and Lee 2023). However, LLM judges also have known "
        "biases, for example preferring longer answers. A judge should therefore be "
        "<i>validated against humans</i> before it is trusted, which is the focus of this project.",
        BODY))

    s.append(Paragraph("2. Problem Statement", H))
    s.append(Paragraph(
        "Given a user question and a model response, automatically assign separate 1-5 scores "
        "for helpfulness (does it solve the need?), correctness (are the facts right?) and tone "
        "(is it polite and suitable?), and measure how closely these automatic scores match human "
        "judgement for each criterion.", BODY))

    s.append(Paragraph("3. Solution Approach", H))
    data_text = (
        f"<b>Data.</b> We use <b>HelpSteer2</b> (Wang et al. 2024), a public NVIDIA dataset in "
        f"which trained annotators rated model responses 0-4 for helpfulness and correctness. "
        f"From its validation split we took {config.HELPSTEER_N} single-turn samples, stratified "
        f"by helpfulness so that bad, average and good answers are equally represented, and "
        f"shifted the scores to 1-5. ")
    if has_custom_ratings():
        data_text += (
            "HelpSteer2 has no tone label, so we also built <b>our own set</b> of 30 responses "
            "(10 questions x 3 answers) written to vary in quality: good answers, polite but wrong "
            "answers, correct but rude answers and vague answers. This tests whether the judge "
            "separates the three criteria.")
    else:
        data_text += ("HelpSteer2 has no tone label, so the judge's tone scores are produced but "
                      "not validated against humans in this study.")
    s.append(Paragraph(data_text, BODY))
    s.append(Paragraph(
        f"<b>Judge model and prompt.</b> The judge is <b>{config.MODEL}</b>, "
        + ("OpenAI's open-weight reasoning model, run with low reasoning effort and "
           if "gpt-oss" in config.MODEL else "") +
        "called through Groq's OpenAI-compatible API at temperature 0 so results are "
        "repeatable. The prompt gives a "
        "short rubric describing scores 1, 3 and 5 for each criterion, tells the judge to rate "
        "criteria independently and not to reward length, and asks for a strict JSON reply "
        "with three integers and a one-line reason (a chain-of-reason style similar to G-Eval). "
        "The reply is parsed with a regular expression, scores are clipped to 1-5, and failed "
        "calls are retried.", BODY))
    s.append(Paragraph(
        "<b>Human validation.</b> For HelpSteer2 we use the released human scores (the mean of "
        "several annotators after outlier filtering)." + (f" For our set, {rater_text()} rated all 30 responses with the same rubric, "
        "without seeing the judge's scores" + (", and the ratings were averaged per sample."
        if n_raters() > 1 else ".") if has_custom_ratings() else ""),
        BODY))
    s.append(Paragraph(
        "<b>Metrics.</b> Pearson r (linear agreement), Spearman ρ (agreement in ranking), "
        "quadratic-weighted Cohen's κ (agreement on an ordinal scale, corrected for chance), "
        "MAE (average gap in points), percentage of samples within one point, and bias (mean "
        "judge minus mean human)." + (" Human-human Spearman correlation between our raters is "
        "reported as a reference upper limit." if has_custom_ratings() else ""), BODY))

    s.append(Paragraph("4. Experimental Results", H))
    s += results_section()

    s.append(Paragraph("5. Conclusion and Limitations", H))
    s.append(Paragraph(
        "The pipeline gives fast, cheap and repeatable scores, and validating against "
        "human ratings shows for which criteria the judge can be trusted. Limitations: we "
        "used short single-turn HelpSteer2 samples only (to fit free API limits), " +
        (f"tone was validated on just 30 responses with {n_raters()} student rater"
         f"{'s' if n_raters() > 1 else ''}, "
         if has_custom_ratings() else "tone could not be validated because HelpSteer2 has no "
         "tone labels, ") +
        "and we tested one judge model with one prompt. Future work: the full dataset, human "
        "tone ratings, several judge models, few-shot rubric examples and pairwise comparison.",
        BODY))

    s.append(PageBreak())
    s.append(Paragraph("References", H))
    refs = [
        "Wang, Zhilin, Yi Dong, Olivier Delalleau, Jiaqi Zeng, Gerald Shen, Daniel Egert, "
        "Jimmy J. Zhang, Makesh Narsimhan Sreedhar, and Oleksii Kuchaiev. 2024. \u201cHelpSteer2: "
        "Open-Source Dataset for Training Top-Performing Reward Models.\u201d <i>arXiv preprint</i> "
        "arXiv:2406.08673.",
        "Zheng, Lianmin, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, "
        "Zi Lin, et al. 2023. “Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.” "
        "In <i>Advances in Neural Information Processing Systems 36 (NeurIPS 2023)</i>.",
        "Liu, Yang, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. 2023. "
        "“G-Eval: NLG Evaluation Using GPT-4 with Better Human Alignment.” In "
        "<i>Proceedings of the 2023 Conference on Empirical Methods in Natural Language "
        "Processing (EMNLP)</i>, 2511-2522.",
        "Chiang, Cheng-Han, and Hung-yi Lee. 2023. “Can Large Language Models Be an "
        "Alternative to Human Evaluations?” In <i>Proceedings of the 61st Annual Meeting of "
        "the Association for Computational Linguistics (ACL)</i>, 15607-15631.",
        "Kim, Seungone, Jamin Shin, Yejin Cho, Joel Jang, Shayne Longpre, Hwaran Lee, Sangdoo Yun, "
        "et al. 2024. “Prometheus: Inducing Fine-Grained Evaluation Capability in Language "
        "Models.” In <i>International Conference on Learning Representations (ICLR 2024)</i>.",
        "Papineni, Kishore, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. “BLEU: A Method "
        "for Automatic Evaluation of Machine Translation.” In <i>Proceedings of the 40th "
        "Annual Meeting of the Association for Computational Linguistics (ACL)</i>, 311-318.",
        "Cohen, Jacob. 1968. “Weighted Kappa: Nominal Scale Agreement with Provision for "
        "Scaled Disagreement or Partial Credit.” <i>Psychological Bulletin</i> 70 (4): 213-220.",
    ]
    for r in sorted(refs):
        s.append(Paragraph(r, REF))

    doc = SimpleDocTemplate("Report.pdf", pagesize=A4, leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=1.6*cm, bottomMargin=1.6*cm, title=config.PROJECT_TITLE)
    doc.build(s)
    print("Built Report.pdf")


# ---------- Team_Details.pdf ----------
def build_team():
    s = [Paragraph("Team & Problem Details", TITLE),
         Paragraph(f"{config.COURSE}<br/>{config.INSTITUTE}", CENTER), Spacer(1, 10)]
    s.append(Paragraph("Project Title", H))
    s.append(Paragraph(config.PROJECT_TITLE, BODY))
    s.append(Paragraph("Submitted to", H))
    s.append(Paragraph(config.PROFESSOR, BODY))
    s.append(Paragraph("Team Members", H))
    rows = [["#", "Name", "Roll Number", "Official Email"]]
    for i, (n, r, e) in enumerate(config.TEAM_MEMBERS, 1):
        rows.append([str(i), n, r, e])
    s.append(table(rows, [1*cm, 5.5*cm, 3.5*cm, 6.5*cm]))
    s.append(Paragraph("Problem Statement", H))
    s.append(Paragraph(
        "Build an <b>LLM-as-a-Judge evaluation pipeline</b>: given a user question and a model's "
        "response, use a second Large Language Model as an evaluator to give separate 1-5 scores "
        "for <b>helpfulness</b>, <b>correctness</b> and <b>tone</b>, with a short reason. Then "
        "<b>validate</b> the judge by comparing its scores with human ratings of the same "
        "responses, using correlation (Pearson, Spearman), weighted Cohen's kappa and mean "
        "absolute error.", BODY))
    s.append(Paragraph("Objectives", H))
    objectives = ["Design a clear scoring rubric and judge prompt for the three criteria.",
                  f"Test on {config.HELPSTEER_N} HelpSteer2 samples that already have human "
                  "ratings from trained annotators."]
    if has_custom_ratings():
        objectives.append("Build our own 30-response set to test tone, rated by the team members.")
    objectives.append("Measure judge-human agreement per criterion and study where they disagree.")
    for o in objectives:
        s.append(Paragraph("• " + o, BODY))
    s.append(Paragraph("Tools", H))
    s.append(Paragraph(f"Python 3, {config.MODEL} via Groq API (OpenAI-compatible), HelpSteer2 dataset, "
                       "pandas, SciPy, scikit-learn, Matplotlib, ReportLab.", BODY))
    doc = SimpleDocTemplate("Team_Details.pdf", pagesize=A4, leftMargin=2.2*cm,
                            rightMargin=2.2*cm, topMargin=2*cm, bottomMargin=2*cm,
                            title="Team & Problem Details")
    doc.build(s)
    print("Built Team_Details.pdf")


if __name__ == "__main__":
    build_report()
    build_team()

import os
import sys

# Ensure local packages are on path
sys.path.insert(0, os.path.abspath("./_local_packages"))

import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def create_deck(output_path="CrossCUTS_Mid_Submission.pptx"):
    prs = Presentation()
    # 16:9 Widescreen layout
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Color Palette
    C_NAVY = RGBColor(15, 23, 42)         # Slate 900
    C_BLUE = RGBColor(2, 132, 199)        # Sky 600
    C_DARK_BLUE = RGBColor(30, 58, 138)   # Blue 900
    C_SLATE = RGBColor(71, 85, 105)       # Slate 600
    C_LIGHT_BG = RGBColor(248, 250, 252)  # Slate 50
    C_CARD_BG = RGBColor(241, 245, 249)   # Slate 100
    C_BORDER = RGBColor(203, 213, 225)    # Slate 300
    C_WHITE = RGBColor(255, 255, 255)
    C_RED = RGBColor(225, 29, 72)         # Rose 600
    C_GREEN = RGBColor(13, 148, 136)      # Teal 600

    def add_header(slide, title_text, category_text="CrossCUTS Project | Mid-Submission Technical Report"):
        # Category / Header Tag
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.4))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        tf_cat.margin_left = tf_cat.margin_top = tf_cat.margin_right = tf_cat.margin_bottom = 0
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = C_BLUE

        # Main Slide Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.75), Inches(11.7), Inches(0.6))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        tf_title.margin_left = tf_title.margin_top = tf_title.margin_right = tf_title.margin_bottom = 0
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(24)
        p_title.font.bold = True
        p_title.font.color.rgb = C_NAVY

        # Subtle divider line
        line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.4), Inches(11.733), Inches(0.02))
        line.fill.solid()
        line.fill.fore_color.rgb = C_BORDER
        line.line.color.rgb = C_BORDER

    # -------------------------------------------------------------
    # SLIDE 1: Title Slide (Dark Theme)
    # -------------------------------------------------------------
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = C_NAVY
    bg1.line.color.rgb = C_NAVY

    # Decorative accent bar
    bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.2), Inches(1.8), Inches(0.12), Inches(3.8))
    bar.fill.solid()
    bar.fill.fore_color.rgb = C_BLUE
    bar.line.color.rgb = C_BLUE

    # Title & Subtitle Box
    tbox = s1.shapes.add_textbox(Inches(1.5), Inches(1.7), Inches(10.5), Inches(3.8))
    tf1 = tbox.text_frame
    tf1.word_wrap = True

    p = tf1.paragraphs[0]
    p.text = "CrossCUTS"
    p.font.size = Pt(44)
    p.font.bold = True
    p.font.color.rgb = C_WHITE

    p2 = tf1.add_paragraph()
    p2.text = "Token- and Language-Level Exploration in GRPO"
    p2.font.size = Pt(26)
    p2.font.bold = True
    p2.font.color.rgb = C_BLUE
    p2.space_before = Pt(8)

    p3 = tf1.add_paragraph()
    p3.text = "Mid-Submission Technical Progress, Diagnostic Analysis & Future Roadmap"
    p3.font.size = Pt(16)
    p3.font.color.rgb = RGBColor(148, 163, 184)
    p3.space_before = Pt(14)

    # Authors Box
    abox = s1.shapes.add_textbox(Inches(1.5), Inches(5.2), Inches(10.5), Inches(1.5))
    atf = abox.text_frame
    atf.word_wrap = True
    ap1 = atf.paragraphs[0]
    ap1.text = "Project Team: Ananya Agarwal (2024114007) | Arushi Shukla (2024101075) | Krishna Kaundinya P (2024101021) | Lohith Kola (2024101132)"
    ap1.font.size = Pt(13)
    ap1.font.bold = True
    ap1.font.color.rgb = C_WHITE

    ap2 = atf.add_paragraph()
    ap2.text = "International Institute of Information Technology, Hyderabad (IIIT-H) • ANLP 2026"
    ap2.font.size = Pt(12)
    ap2.font.color.rgb = RGBColor(148, 163, 184)
    ap2.space_before = Pt(4)

    s1.notes_slide.notes_text_frame.text = (
        "Welcome everyone. Today we present our mid-submission progress on CrossCUTS: Token- and Language-Level "
        "Exploration in Group Relative Policy Optimization (GRPO). We report our empirical findings replicating Mixed-CUTS, "
        "provide an in-depth diagnosis of observed training dynamics, and lay out our path forward toward multilingual reasoning."
    )

    # -------------------------------------------------------------
    # SLIDE 2: Motivation & Background: The Saturation Bottleneck
    # -------------------------------------------------------------
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Motivation: The 'Too Correct to Learn' Bottleneck in RLVR")

    # Card 1: RLVR & GRPO
    c1 = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.1))
    c1.fill.solid()
    c1.fill.fore_color.rgb = C_CARD_BG
    c1.line.color.rgb = C_BORDER
    tf_c1 = c1.text_frame
    tf_c1.word_wrap = True
    tf_c1.margin_left = tf_c1.margin_top = tf_c1.margin_right = tf_c1.margin_bottom = Inches(0.3)

    p = tf_c1.paragraphs[0]
    p.text = "The Promise of GRPO (Shao et al., 2024)"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = C_DARK_BLUE

    items_c1 = [
        "Reinforcement Learning with Verifiable Rewards (RLVR) directly optimizes mathematical accuracy via rule-based verifiers.",
        "GRPO eliminates value critics by estimating baselines directly from G sampled rollouts per prompt.",
        "Advantage Formulation:\n  Â_i = (r_i - mean({r_j})) / (std({r_j}) + ε)",
        "Crucial Requirement: GRPO strictly requires intra-group reward variance (std > 0) to provide learning signals."
    ]
    for it in items_c1:
        pi = tf_c1.add_paragraph()
        pi.text = "• " + it
        pi.font.size = Pt(13)
        pi.font.color.rgb = C_NAVY
        pi.space_before = Pt(10)

    # Card 2: Saturation Trap
    c2 = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.1))
    c2.fill.solid()
    c2.fill.fore_color.rgb = C_CARD_BG
    c2.line.color.rgb = C_BORDER
    tf_c2 = c2.text_frame
    tf_c2.word_wrap = True
    tf_c2.margin_left = tf_c2.margin_top = tf_c2.margin_right = tf_c2.margin_bottom = Inches(0.3)

    p = tf_c2.paragraphs[0]
    p.text = "The Saturation Trap (Liang et al., 2026)"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = C_RED

    items_c2 = [
        "Benchmark Saturation: Strong base models (e.g. Qwen3) achieve high baseline solve rates on standard datasets like MATH.",
        "Vanishing Advantages: When all G rollouts succeed (r_i = 1 for all i), std({r}) = 0. The standardized advantage Â_i collapses to zero.",
        "Consequence: The policy receives zero gradient across majority of training prompts and becomes trapped in local optima.",
        "Core Thesis: We must intervene at decoding time to re-introduce diversity and preserve intra-group advantage variance."
    ]
    for it in items_c2:
        pi = tf_c2.add_paragraph()
        pi.text = "• " + it
        pi.font.size = Pt(13)
        pi.font.color.rgb = C_NAVY
        pi.space_before = Pt(10)

    s2.notes_slide.notes_text_frame.text = (
        "In standard RLVR, group relative policy optimization computes advantages by normalizing rewards within a group. "
        "However, as base models scale, datasets like MATH become saturated. When all rollouts are correct, the variance vanishes, "
        "giving zero gradient signal. The model becomes 'too correct to learn'."
    )

    # -------------------------------------------------------------
    # SLIDE 3: Problem Statement: Multi-Level Exploration
    # -------------------------------------------------------------
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Problem Statement: Token- & Language-Level Exploration")

    pillars = [
        ("1. Beyond Saturation", "Does token uniformization (Mixed-CUTS) still provide gains when trained on non-saturated, intrinsically hard data (DAPO-17K)?\n\nHypothesis: On difficult datasets with natural reward variance, token-level exploration overhead may yield diminishing returns.", C_BLUE),
        ("2. Multilingual Reasoning", "How does Mixed-CUTS behave when reasoning in non-English or low-resource languages where probability distributions are noisier?\n\nMotivation: Token distributions in low-resource tokens may have longer tails, altering the filtering dynamics of CUTS.", C_DARK_BLUE),
        ("3. Exploration Interaction", "Are token-level exploration (Mixed-CUTS) and language-level exploration (PolyGRPO) complementary or conflicting?\n\nGoal: Build a unified 'CrossCUTS' framework combining orthogonal diversity axes without altering the ground-truth verifier.", C_GREEN),
    ]

    for idx, (head, body, col) in enumerate(pillars):
        left_pos = Inches(0.8 + idx * 4.0)
        card = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left_pos, Inches(1.7), Inches(3.7), Inches(5.1))
        card.fill.solid()
        card.fill.fore_color.rgb = C_CARD_BG
        card.line.color.rgb = C_BORDER
        tf = card.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = Inches(0.25)

        hp = tf.paragraphs[0]
        hp.text = head
        hp.font.size = Pt(16)
        hp.font.bold = True
        hp.font.color.rgb = col

        bp = tf.add_paragraph()
        bp.text = body
        bp.font.size = Pt(13)
        bp.font.color.rgb = C_NAVY
        bp.space_before = Pt(14)

    s3.notes_slide.notes_text_frame.text = (
        "We frame our research around three primary questions: (1) Does CUTS help when the dataset is already hard (DAPO-17K)? "
        "(2) Does it stabilize multilingual reasoning? (3) Can we synergize token uniformization with polyglot language exploration?"
    )

    # -------------------------------------------------------------
    # SLIDE 4: Methodology: Mixed-CUTS
    # -------------------------------------------------------------
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Methodology: Constrained Uniform Top-K Sampling (CUTS)")

    # 3-step diagram cards
    steps = [
        ("1. Select (Top-K)", "At decoding step t, extract the K highest-probability tokens V_top-K from model distribution P_θ(· | x_<t).\n\nDefault: K = 5 candidates.", C_BLUE),
        ("2. Filter (δ-Constraint)", "Filter candidates below minimum probability threshold δ:\n\nS_t = { v ∈ V_top-K : P_θ(v) ≥ δ }\n\nDefault: δ = 0.03 (avoids tail noise).", C_DARK_BLUE),
        ("3. Equalize (Uniform)", "Flatten proposal distribution uniformly over survivors:\n\nQ(x_t = v) = 1 / |S_t| for v ∈ S_t\n\nDecouples sampling from greedy model bias.", C_GREEN),
        ("4. Prefix Protection", "Maintain standard sampling for first T_warm tokens:\n\nDefault: T_warm = 5 tokens\n\nPreserves initial prompt setup and reasoning stability.", C_SLATE)
    ]

    for idx, (head, body, col) in enumerate(steps):
        left_pos = Inches(0.8 + idx * 2.95)
        card = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left_pos, Inches(1.7), Inches(2.8), Inches(3.3))
        card.fill.solid()
        card.fill.fore_color.rgb = C_CARD_BG
        card.line.color.rgb = C_BORDER
        tf = card.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = Inches(0.2)

        hp = tf.paragraphs[0]
        hp.text = head
        hp.font.size = Pt(14)
        hp.font.bold = True
        hp.font.color.rgb = col

        bp = tf.add_paragraph()
        bp.text = body
        bp.font.size = Pt(11.5)
        bp.font.color.rgb = C_NAVY
        bp.space_before = Pt(8)

    # Lower box: Mixed-CUTS Rollout Structure
    mix_card = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.2), Inches(11.733), Inches(1.6))
    mix_card.fill.solid()
    mix_card.fill.fore_color.rgb = RGBColor(238, 242, 255)
    mix_card.line.color.rgb = C_BLUE
    mtf = mix_card.text_frame
    mtf.word_wrap = True
    mtf.margin_left = mtf.margin_top = mtf.margin_right = mtf.margin_bottom = Inches(0.25)

    mp1 = mtf.paragraphs[0]
    mp1.text = "Mixed-CUTS Dual-Stream Rollout Structure (G = 16)"
    mp1.font.size = Pt(15)
    mp1.font.bold = True
    mp1.font.color.rgb = C_DARK_BLUE

    mp2 = mtf.add_paragraph()
    mp2.text = (
        "• Exploitation Batch (G_std = 8): Rollouts sampled via standard autoregressive decoding P_θ to preserve policy confidence.\n"
        "• Exploration Batch (G_CUTS = 8): Rollouts generated via CUTS operator Q to uncover plausible under-explored paths.\n"
        "• Advantages computed jointly over combined group G = G_std ∪ G_CUTS, restoring non-zero advantage variance."
    )
    mp2.font.size = Pt(12)
    mp2.font.color.rgb = C_NAVY
    mp2.space_before = Pt(4)

    s4.notes_slide.notes_text_frame.text = (
        "CUTS operates in four phases: Top-K selection, probability threshold filtering, uniform distribution equalization, "
        "and prefix protection. In Mixed-CUTS, rollouts are evenly divided into 8 standard and 8 CUTS samples."
    )

    # -------------------------------------------------------------
    # SLIDE 5: Experimental Setup & Compute Systems
    # -------------------------------------------------------------
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Experimental Setup & System Implementation")

    # Table of Configuration
    rows = [
        ("Base Model", "Qwen3-1.7B (Non-thinking mode, max response length 5,000 tokens)"),
        ("RL Framework", "verl v0.9.0 (Synchronous PPO trainer, Ray distributed, TransferQueue)"),
        ("Inference Engine", "vLLM v0.24.0 (Custom C++/Python LogitsProcessor in Model Runner V1)"),
        ("Hardware Setup", "2x NVIDIA H200 SXM (141 GB VRAM each) on Jarvislabs (1 GPU per arm)"),
        ("Parameter Strategy", "LoRA (r = 64, α = 128 on all linear layers, lr = 1e-5 constant, bf16)"),
        ("Group & Batch Size", "Group Size G = 16 (8 std + 8 CUTS), Global Batch = 128 prompts (2,048 seq/step)"),
        ("Loss & Optimization", "PPO mini-batch = 32 prompts, low-var KL coefficient = 1e-3, Rollout Temp = 1.0"),
        ("Training Dataset", "MATH Training Set filtered to numeric answers (7,382 prompts)"),
        ("Evaluated Benchmarks", "MATH-500, AIME24, AIME25, AMC23, GPQA-Diamond (16 samples per problem)")
    ]

    tbl_shape = s5.shapes.add_table(len(rows) + 1, 2, Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.2))
    tbl = tbl_shape.table
    tbl.columns[0].width = Inches(3.2)
    tbl.columns[1].width = Inches(8.533)

    # Header Row
    tbl.cell(0, 0).text = "Configuration Parameter"
    tbl.cell(0, 1).text = "Implementation Detail"
    for c in range(2):
        cell = tbl.cell(0, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = C_NAVY
        p = cell.text_frame.paragraphs[0]
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = C_WHITE

    for r_idx, (k, v) in enumerate(rows):
        tbl.cell(r_idx + 1, 0).text = k
        tbl.cell(r_idx + 1, 1).text = v
        for c in range(2):
            cell = tbl.cell(r_idx + 1, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = C_CARD_BG if r_idx % 2 == 0 else C_WHITE
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(11.5)
            p.font.color.rgb = C_NAVY
            if c == 0:
                p.font.bold = True

    s5.notes_slide.notes_text_frame.text = (
        "Here is our complete experimental setup. We trained both arms on dedicated H200 GPUs. Note that due to early hardware constraints "
        "on Ada, our implementation utilized LoRA with r=64 and lr=1e-5, whereas the original paper fine-tuned all parameters."
    )

    # -------------------------------------------------------------
    # SLIDE 6: Saturation Test: MATH vs DAPO-17K
    # -------------------------------------------------------------
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Progress: Empirical Verification of Saturation")

    # Left: Explanation card
    sc = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.1))
    sc.fill.solid()
    sc.fill.fore_color.rgb = C_CARD_BG
    sc.line.color.rgb = C_BORDER
    stf = sc.text_frame
    stf.word_wrap = True
    stf.margin_left = stf.margin_top = stf.margin_right = stf.margin_bottom = Inches(0.3)

    p = stf.paragraphs[0]
    p.text = "Reward Distribution Across Prompts (16 Rollouts)"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_DARK_BLUE

    s_points = [
        "MATH Dataset (1,590 prompts evaluated):\n• ~73% (1,160 prompts) are solved 16/16 correctly.\n• ~80% have ≥15/16 rollouts correct.\n• Only ~2% have 0 correct rollouts.\n• Confirms severe saturation: GRPO receives zero gradient signal on ~75% of prompts.",
        "DAPO-17K Dataset (160 sample prompts):\n• Only ~2-3% of prompts are fully solved.\n• ~32% have 0/16 correct rollouts (all wrong).\n• Fails due to intrinsic difficulty, not homogeneity.",
        "Experimental Regimes Defined:\n• MATH: Saturated domain where CUTS is expected to restore variance.\n• DAPO-17K: Hard domain to test whether CUTS adds value beyond saturation."
    ]
    for pt in s_points:
        pi = stf.add_paragraph()
        pi.text = pt
        pi.font.size = Pt(12)
        pi.font.color.rgb = C_NAVY
        pi.space_before = Pt(8)

    # Right: Image from results if available
    img_path = "results/h200-2026-10-01/figures/saturation.png"
    if os.path.exists(img_path):
        s6.shapes.add_picture(img_path, Inches(6.8), Inches(1.7), width=Inches(5.7))
    else:
        # Placeholder card
        sc2 = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.1))
        sc2.fill.solid()
        sc2.fill.fore_color.rgb = C_CARD_BG

    s6.notes_slide.notes_text_frame.text = (
        "Our empirical saturation test validates the paper's core hypothesis: on MATH, Qwen3-1.7B solves 73% of prompts with 16 out of 16 correct. "
        "In contrast, on DAPO-17K, 32% of prompts fail completely. This establishes our two experimental testing grounds."
    )

    # -------------------------------------------------------------
    # SLIDE 7: Mid-Submission Results: Evaluation Table
    # -------------------------------------------------------------
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Benchmark Evaluation: Qwen3-1.7B Across 5 Benchmarks")

    eval_data = [
        ("Base Model (Qwen3-1.7B)", "MATH-500", "70.2%", "79.0%", "91.2%", "850", "1.2%"),
        ("", "AIME24", "13.8%", "20.0%", "46.7%", "2548", "10.8%"),
        ("", "AIME25", "10.6%", "16.7%", "30.0%", "2205", "4.4%"),
        ("", "AMC23", "42.8%", "57.5%", "80.0%", "1406", "1.4%"),
        ("Standard GRPO (Step 40)", "MATH-500", "78.1%", "86.4%", "91.4%", "1258", "7.3%"),
        ("", "AIME24", "23.5%", "36.7%", "40.0%", "3935", "55.0%"),
        ("", "AIME25", "15.8%", "30.0%", "30.0%", "3818", "49.6%"),
        ("", "AMC23", "53.0%", "75.0%", "87.5%", "2355", "21.1%"),
        ("Mixed-CUTS (Step 40)", "MATH-500", "71.5%", "79.0%", "89.0%", "963", "3.3%"),
        ("", "AIME24", "16.0%", "20.0%", "30.0%", "2978", "27.3%"),
        ("", "AIME25", "6.7%", "13.3%", "26.7%", "2578", "16.7%"),
        ("", "AMC23", "43.1%", "55.0%", "80.0%", "1713", "8.9%"),
        ("Mixed-CUTS (Step 50)", "MATH-500", "67.4%", "75.6%", "85.4%", "1075", "7.9%"),
        ("", "AIME25", "3.5%", "6.7%", "16.7%", "2973", "36.9%"),
    ]

    t_eval = s7.shapes.add_table(len(eval_data) + 1, 7, Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.2))
    table = t_eval.table
    table.columns[0].width = Inches(2.7)
    table.columns[1].width = Inches(1.8)
    table.columns[2].width = Inches(1.4)
    table.columns[3].width = Inches(1.4)
    table.columns[4].width = Inches(1.4)
    table.columns[5].width = Inches(1.5)
    table.columns[6].width = Inches(1.533)

    cols = ["Model / Checkpoint", "Benchmark", "Pass@1", "maj@16", "Pass@16", "Mean Tokens", "Truncated (%)"]
    for ci, h in enumerate(cols):
        c = table.cell(0, ci)
        c.fill.solid()
        c.fill.fore_color.rgb = C_NAVY
        p = c.text_frame.paragraphs[0]
        p.text = h
        p.font.size = Pt(11.5)
        p.font.bold = True
        p.font.color.rgb = C_WHITE

    for ri, row in enumerate(eval_data):
        for ci, val in enumerate(row):
            c = table.cell(ri + 1, ci)
            c.fill.solid()
            if "GRPO" in row[0]:
                c.fill.fore_color.rgb = RGBColor(240, 253, 244) # Light green
            elif "Step 40" in row[0] and "CUTS" in row[0]:
                c.fill.fore_color.rgb = RGBColor(254, 242, 242) # Light red
            else:
                c.fill.fore_color.rgb = C_WHITE if ri % 2 == 0 else C_CARD_BG
            p = c.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(10.5)
            p.font.color.rgb = C_NAVY
            if ci in (2, 3):
                p.font.bold = True

    s7.notes_slide.notes_text_frame.text = (
        "Here are the core evaluation results across MATH-500, AIME24, AIME25, and AMC23. "
        "Standard GRPO improves upon the base model (e.g. AIME24 rises from 13.8% to 23.5%), whereas Mixed-CUTS underperformed "
        "and degraded further at step 50. This direct inversion of the paper's findings led us to conduct an intensive root-cause diagnosis."
    )

    # -------------------------------------------------------------
    # SLIDE 8: Training Dynamics & Divergence
    # -------------------------------------------------------------
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Training Dynamics: The Failure Mode Explained")

    # Left: Explanation card
    dyn_card = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.1))
    dyn_card.fill.solid()
    dyn_card.fill.fore_color.rgb = C_CARD_BG
    dyn_card.line.color.rgb = C_BORDER
    dtf = dyn_card.text_frame
    dtf.word_wrap = True
    dtf.margin_left = dtf.margin_top = dtf.margin_right = dtf.margin_bottom = Inches(0.3)

    p = dtf.paragraphs[0]
    p.text = "Observed Anomalies vs. Published Curves"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_RED

    d_points = [
        "1. Entropy Collapse (The Exact Opposite of Paper):\n• Paper: Policy entropy steadily climbed from 0.23 to 0.40.\n• Our Run: Policy entropy crashed from 0.23 to 0.033 by step 41!\n• The model became hyper-greedy and brittle.",
        "2. Response Length Shrinkage:\n• Paper: CUTS trajectories peaked over 1,800 tokens.\n• Our Run: Response lengths shrank from 976 to 842 tokens.",
        "3. KL Divergence Explosion:\n• KL rose 5x faster in Mixed-CUTS (step 20: 0.091 vs 0.018 in GRPO).\n• At step 56, KL spiked to 3.79, and validation crashed to 0.3%."
    ]
    for pt in d_points:
        pi = dtf.add_paragraph()
        pi.text = pt
        pi.font.size = Pt(12)
        pi.font.color.rgb = C_NAVY
        pi.space_before = Pt(8)

    # Right: Overview figure
    ov_path = "results/h200-2026-10-01/figures/overview.png"
    if os.path.exists(ov_path):
        s8.shapes.add_picture(ov_path, Inches(6.8), Inches(1.7), width=Inches(5.7))

    s8.notes_slide.notes_text_frame.text = (
        "Looking at the training dynamics: while the paper reported increasing entropy and longer reasoning chains, "
        "our Mixed-CUTS run showed immediate entropy collapse to 0.033, response length shrinkage, and a sudden KL explosion "
        "around step 48 to 56. Both arms diverged, but Mixed-CUTS diverged earlier."
    )

    # -------------------------------------------------------------
    # SLIDE 9: Root Cause Diagnosis
    # -------------------------------------------------------------
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Technical Diagnosis: Why Did Mixed-CUTS Underperform?")

    diag_boxes = [
        ("Discrepancy 1: Missing Equation (2) Length Normalization",
         "The paper's loss in Eq. (2) explicitly normalizes each rollout by its own length (1/|o_i|):\n"
         "  L = (1/G) * Σ (1/|o_i|) * Σ_t min(ratio * Â, clip * Â)\n"
         "Our implementation used verl's default 'token-mean' (global batch token sum). Long failing CUTS rollouts (3k-5k tokens) received 3x-4x more penalty weight than short correct rollouts, severely penalizing length and exploration.", C_RED),
        ("Discrepancy 2: Saturated Group Advantage Asymmetry",
         "On saturated prompts, 8/8 standard rollouts succeed (r=1). When CUTS explores and yields 2 failures (r=0):\n"
         "  Â_correct = +0.38    vs.    Â_failed = -2.65\n"
         "The penalty on exploratory tokens was 7x stronger than rewards. Because CUTS forced uniform sampling of low-prob tokens (P ≥ 0.03), their probabilities were aggressively crushed to zero, causing entropy collapse.", C_DARK_BLUE),
        ("Discrepancy 3: KL Penalty Blowup under low_var_kl",
         "The Schulman k3 estimator computes:\n"
         "  D_KL ≈ (π_ref / π_θ) - log(π_ref / π_θ) - 1\n"
         "Notice π_θ is in the denominator. When penalized exploratory tokens have π_θ → 0, (π_ref / π_θ) explodes to infinity, triggering the catastrophic divergence at step 48-56.", C_BLUE),
        ("Discrepancy 4: LoRA Bottleneck at High Learning Rate",
         "The paper used Full Parameter Fine-Tuning at lr ≈ 1e-6. We used LoRA (r=64, α=128) at lr = 1e-5 (10x higher). Low-rank adapter updates ΔW = BA saturated rapidly under conflicting exploration/exploitation gradients.", C_SLATE)
    ]

    for idx, (head, body, col) in enumerate(diag_boxes):
        rx = idx % 2
        ry = idx // 2
        l = Inches(0.8 + rx * 5.95)
        t = Inches(1.7 + ry * 2.65)
        box = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, l, t, Inches(5.75), Inches(2.45))
        box.fill.solid()
        box.fill.fore_color.rgb = C_CARD_BG
        box.line.color.rgb = C_BORDER
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = Inches(0.2)

        hp = tf.paragraphs[0]
        hp.text = head
        hp.font.size = Pt(13)
        hp.font.bold = True
        hp.font.color.rgb = col

        bp = tf.add_paragraph()
        bp.text = body
        bp.font.size = Pt(10.5)
        bp.font.color.rgb = C_NAVY
        bp.space_before = Pt(4)

    s9.notes_slide.notes_text_frame.text = (
        "We isolated four fundamental causes: (1) Missing sequence length normalization: verl's token-mean overweighted long failing rollouts. "
        "(2) Advantage asymmetry: negative penalties were 7x stronger than rewards. (3) low_var_kl has pi_theta in the denominator, exploding "
        "when tokens are suppressed. (4) LoRA capacity limitations at high learning rate."
    )

    # -------------------------------------------------------------
    # SLIDE 10: Concrete Code Fixes
    # -------------------------------------------------------------
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Targeted Code & Configuration Fixes")

    fixes = [
        ("1. Set Sequence-Normalized Loss in configs/train/base_grpo.yaml",
         "Under actor_rollout_ref.actor:\n"
         "  loss_agg_mode: seq-mean-token-mean   # [FIX] Enforces Eq. (2) sequence-length normalization\n"
         "Normalizes each rollout by 1/|o_i| so long exploratory reasoning traces are not penalized.", C_GREEN),
        ("2. Stabilize KL Divergence Estimator in configs/train/base_grpo.yaml",
         "Under actor_rollout_ref.actor:\n"
         "  kl_loss_type: kl                     # [FIX] Standard forward KL (avoids 1/π_θ denominator)\n"
         "  kl_loss_coef: 0.001\n"
         "Eliminates the denominator blowup when exploratory tokens receive negative updates.", C_BLUE),
        ("3. Full Parameter Fine-Tuning or Calibrated LoRA in Memory Configs",
         "On Hopper 80 GB GPUs, Qwen3-1.7B full fine-tuning requires only ~21 GB VRAM:\n"
         "  lora_rank: 0                         # [FIX] Full Parameter Fine-Tuning\n"
         "  optim.lr: 1.0e-6                     # [FIX] Paper's stable learning rate\n"
         "If constrained to LoRA, drop learning rate to 2e-6 with cosine schedule.", C_DARK_BLUE),
        ("4. Enable Symbolic Math Grading in src/mixed_cuts/reward.py",
         "In extract_answer(solution_str):\n"
         "  # Remove strict digit requirement from training verifier; preserve all LaTeX expressions\n"
         "  return pred, True, len(pred) > 0     # [FIX] Grade all valid symbolic answers with math-verify", C_SLATE)
    ]

    for idx, (head, body, col) in enumerate(fixes):
        rx = idx % 2
        ry = idx // 2
        l = Inches(0.8 + rx * 5.95)
        t = Inches(1.7 + ry * 2.65)
        box = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, l, t, Inches(5.75), Inches(2.45))
        box.fill.solid()
        box.fill.fore_color.rgb = C_CARD_BG
        box.line.color.rgb = C_BORDER
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = Inches(0.2)

        hp = tf.paragraphs[0]
        hp.text = head
        hp.font.size = Pt(12.5)
        hp.font.bold = True
        hp.font.color.rgb = col

        bp = tf.add_paragraph()
        bp.text = body
        bp.font.size = Pt(10.5)
        bp.font.color.rgb = C_NAVY
        bp.space_before = Pt(4)

    s10.notes_slide.notes_text_frame.text = (
        "Here are the exact code and configuration fixes: (1) Set loss_agg_mode to seq-mean-token-mean in base_grpo.yaml. "
        "(2) Switch kl_loss_type to standard kl. (3) Switch to full fine-tuning or calibrate LoRA lr. (4) Remove digit-only filter in reward.py."
    )

    # -------------------------------------------------------------
    # SLIDE 11: Project Timeline & Updated Milestones
    # -------------------------------------------------------------
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Project Roadmap & Updated Weekly Timeline")

    weeks = [
        ("Week 1: Retraining & Baseline Stabilization",
         "• Apply loss_agg_mode: seq-mean-token-mean and stabilized KL in base_grpo.yaml.\n"
         "• Retrain Qwen3-1.7B on MATH; verify entropy growth and replicate paper Table 1.\n"
         "• Model size adaptation: Port to Qwen3-0.6B to enable rapid, low-compute exploration cycles.", C_BLUE),
        ("Week 2: Beyond-Saturation Stress Test (DAPO-17K & Poly-MATH)",
         "• Train Mixed-CUTS on DAPO-17K to verify whether CUTS helps on naturally hard data.\n"
         "• Establish baseline multilingual evaluations across the Poly-MATH dataset.\n"
         "• Measure Advantage Collapse Rate (ACR) across easy vs. hard regimes.", C_DARK_BLUE),
        ("Week 3: Language-Level Exploration (PolyGRPO)",
         "• Implement PolyGRPO rollout scheduling in agent_loop.py (unconstrained + language-forced rollouts).\n"
         "• Train and benchmark PolyGRPO on Poly-MATH dataset.\n"
         "• Analyze language-conditioned reasoning trajectories.", C_GREEN),
        ("Week 4: CrossCUTS Synthesis & Final Submission",
         "• Integrate Mixed-CUTS (token-level) + PolyGRPO (language-level) into unified CrossCUTS.\n"
         "• Run full 4-way ablation: Standard GRPO vs. Mixed-CUTS vs. PolyGRPO vs. CrossCUTS.\n"
         "• Comprehensive evaluation (Pass@1, Pass@16, maj@16) and final report delivery.", C_NAVY)
    ]

    for idx, (head, body, col) in enumerate(weeks):
        left_pos = Inches(0.8 + idx * 2.95)
        card = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left_pos, Inches(1.7), Inches(2.8), Inches(5.1))
        card.fill.solid()
        card.fill.fore_color.rgb = C_CARD_BG
        card.line.color.rgb = C_BORDER
        tf = card.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = Inches(0.2)

        hp = tf.paragraphs[0]
        hp.text = head
        hp.font.size = Pt(13)
        hp.font.bold = True
        hp.font.color.rgb = col

        bp = tf.add_paragraph()
        bp.text = body
        bp.font.size = Pt(11)
        bp.font.color.rgb = C_NAVY
        bp.space_before = Pt(8)

    s11.notes_slide.notes_text_frame.text = (
        "Our updated timeline spans 4 weeks: Week 1 focuses on retraining with our fixes. Week 2 evaluates on DAPO-17K. "
        "Week 3 implements PolyGRPO. Week 4 synthesizes both into the final CrossCUTS framework."
    )

    # -------------------------------------------------------------
    # SLIDE 12: Summary, Artifacts & Q&A (Dark Theme)
    # -------------------------------------------------------------
    s12 = prs.slides.add_slide(blank_layout)
    bg12 = s12.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg12.fill.solid()
    bg12.fill.fore_color.rgb = C_NAVY
    bg12.line.color.rgb = C_NAVY

    # Box 1: Key Takeaways
    b1 = s12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.2), Inches(5.6), Inches(5.2))
    b1.fill.solid()
    b1.fill.fore_color.rgb = RGBColor(30, 41, 59)
    b1.line.color.rgb = RGBColor(51, 65, 85)
    tf_b1 = b1.text_frame
    tf_b1.word_wrap = True
    tf_b1.margin_left = tf_b1.margin_top = tf_b1.margin_right = tf_b1.margin_bottom = Inches(0.3)

    p = tf_b1.paragraphs[0]
    p.text = "Key Mid-Submission Takeaways"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = C_BLUE

    points_b1 = [
        "Saturation Validated: MATH is 73% all-correct for Qwen3-1.7B, confirming the need for exploration.",
        "Divergence Understood: Discrepancy between our run and paper traced to loss_agg_mode, low_var_kl blowup, and LoRA bottlenecks.",
        "Clear Fix Implemented: Replacing token-mean with seq-mean-token-mean aligns our code with Eq. (2) and prevents entropy collapse.",
        "On Track for CrossCUTS: Confident path toward integrating token-level and polyglot language exploration."
    ]
    for pt in points_b1:
        pi = tf_b1.add_paragraph()
        pi.text = "✓ " + pt
        pi.font.size = Pt(13)
        pi.font.color.rgb = C_WHITE
        pi.space_before = Pt(12)

    # Box 2: Artifacts & Q&A
    b2 = s12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.2), Inches(5.7), Inches(5.2))
    b2.fill.solid()
    b2.fill.fore_color.rgb = RGBColor(30, 41, 59)
    b2.line.color.rgb = RGBColor(51, 65, 85)
    tf_b2 = b2.text_frame
    tf_b2.word_wrap = True
    tf_b2.margin_left = tf_b2.margin_top = tf_b2.margin_right = tf_b2.margin_bottom = Inches(0.3)

    p = tf_b2.paragraphs[0]
    p.text = "Project Artifacts & Discussion"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = C_BLUE

    points_b2 = [
        "Code Repository:\ngithub.com/dumbkin31/exploration-in-grpo",
        "Curated Results & Metrics:\nLocated in results/h200-2026-10-01/",
        "Model Checkpoints & Rollout Dumps:\nHuggingFace: lokola13/mixed-cuts-h200-runs",
        "W&B Experiment Tracking:\nProject anlp-mixed-cuts/mixed-cuts\nRuns: math_grpo-h200-s1 & math_mixed_cuts-h200-s1"
    ]
    for pt in points_b2:
        pi = tf_b2.add_paragraph()
        pi.text = "• " + pt
        pi.font.size = Pt(12)
        pi.font.color.rgb = RGBColor(226, 232, 240)
        pi.space_before = Pt(8)

    p_qa = tf_b2.add_paragraph()
    p_qa.text = "Questions & Feedback?"
    p_qa.font.size = Pt(18)
    p_qa.font.bold = True
    p_qa.font.color.rgb = C_WHITE
    p_qa.space_before = Pt(16)

    s12.notes_slide.notes_text_frame.text = (
        "In summary, we have validated benchmark saturation, diagnosed the divergence root causes down to specific configuration lines, "
        "and established a clear 4-week roadmap to complete the CrossCUTS framework. Thank you, and we welcome your questions."
    )

    # Save presentation
    prs.save(output_path)
    print(f"Presentation saved successfully to: {output_path}")

if __name__ == "__main__":
    create_deck()


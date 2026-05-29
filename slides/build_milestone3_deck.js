// CHIRP Milestone 3 — Preliminary Results deck.
// Palette: "Forest & Moss" (oak woodland — fits the Stanford bird theme).
//
// Numbers match REPORT.md (linear-probe results, +0.013 picture-vs-video lift,
// 6 perfect-F1 species all from Birds-525, 4 zero-F1 species all from iNat).
//
// Usage:
//   node slides/build_milestone3_deck.js
// Output: slides/chirp_milestone3.pptx

const pptxgen = require("pptxgenjs");
const path    = require("path");
const fs      = require("fs");

const REPO     = path.resolve(__dirname, "..");
const FIG_DIR  = path.join(REPO, "outputs", "figures");
const OUT_PPTX = path.join(__dirname, "chirp_milestone3.pptx");

// ---------- design tokens ----------
const COL_PRIMARY   = "2C5F2D";    // deep forest green
const COL_SECONDARY = "97BC62";    // moss
const COL_ACCENT    = "F18F01";    // burnt orange (highlights / numbers)
const COL_DARK      = "1A2E1B";    // near-black green for dark slides
const COL_BG_LIGHT  = "F8F7F2";    // off-white cream
const COL_TEXT      = "1A2A1A";
const COL_TEXT_MUTED = "5A6B5A";
const COL_RED       = "B94A48";    // for low-F1 callouts

const FONT_HEAD = "Georgia";
const FONT_BODY = "Calibri";

// ---------- helpers ----------
function safeImage(slide, name, opts) {
    const p = path.join(FIG_DIR, name);
    if (fs.existsSync(p)) {
        slide.addImage({ path: p, ...opts });
    } else {
        slide.addText(`[missing: ${name}]`, {
            ...opts, fontSize: 10, color: COL_TEXT_MUTED, italic: true,
            align: "center", valign: "middle",
        });
    }
}
function pageNumber(s, n, total) {
    s.addText(`${n} / ${total}`, {
        x: 12.0, y: 7.05, w: 1.0, h: 0.3,
        fontSize: 9, fontFace: FONT_BODY, color: COL_TEXT_MUTED, align: "right",
    });
}
function projectLabel(s) {
    s.addText("CHIRP · Milestone 3", {
        x: 0.5, y: 7.05, w: 4.0, h: 0.3,
        fontSize: 9, fontFace: FONT_BODY, color: COL_TEXT_MUTED,
    });
}
function slideTitle(s, title, subtitle) {
    s.addText(title, {
        x: 0.5, y: 0.4, w: 12.3, h: 0.65,
        fontSize: 32, fontFace: FONT_HEAD, bold: true, color: COL_PRIMARY, margin: 0,
    });
    if (subtitle) {
        s.addText(subtitle, {
            x: 0.5, y: 1.05, w: 12.3, h: 0.4,
            fontSize: 14, fontFace: FONT_BODY, color: COL_TEXT_MUTED, italic: true, margin: 0,
        });
    }
}

// =============================================================
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";    // 13.3 × 7.5
pres.author = "CHIRP project";
pres.title  = "CHIRP — Milestone 3 Preliminary Results";

const TOTAL_SLIDES = 9;

// -------------------------------------------------------------
// 1 — Title
// -------------------------------------------------------------
{
    const s = pres.addSlide();
    s.background = { color: COL_DARK };
    s.addShape(pres.shapes.RECTANGLE, {
        x: 0, y: 0, w: 0.45, h: 7.5,
        fill: { color: COL_ACCENT }, line: { type: "none" },
    });
    s.addText("CHIRP", {
        x: 1.0, y: 1.6, w: 11.0, h: 1.4,
        fontSize: 96, fontFace: FONT_HEAD, bold: true, color: "FFFFFF",
        margin: 0, charSpacing: 6,
    });
    s.addText("Bird species classification on Stanford campus", {
        x: 1.0, y: 3.0, w: 11.0, h: 0.6,
        fontSize: 24, fontFace: FONT_BODY, color: COL_SECONDARY, italic: true, margin: 0,
    });
    s.addText("Milestone 3 — Preliminary Results", {
        x: 1.0, y: 3.7, w: 11.0, h: 0.5,
        fontSize: 18, fontFace: FONT_BODY, color: "FFFFFF", margin: 0,
    });

    const stats = [
        { val: "20",     lbl: "Stanford species" },
        { val: "5,966",  lbl: "samples (3 sources)" },
        { val: "0.671",  lbl: "best val macro-F1 (RF)" },
        { val: "6 / 20", lbl: "species at 1.00 test F1" },
    ];
    stats.forEach((stat, i) => {
        const x = 1.0 + i * 2.85;
        s.addText(stat.val, {
            x, y: 5.0, w: 2.6, h: 0.8,
            fontSize: 48, fontFace: FONT_HEAD, bold: true, color: COL_ACCENT, margin: 0,
        });
        s.addText(stat.lbl, {
            x, y: 5.8, w: 2.6, h: 0.4,
            fontSize: 12, fontFace: FONT_BODY, color: "FFFFFF", margin: 0,
        });
    });
    s.addText("github.com/medha156/CHIRP   ·   2026-05-29", {
        x: 1.0, y: 7.0, w: 11.0, h: 0.3,
        fontSize: 10, fontFace: FONT_BODY, color: COL_SECONDARY, margin: 0,
    });
}

// -------------------------------------------------------------
// 2 — What we built
// -------------------------------------------------------------
{
    const s = pres.addSlide();
    s.background = { color: COL_BG_LIGHT };
    slideTitle(s, "What we built",
        "20-class bird classifier from short video clips, deployable on Stanford campus");

    // Left: architecture boxes
    s.addText("Dual-backbone ensemble", {
        x: 0.5, y: 1.6, w: 5.5, h: 0.35,
        fontSize: 16, fontFace: FONT_HEAD, bold: true, color: COL_PRIMARY, margin: 0,
    });
    const arch = [
        { y: 2.05, title: "Video Swin-T",      body: "16 frames → [B, 768]\nKinetics-400 pretrained" },
        { y: 3.00, title: "EfficientNet-B3",   body: "1–16 keyframes → [B, 1536]\nImageNet pretrained" },
        { y: 3.95, title: "Fusion MLP",        body: "concat → 2304 → 512 → 20", highlight: true },
    ];
    arch.forEach(b => {
        s.addShape(pres.shapes.RECTANGLE, {
            x: 0.5, y: b.y, w: 5.5, h: 0.85,
            fill: { color: b.highlight ? COL_PRIMARY : "FFFFFF" },
            line: { color: COL_PRIMARY, width: 1.5 },
        });
        s.addText(b.title, {
            x: 0.7, y: b.y + 0.05, w: 5.1, h: 0.35,
            fontSize: 14, fontFace: FONT_HEAD, bold: true,
            color: b.highlight ? "FFFFFF" : COL_PRIMARY, margin: 0,
        });
        s.addText(b.body, {
            x: 0.7, y: b.y + 0.42, w: 5.1, h: 0.4,
            fontSize: 11, fontFace: FONT_BODY,
            color: b.highlight ? COL_SECONDARY : COL_TEXT, margin: 0,
        });
    });

    // Right: bulleted milestones
    s.addText("Pipeline milestones", {
        x: 6.5, y: 1.6, w: 6.3, h: 0.35,
        fontSize: 16, fontFace: FONT_HEAD, bold: true, color: COL_PRIMARY, margin: 0,
    });
    const ms = [
        "Data pipeline: VideoDataset, preprocess, augment, optical flow",
        "Models: Swin-T, EB3, fusion head, classical baselines (KNN/RF/XGB)",
        "Training: AdamW + cosine warm-up + early stop + WandB",
        "Eval: confusion matrix, per-class F1, SHAP, GradCAM + rollout",
        "Experiments: ablation sweep + picture-vs-video temporal sweep",
        "Infra: 75 unit tests + GitHub Actions CI + REPORT.md",
    ];
    s.addText(ms.map((t, i) => ({
        text: t,
        options: { bullet: { code: "25A0" }, color: COL_TEXT, breakLine: i < ms.length - 1 },
    })), {
        x: 6.5, y: 2.05, w: 6.3, h: 4.5,
        fontSize: 13, fontFace: FONT_BODY, paraSpaceAfter: 8,
    });

    projectLabel(s); pageNumber(s, 2, TOTAL_SLIDES);
}

// -------------------------------------------------------------
// 3 — Dataset (3 free sources, 20/20 coverage)
// -------------------------------------------------------------
{
    const s = pres.addSlide();
    s.background = { color: COL_BG_LIGHT };
    slideTitle(s, "Dataset — built from 3 free public sources",
        "Original plan (FBD-SV-2024 + VB100) had to be rebuilt: FBD has no species labels, iNat has no videos");

    const stats = [
        { v: "20 / 20", l: "Stanford species covered", c: COL_PRIMARY },
        { v: "5,966",   l: "total samples",            c: COL_PRIMARY },
        { v: "76",      l: "real video clips (VB100)", c: COL_ACCENT  },
        { v: "5,890",   l: "photos (Birds-525 + iNat)", c: COL_ACCENT  },
    ];
    stats.forEach((stat, i) => {
        const x = 0.5 + i * 3.2;
        s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
            x, y: 1.6, w: 2.95, h: 1.4,
            fill: { color: "FFFFFF" }, line: { color: COL_SECONDARY, width: 1 },
            rectRadius: 0.08,
        });
        s.addText(stat.v, {
            x, y: 1.7, w: 2.95, h: 0.7,
            fontSize: 36, fontFace: FONT_HEAD, bold: true, color: stat.c,
            align: "center", margin: 0,
        });
        s.addText(stat.l, {
            x, y: 2.4, w: 2.95, h: 0.45,
            fontSize: 11, fontFace: FONT_BODY, color: COL_TEXT_MUTED,
            align: "center", margin: 0,
        });
    });

    safeImage(s, "merged_class_distribution.png", {
        x: 0.5, y: 3.2, w: 8.0, h: 3.7,
    });

    s.addText("Sources & coverage", {
        x: 8.7, y: 3.2, w: 4.4, h: 0.3,
        fontSize: 14, fontFace: FONT_HEAD, bold: true, color: COL_PRIMARY, margin: 0,
    });
    s.addText([
        { text: "VB100 ",       options: { bold: true, color: COL_ACCENT  } },
        { text: "(Zenodo, CC BY-NC-SA): 5 species, ~14 video clips each",
          options: { breakLine: true } },
        { text: "Birds-525 ",   options: { bold: true, color: COL_PRIMARY } },
        { text: "(HuggingFace, CC0): 8 species, ≤150 photos each",
          options: { breakLine: true } },
        { text: "iNaturalist ", options: { bold: true, color: COL_PRIMARY } },
        { text: "(API, CC*): 8 gap species, ~590 Bay Area photos each",
          options: { breakLine: true } },
        { text: " ",            options: { breakLine: true, fontSize: 6 } },
        { text: "For Milestone 3 we subsampled to 538 (30/class) so the sweep would fit on CPU.",
          options: { italic: true, color: COL_TEXT_MUTED } },
    ], {
        x: 8.7, y: 3.55, w: 4.4, h: 3.35, fontSize: 11, fontFace: FONT_BODY,
        paraSpaceAfter: 4,
    });

    projectLabel(s); pageNumber(s, 3, TOTAL_SLIDES);
}

// -------------------------------------------------------------
// 4 — Results headline
// -------------------------------------------------------------
{
    const s = pres.addSlide();
    s.background = { color: COL_BG_LIGHT };
    slideTitle(s, "Results — Random Forest leads on the mini-sweep",
        "3 experiments completed on CPU; Swin-T + Fusion deferred to GPU");

    // Big number card
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
        x: 0.5, y: 1.7, w: 5.0, h: 5.3,
        fill: { color: COL_DARK }, line: { type: "none" }, rectRadius: 0.1,
    });
    s.addText("0.671", {
        x: 0.5, y: 2.25, w: 5.0, h: 1.6,
        fontSize: 96, fontFace: FONT_HEAD, bold: true, color: COL_ACCENT,
        align: "center", margin: 0,
    });
    s.addText("best val macro-F1", {
        x: 0.5, y: 3.7, w: 5.0, h: 0.5,
        fontSize: 22, fontFace: FONT_BODY, color: "FFFFFF",
        align: "center", margin: 0,
    });
    s.addText("Random Forest on frozen EB3 features (one-shot fit)", {
        x: 0.5, y: 4.15, w: 5.0, h: 0.4,
        fontSize: 12, fontFace: FONT_BODY, color: COL_SECONDARY,
        align: "center", italic: true, margin: 0,
    });

    s.addText([
        { text: "13× ",    options: { bold: true, color: COL_ACCENT } },
        { text: "above 20-class chance (0.050)",
          options: { color: "FFFFFF", breakLine: true } },
        { text: "+0.11 ",  options: { bold: true, color: COL_ACCENT } },
        { text: "over linear-probe EB3 (0.549)",
          options: { color: "FFFFFF", breakLine: true } },
        { text: "6 / 20 ", options: { bold: true, color: COL_ACCENT } },
        { text: "species at perfect 1.00 test F1 (EB3)",
          options: { color: "FFFFFF" } },
    ], {
        x: 0.7, y: 4.7, w: 4.6, h: 2.0, fontSize: 14, fontFace: FONT_BODY,
        paraSpaceAfter: 4, margin: 0,
    });

    safeImage(s, "mini_per_experiment_f1.png", { x: 5.8, y: 1.7, w: 7.2, h: 5.0 });

    projectLabel(s); pageNumber(s, 4, TOTAL_SLIDES);
}

// -------------------------------------------------------------
// 5 — Per-class F1 — the actual story
// -------------------------------------------------------------
{
    const s = pres.addSlide();
    s.background = { color: COL_BG_LIGHT };
    slideTitle(s, "Per-class F1 reveals the real story",
        "EB3 T=4 linear-probe, test split (81 samples) — performance correlates with data source");

    safeImage(s, "mini_per_class_f1_eb3_T4.png", { x: 0.4, y: 1.55, w: 8.2, h: 5.6 });

    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
        x: 8.9, y: 1.55, w: 4.1, h: 5.6,
        fill: { color: "FFFFFF" }, line: { color: COL_SECONDARY, width: 1 },
        rectRadius: 0.08,
    });
    s.addText("Insights", {
        x: 9.1, y: 1.7, w: 3.7, h: 0.4,
        fontSize: 16, fontFace: FONT_HEAD, bold: true, color: COL_PRIMARY, margin: 0,
    });
    s.addText([
        { text: "6 perfect-F1 species (1.00)", options: { bold: true, color: COL_PRIMARY, breakLine: true } },
        { text: "ALL from Birds-525 — curated single-bird photos. ImageNet features transfer perfectly here.",
          options: { color: COL_TEXT, breakLine: true, fontSize: 10 } },
        { text: " ", options: { breakLine: true, fontSize: 6 } },
        { text: "Middle tier (0.18 – 0.89)", options: { bold: true, color: COL_PRIMARY, breakLine: true } },
        { text: "VB100 video species + a few iNat species — model partially generalizes.",
          options: { color: COL_TEXT, breakLine: true, fontSize: 10 } },
        { text: " ", options: { breakLine: true, fontSize: 6 } },
        { text: "4 zero-F1 species (0.00)", options: { bold: true, color: COL_RED, breakLine: true } },
        { text: "ALL from iNaturalist — cluttered citizen-science photos. Frozen ImageNet features don't generalize without fine-tuning.",
          options: { color: COL_TEXT, breakLine: true, fontSize: 10 } },
        { text: " ", options: { breakLine: true, fontSize: 6 } },
        { text: "Takeaway: data source dominates model architecture at this scale.",
          options: { italic: true, color: COL_TEXT_MUTED, fontSize: 10 } },
    ], {
        x: 9.1, y: 2.15, w: 3.7, h: 4.9, fontSize: 11, fontFace: FONT_BODY,
        paraSpaceAfter: 4, margin: 0,
    });

    projectLabel(s); pageNumber(s, 5, TOTAL_SLIDES);
}

// -------------------------------------------------------------
// 6 — Picture-vs-video comparison
// -------------------------------------------------------------
{
    const s = pres.addSlide();
    s.background = { color: COL_BG_LIGHT };
    slideTitle(s, "Picture vs video — small lift, explainable",
        "EfficientNet-B3: T=1 (picture) → T=4 (frames) on the mini sweep");

    safeImage(s, "mini_picture_vs_video.png", { x: 0.4, y: 1.55, w: 7.8, h: 4.7 });

    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
        x: 8.5, y: 1.55, w: 4.5, h: 4.7,
        fill: { color: "FFFFFF" }, line: { color: COL_SECONDARY, width: 1 },
        rectRadius: 0.08,
    });
    s.addText("Why the lift is small", {
        x: 8.7, y: 1.7, w: 4.1, h: 0.4,
        fontSize: 16, fontFace: FONT_HEAD, bold: true, color: COL_PRIMARY, margin: 0,
    });
    s.addText([
        { text: "Only ", options: { color: COL_TEXT } },
        { text: "5 of 20", options: { bold: true, color: COL_ACCENT } },
        { text: " species have actual video clips (the VB100 subset).",
          options: { color: COL_TEXT, breakLine: true } },
        { text: " ", options: { breakLine: true, fontSize: 6 } },
        { text: "The other 15 species are photos. At T=4 the model sees 4 augmented copies of the same image — no real temporal information.",
          options: { color: COL_TEXT, breakLine: true } },
        { text: " ", options: { breakLine: true, fontSize: 6 } },
        { text: "So the +0.013 F1 reflects mostly augmentation noise, not temporal learning.",
          options: { color: COL_TEXT, italic: true, breakLine: true } },
        { text: " ", options: { breakLine: true, fontSize: 6 } },
        { text: "Cleaner experiment ", options: { bold: true, color: COL_PRIMARY } },
        { text: "would restrict to the 5 VB100 species, OR request Macaulay Library videos for the missing 15.",
          options: { color: COL_TEXT } },
    ], {
        x: 8.7, y: 2.15, w: 4.1, h: 4.0, fontSize: 12, fontFace: FONT_BODY,
        paraSpaceAfter: 4, margin: 0,
    });

    // Bottom takeaway band (above footer)
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
        x: 0.5, y: 6.4, w: 12.3, h: 0.55,
        fill: { color: COL_DARK }, line: { type: "none" }, rectRadius: 0.06,
    });
    s.addText([
        { text: "Picture-vs-video lift on the mini sweep: ", options: { color: "FFFFFF" } },
        { text: "+0.013 F1", options: { bold: true, color: COL_ACCENT } },
        { text: " — bounded by dataset composition, not architecture.",
          options: { color: "FFFFFF" } },
    ], {
        x: 0.75, y: 6.45, w: 11.9, h: 0.45, fontSize: 13, fontFace: FONT_BODY,
        valign: "middle", margin: 0,
    });

    projectLabel(s); pageNumber(s, 6, TOTAL_SLIDES);
}

// -------------------------------------------------------------
// 7 — Analysis (what's working / what isn't)
// -------------------------------------------------------------
{
    const s = pres.addSlide();
    s.background = { color: COL_BG_LIGHT };
    slideTitle(s, "Analysis — what's working, what isn't",
        "Reading between the per-class numbers");

    // LEFT — Working
    s.addShape(pres.shapes.RECTANGLE, {
        x: 0.5, y: 1.55, w: 0.12, h: 5.4,
        fill: { color: COL_PRIMARY }, line: { type: "none" },
    });
    s.addText("✓  What's working", {
        x: 0.85, y: 1.55, w: 5.5, h: 0.4,
        fontSize: 18, fontFace: FONT_HEAD, bold: true, color: COL_PRIMARY, margin: 0,
    });
    const working = [
        ["Pipeline produces real learning",
         "All 3 experiments land 10-13× above random chance."],
        ["ImageNet features transfer on curated photos",
         "6 Birds-525 species hit perfect F1 with zero gradient updates to the backbone."],
        ["Random Forest is a strong baseline on frozen features",
         "0.671 val F1 beats the linear probe (0.565) by +0.11 — non-linear head matters."],
        ["Mixed video + photo input flows end-to-end",
         "Same training loop handles VB100 mp4, Birds-525 JPGs, iNat JPGs without per-source branching."],
    ];
    working.forEach((it, i) => {
        const y = 2.05 + i * 1.2;
        s.addText(it[0], {
            x: 0.85, y, w: 5.5, h: 0.35,
            fontSize: 13, fontFace: FONT_BODY, bold: true, color: COL_PRIMARY, margin: 0,
        });
        s.addText(it[1], {
            x: 0.85, y: y + 0.35, w: 5.5, h: 0.75,
            fontSize: 11, fontFace: FONT_BODY, color: COL_TEXT, margin: 0,
        });
    });

    // RIGHT — Not working
    s.addShape(pres.shapes.RECTANGLE, {
        x: 6.8, y: 1.55, w: 0.12, h: 5.4,
        fill: { color: COL_RED }, line: { type: "none" },
    });
    s.addText("✗  What isn't (yet)", {
        x: 7.15, y: 1.55, w: 5.7, h: 0.4,
        fontSize: 18, fontFace: FONT_HEAD, bold: true, color: COL_RED, margin: 0,
    });
    const notWorking = [
        ["iNaturalist photo classes are completely missed",
         "4 species (Chickadee, Bushtit, Oak Titmouse, Yellow-rumped Warbler) at 0.0 F1 — frozen ImageNet doesn't generalize to cluttered citizen-science photos."],
        ["Picture-vs-video lift is tiny",
         "Only +0.013 F1 — because 15 of 20 species are photos with no real temporal info."],
        ["Video / Fusion experiments couldn't complete",
         "Swin-T at T=8 hits 25+ sec/batch on CPU. Deferred to GPU."],
        ["Backbone fine-tuning untested",
         "All ran with frozen backbone. Full fine-tune likely closes the iNat gap but needs GPU."],
    ];
    notWorking.forEach((it, i) => {
        const y = 2.05 + i * 1.2;
        s.addText(it[0], {
            x: 7.15, y, w: 5.7, h: 0.35,
            fontSize: 13, fontFace: FONT_BODY, bold: true, color: COL_RED, margin: 0,
        });
        s.addText(it[1], {
            x: 7.15, y: y + 0.35, w: 5.7, h: 0.75,
            fontSize: 11, fontFace: FONT_BODY, color: COL_TEXT, margin: 0,
        });
    });

    projectLabel(s); pageNumber(s, 7, TOTAL_SLIDES);
}

// -------------------------------------------------------------
// 8 — Limitations
// -------------------------------------------------------------
{
    const s = pres.addSlide();
    s.background = { color: COL_BG_LIGHT };
    slideTitle(s, "Limitations — what you should not over-read",
        "Honest caveats on the 0.671 headline");

    const lims = [
        {
            n: "1", title: "Sample size",
            body: "538-sample subset of the 5,966 we have. Full-data numbers should match or improve. Test split = 81 samples; per-class noise is non-trivial.",
        },
        {
            n: "2", title: "Class imbalance",
            body: "5 VB100-only classes have 13–18 samples vs 30 elsewhere. WeightedRandomSampler helps but the per-class variance is still meaningful.",
        },
        {
            n: "3", title: "CPU compute ceiling",
            body: "Only frozen-backbone experiments completed. Full fine-tune + Swin-T + Fusion all need GPU. No multi-frame results beyond T=4.",
        },
        {
            n: "4", title: "Picture-vs-video story is incomplete",
            body: "Only 5 of 20 species have real video. The +0.013 F1 lift between T=1 and T=4 is not a clean test of temporal modelling.",
        },
        {
            n: "5", title: "Train / test domain mix",
            body: "Birds-525 (clean) and iNaturalist (cluttered) photos have different distributions. The model is strongest on its training distribution.",
        },
    ];
    lims.forEach((l, i) => {
        const y = 1.55 + i * 1.1;
        s.addShape(pres.shapes.OVAL, {
            x: 0.5, y: y + 0.1, w: 0.7, h: 0.7,
            fill: { color: COL_ACCENT }, line: { type: "none" },
        });
        s.addText(l.n, {
            x: 0.5, y: y + 0.1, w: 0.7, h: 0.7,
            fontSize: 28, fontFace: FONT_HEAD, bold: true, color: "FFFFFF",
            align: "center", valign: "middle", margin: 0,
        });
        s.addText(l.title, {
            x: 1.4, y: y + 0.05, w: 11.5, h: 0.4,
            fontSize: 15, fontFace: FONT_HEAD, bold: true, color: COL_PRIMARY, margin: 0,
        });
        s.addText(l.body, {
            x: 1.4, y: y + 0.45, w: 11.5, h: 0.55,
            fontSize: 12, fontFace: FONT_BODY, color: COL_TEXT, margin: 0,
        });
    });

    projectLabel(s); pageNumber(s, 8, TOTAL_SLIDES);
}

// -------------------------------------------------------------
// 9 — Next steps
// -------------------------------------------------------------
{
    const s = pres.addSlide();
    s.background = { color: COL_DARK };

    s.addText("Next steps", {
        x: 0.5, y: 0.4, w: 12.3, h: 0.7,
        fontSize: 38, fontFace: FONT_HEAD, bold: true, color: COL_ACCENT, margin: 0,
    });
    s.addText("In priority order — pipeline is ready, compute is the bottleneck", {
        x: 0.5, y: 1.05, w: 12.3, h: 0.4,
        fontSize: 14, fontFace: FONT_BODY, color: COL_SECONDARY, italic: true, margin: 0,
    });

    const steps = [
        {
            n: "1",
            title: "Rent a GPU window (~$5 on Lambda Labs / AWS spot, ~6 hrs)",
            body: "Full fine-tuning of EB3, Swin-T baseline, Fusion ensemble — fills every TBD cell in the results table.",
        },
        {
            n: "2",
            title: "Train on full 5,966 samples instead of the 538 subset",
            body: "Better-calibrated F1; less noisy per-class numbers; full-data confusion matrix.",
        },
        {
            n: "3",
            title: "Request Macaulay Library video access",
            body: "Free for research. Would give real video for the 15 photo-only species and enable a clean picture-vs-video test.",
        },
        {
            n: "4",
            title: "GradCAM + SHAP analysis on the trained model",
            body: "Diagnose the iNaturalist zero-F1 species at the pixel level.",
        },
        {
            n: "5",
            title: "Optical-flow + Swin ablation on the 5 video species",
            body: "Test whether 5-channel input justifies the 1.5× compute cost.",
        },
    ];
    steps.forEach((stp, i) => {
        const y = 1.7 + i * 1.05;
        s.addShape(pres.shapes.OVAL, {
            x: 0.5, y: y + 0.1, w: 0.7, h: 0.7,
            fill: { color: COL_ACCENT }, line: { type: "none" },
        });
        s.addText(stp.n, {
            x: 0.5, y: y + 0.1, w: 0.7, h: 0.7,
            fontSize: 28, fontFace: FONT_HEAD, bold: true, color: COL_DARK,
            align: "center", valign: "middle", margin: 0,
        });
        s.addText(stp.title, {
            x: 1.4, y: y + 0.05, w: 11.5, h: 0.4,
            fontSize: 15, fontFace: FONT_HEAD, bold: true, color: "FFFFFF", margin: 0,
        });
        s.addText(stp.body, {
            x: 1.4, y: y + 0.45, w: 11.5, h: 0.55,
            fontSize: 12, fontFace: FONT_BODY, color: COL_SECONDARY, margin: 0,
        });
    });

    s.addText("github.com/medha156/CHIRP   ·   REPORT.md has the full numbers + reproduction commands", {
        x: 0.5, y: 7.05, w: 12.3, h: 0.3,
        fontSize: 10, fontFace: FONT_BODY, color: COL_SECONDARY, italic: true, margin: 0,
    });
}

// =============================================================
pres.writeFile({ fileName: OUT_PPTX })
    .then(p => console.log(`Wrote ${p}`))
    .catch(e => { console.error(e); process.exit(1); });

# INTELLIGATE — MILESTONE 3C.1 EXPERIMENT REPORT
## Real Pakistani License Plate OCR Scientific Evaluation

**Author:** Hussain Raza (SP23-BAI-057)  
**Academic Role:** BS Artificial Intelligence FYP Lead Researcher  
**Dataset Reference:** 50 Manually Annotated & Verified Genuine Pakistani Plate Crops  
**Annotation Manifest:** [`data/pakistan_plates/annotations/ocr_annotation_manifest_verified.csv`](file:///d:/MZHR_FYP/INTELLIGATE/data/pakistan_plates/annotations/ocr_annotation_manifest_verified.csv)  
**Machine-Readable Evaluation Results:** [`logs/benchmarks/m03c_1_ocr_evaluation_results.json`](file:///d:/MZHR_FYP/INTELLIGATE/logs/benchmarks/m03c_1_ocr_evaluation_results.json)  
**Error Analysis Artifact:** [`logs/benchmarks/m03c_1_ocr_failures.csv`](file:///d:/MZHR_FYP/INTELLIGATE/logs/benchmarks/m03c_1_ocr_failures.csv)

---

## 1. Executive Summary & Experimental Methodology

In this milestone, 50 genuine Pakistani license plate crops extracted from real Pakistani vehicle imagery were evaluated using the active EasyOCR baseline. Each crop was previously inspected and assigned human-verified ground truth text without reliance on synthetic test generation.

Two distinct evaluation protocols were executed:
1. **Evaluation A — Raw Exact Transcription:** Evaluates the exact raw output string against human ground truth, preserving exact casing, spacing, and hyphenation separators.
2. **Evaluation B — Normalized Registration Text:** Evaluates the canonical registration text under the project's standard Pakistani plate normalization policy:
   - Convert all characters to uppercase.
   - Remove hyphens, spaces, and punctuation (`[^A-Za-z0-9]`).
   - Retain every meaningful letter and digit (specifically preserving 2-digit registration year codes and 2nd-line digits).
   - Character Error Rate (CER) computed using the standard Levenshtein distance formulation:
     $$\text{CER} = \frac{\sum (S + D + I)}{\sum N_{\text{ref}}} \times 100\%$$
     where $S$ is substitutions, $D$ is deletions, $I$ is insertions, and $N_{\text{ref}}$ is total ground-truth characters.

---

## 2. Independent Evaluation Results

### Table 1: EasyOCR Performance on Real Pakistani Plate Crops ($N = 50$)

| Evaluation Protocol | Exact Matches | Exact Accuracy (%) | Total Edits | Reference Chars | CER (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Evaluation A: Raw Exact Transcription** | 13 / 50 | **26.00%** | 128 | 437 | **29.29%** |
| **Evaluation B: Normalized Registration Text** | 18 / 50 | **36.00%** | 86 | 362 | **23.76%** |

### Independent Reproduction of Preliminary Estimates:
- **Preliminary Reported Figures:** 18 / 50 (36.00%) exact matches, ~23.76% CER.
- **Independently Reproduced Figures:** 18 / 50 (**36.00%**) exact matches, **23.76%** CER (86 edits / 362 reference characters).
- **Exact Reproduction Delta:** **0.00%** difference; preliminary estimates are 100% verified.

---

## 3. Levenshtein Edit Operation Breakdown

Out of 86 total character edit operations across the 50 test samples:
- **Deletions ($D$):** **65 (75.58%)** — Character omissions are by far the dominant failure mode.
- **Substitutions ($S$):** **17 (19.77%)** — Character misidentifications.
- **Insertions ($I$):** **4 (4.65%)** — Spurious artifact insertions.
- **Empty OCR Predictions:** 2 / 50 samples (Sample ID 30 `AJJ-432`, Sample ID 41 `BW-411`).

---

## 4. Confidence Distribution and Calibration

EasyOCR confidence scores were recorded per prediction:
- **Overall Mean Confidence:** 0.8545 (85.45%)
- **Correct Predictions ($N = 18$):** Mean Confidence = **0.9614 (96.14%)**
- **Incorrect Predictions ($N = 32$):** Mean Confidence = **0.7944 (79.44%)**

> [!WARNING]
> **Confidence Miscalibration Risk:** While correct predictions average high confidence ($0.9614$), several incorrect predictions also exhibit high confidence ($>0.85$), such as Sample ID 4 (`LEF-1981` instead of `LEF-12-1981`, conf = 0.995). The missing year code is completely omitted, but EasyOCR assigns near-perfect confidence to the characters it did see. Therefore, **raw OCR confidence alone is insufficient for gate access authorization** without strict exact-match registration verification.

---

## 5. Granular Categorical Breakdowns

### 5.1 Results by Jurisdiction / Province

| Province / Jurisdiction | Sample Count | Exact Matches | Exact Accuracy (%) | Total Edits | Reference Chars | CER (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Punjab** | 40 | 17 | **42.50%** | 62 | 307 | **20.20%** |
| **Islamabad Capital Territory (ICT)** | 10 | 1 | **10.00%** | 24 | 55 | **43.64%** |

*Analysis:* ICT plates exhibit markedly lower accuracy (10.00% vs 42.50%) and double the error rate (43.64% vs 20.20%). ICT plates frequently feature smaller fonts, different letter aspect ratios, and lower crop resolutions in the dataset ($93\times 30$, $81\times 75$), causing CRAFT text-detection dropouts.

### 5.2 Results by Plate Aspect Ratio (1-Line vs 2-Line Formats)

| Plate Format | Aspect Ratio ($W/H$) | Sample Count | Exact Matches | Exact Accuracy (%) | CER (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1-Line (Horizontal)** | $\ge 2.0$ | 14 | 6 | **42.86%** | **16.47%** |
| **2-Line (Stacked / Square)** | $< 2.0$ | 36 | 12 | **33.33%** | **25.99%** |

*Analysis:* 2-line plates suffer from higher CER (25.99% vs 16.47%) primarily due to vertical reading order inversion and small registration year characters printed on the top line.

---

## 6. Systematic Error Analysis & Failure Modes

All 32 incorrect predictions were categorized into discrete failure classes:

```
[Total OCR Failures: 32]
  ├── DELETION_ONLY: 14 (43.8%) ────────► Missing small year codes (e.g., LEF-12-3503 -> LEF-3503)
  ├── YEAR_DROP + CONFUSION: 7 (21.9%) ──► Dropped year + misread char (e.g., MNE-16-254 -> NNE-254)
  ├── SUBSTITUTION_ONLY: 6 (18.8%) ──────► Letter/digit confusion (e.g., UZ-016 -> U2016, B -> 8)
  ├── EMPTY_OUTPUT: 2 (6.2%) ────────────► Low resolution / dark crop (AJJ-432, BW-411)
  ├── COMPLEX_MISREAD: 2 (6.2%) ─────────► Extreme blur / partial occlusion
  └── INSERTION_ONLY: 1 (3.1%) ──────────► Border/screw artifact detected as character
```

### Table 2: Representative Pakistani Plate OCR Failure Cases

| ID | Province | Dimensions | Ground Truth | EasyOCR Prediction | Error Category | Conf | Key Error Mechanism |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | Punjab | 433×247 | `LEF-12-3503` | `LEF-3503` | DELETION_ONLY | 0.960 | Dropped small top-line year digits (`12`). |
| **2** | Punjab | 389×220 | `MNE-16-254` | `NNE-254` | YEAR_DROP + CONFUSION | 0.641 | Dropped year (`16`) and substituted `M` $\to$ `N`. |
| **4** | Punjab | 335×191 | `LEF-12-1981` | `LEF-1981` | DELETION_ONLY | 0.995 | Dropped year (`12`); very high confidence on remainder. |
| **6** | Punjab | 245×145 | `LEB-14-5542` | `5542` | DELETION_ONLY | 1.000 | Completely omitted top line (`LEB-14`), read only bottom line. |
| **7** | Punjab | 145×62 | `MN-1821` | `HN-1821` | SUBSTITUTION_ONLY | 0.669 | Substituted `M` $\to$ `H` due to vertical stroke confusion. |
| **9** | Punjab | 296×176 | `MNA-18-877` | `MNA18827` | SUBSTITUTION_ONLY | 0.982 | Inserted extra digit `2` from plate border reflection. |
| **21**| Punjab | 275×164 | `FSA-5750` | `5750FSA` | INVERTED_ORDER | 0.880 | Inverted 2-line reading order (bottom read before top). |
| **30**| ICT | 190×143 | `AJJ-432` | *(empty)* | EMPTY_OUTPUT | 0.000 | Low contrast and dark background caused detection failure. |
| **37**| ICT | 93×30 | `UZ-016` | `U2016` | SUBSTITUTION_ONLY | 0.939 | Substituted `Z` $\to$ `2` (common alphanumeric confusion). |
| **49**| ICT | 188×82 | `AVB-907` | `AV-8907` | SUBSTITUTION_ONLY | 0.997 | Substituted `B` $\to$ `8` and inserted dash. |

### Summary of Error Dynamics:
1. **65.7% of all failure cases involve missing registration year codes.** Standard Pakistani plates feature small 2-digit year codes (e.g. `12`, `14`, `16`) printed in smaller font sizes. EasyOCR's character segmentation filters these out as noise or small artifacts.
2. **Two-line reading order ambiguity:** EasyOCR sorts bounding boxes horizontally first, occasionally reading bottom numeric lines before top district lines (e.g. `5750FSA` instead of `FSA5750`).
3. **Alphanumeric character confusion:** Common confusions include $M \leftrightarrow N$, $M \leftrightarrow H$, $Z \leftrightarrow 2$, $B \leftrightarrow 8$, and $O \leftrightarrow 0$.

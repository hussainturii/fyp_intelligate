"""
Generate a self-contained, responsive HTML visual review gallery for the 50 Pakistani plate crops.
Allows the human reviewer to:
- Visually inspect high-resolution plate crops
- Compare against EasyOCR prediction and confidence
- Directly edit/fill ground truth text and jurisdiction
- Mark as VALIDATED, AMBIGUOUS, or UNREADABLE
- Export the updated CSV directly from the browser!
"""

import base64
import csv
from pathlib import Path


def generate_html():
    root = Path(__file__).resolve().parent.parent
    manifest_csv = root / "data" / "pakistan_plates" / "ocr_annotation_manifest.csv"
    output_html = root / "tools" / "ocr_review_gallery.html"

    with open(manifest_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    cards_html = []
    for r in rows:
        crop_path = root / r["crop_relative_path"]
        if crop_path.exists():
            with open(crop_path, "rb") as img_f:
                b64_data = base64.b64encode(img_f.read()).decode("utf-8")
                img_src = f"data:image/jpeg;base64,{b64_data}"
        else:
            img_src = ""

        sample_id = r["sample_id"]
        pred_text = r["easyocr_predicted_text"]
        conf = r["easyocr_confidence"]
        source = r["source_image"]
        bbox = r["plate_bbox_xyxy"]

        cards_html.append(f"""
        <div class="crop-card" id="card-{sample_id}">
            <div class="crop-header">
                <span class="badge">#{sample_id}</span>
                <span class="source-img" title="{source}">{source[:24]}...</span>
            </div>
            <div class="img-container">
                <img src="{img_src}" alt="Plate crop #{sample_id}" />
            </div>
            <div class="meta-row">
                <div class="meta-label">EasyOCR Guess:</div>
                <div class="meta-val"><strong>{pred_text if pred_text else '<em>(None)</em>'}</strong> ({conf})</div>
            </div>
            <div class="form-group">
                <label>Verified Ground Truth Text:</label>
                <input type="text" id="gt-{sample_id}" value="{pred_text}" class="input-text" />
            </div>
            <div class="form-group">
                <label>Province / Jurisdiction:</label>
                <select id="prov-{sample_id}" class="select-box">
                    <option value="">Unknown / Federal</option>
                    <option value="Punjab" selected>Punjab</option>
                    <option value="Sindh">Sindh</option>
                    <option value="ICT">Islamabad (ICT)</option>
                    <option value="KPK">Khyber Pakhtunkhwa</option>
                    <option value="Balochistan">Balochistan</option>
                </select>
            </div>
            <div class="status-group">
                <label><input type="radio" name="status-{sample_id}" value="HUMAN_VERIFIED" checked> Verified</label>
                <label><input type="radio" name="status-{sample_id}" value="AMBIGUOUS"> Ambiguous</label>
                <label><input type="radio" name="status-{sample_id}" value="UNREADABLE"> Unreadable</label>
            </div>
        </div>
        """)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>IntelliGate — Real Pakistani Plate OCR Review Gallery</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        background: #0f172a;
        color: #f1f5f9;
        margin: 0;
        padding: 24px;
    }}
    .header {{
        max-width: 1400px;
        margin: 0 auto 24px auto;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #334155;
        padding-bottom: 16px;
    }}
    .title h1 {{
        margin: 0 0 6px 0;
        font-size: 24px;
        color: #38bdf8;
    }}
    .title p {{
        margin: 0;
        color: #94a3b8;
        font-size: 14px;
    }}
    .btn {{
        background: #0284c7;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 6px;
        font-size: 14px;
        font-weight: 600;
        cursor: pointer;
        transition: background 0.2s;
    }}
    .btn:hover {{
        background: #0369a1;
    }}
    .gallery-grid {{
        max-width: 1400px;
        margin: 0 auto;
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
        gap: 20px;
    }}
    .crop-card {{
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        display: flex;
        flex-direction: column;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }}
    .crop-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }}
    .badge {{
        background: #0369a1;
        color: #e0f2fe;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 12px;
    }}
    .source-img {{
        font-size: 12px;
        color: #64748b;
        font-family: monospace;
    }}
    .img-container {{
        background: #090d16;
        border: 1px solid #1e293b;
        border-radius: 6px;
        height: 120px;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        margin-bottom: 12px;
    }}
    .img-container img {{
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
        image-rendering: pixelated;
    }}
    .meta-row {{
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        margin-bottom: 12px;
        background: #0f172a;
        padding: 8px;
        border-radius: 4px;
    }}
    .meta-label {{
        color: #94a3b8;
    }}
    .meta-val {{
        color: #38bdf8;
    }}
    .form-group {{
        margin-bottom: 10px;
    }}
    .form-group label {{
        display: block;
        font-size: 12px;
        color: #94a3b8;
        margin-bottom: 4px;
    }}
    .input-text, .select-box {{
        width: 100%;
        box-sizing: border-box;
        background: #0f172a;
        border: 1px solid #475569;
        color: #f8fafc;
        padding: 8px;
        border-radius: 4px;
        font-size: 14px;
    }}
    .status-group {{
        display: flex;
        justify-content: space-between;
        font-size: 12px;
        color: #cbd5e1;
        margin-top: 8px;
    }}
    .status-group label {{
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 4px;
    }}
</style>
</head>
<body>
<div class="header">
    <div class="title">
        <h1>IntelliGate — Real Pakistani Plate OCR Review</h1>
        <p>Milestone 3A.3 Human-in-the-Loop Verification (50 Real Crops)</p>
    </div>
    <div>
        <button class="btn" onclick="exportCSV()">Export Verified CSV</button>
    </div>
</div>

<div class="gallery-grid">
    {''.join(cards_html)}
</div>

<script>
const originalData = {rows};

function exportCSV() {{
    let csv = "sample_id,source_image,crop_relative_path,plate_bbox_xyxy,easyocr_predicted_text,easyocr_confidence,human_verified_gt_text,jurisdiction_province,verified_status\\n";
    
    for (let r of originalData) {{
        let sid = r.sample_id;
        let gt = document.getElementById("gt-" + sid).value.trim();
        let prov = document.getElementById("prov-" + sid).value;
        let statusEls = document.getElementsByName("status-" + sid);
        let status = "HUMAN_VERIFIED";
        for (let el of statusEls) {{
            if (el.checked) status = el.value;
        }}
        
        let row = [
            sid,
            `"${{r.source_image}}"`,
            `"${{r.crop_relative_path}}"`,
            `"${{r.plate_bbox_xyxy}}"`,
            `"${{r.easyocr_predicted_text}}"`,
            r.easyocr_confidence,
            `"${{gt}}"`,
            `"${{prov}}"`,
            status
        ];
        csv += row.join(",") + "\\n";
    }}
    
    let blob = new Blob([csv], {{ type: "text/csv;charset=utf-8;" }});
    let url = URL.createObjectURL(blob);
    let link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", "ocr_annotation_manifest_verified.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}}
</script>
</body>
</html>
"""
    output_html.write_text(html_content, encoding="utf-8")
    print(f"[SUCCESS] Generated visual OCR review gallery: {output_html}")


if __name__ == "__main__":
    generate_html()

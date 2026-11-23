"""Export endpoints for optimization results (CSV, Excel, PDF)."""

from __future__ import annotations

import io
import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.db import OptimizationJob, User
from backend.deps import db_session, optional_user

router = APIRouter(prefix="/api/results", tags=["exports"])


def _get_job(job_id: str, session: Session, user: User | None) -> OptimizationJob:
    job = session.get(OptimizationJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if user and job.user_id and job.user_id != user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    if not job.result:
        raise HTTPException(status_code=400, detail="Job has no result yet")
    return job


def _result(job: OptimizationJob) -> dict:
    return json.loads(job.result) if isinstance(job.result, str) else (job.result or {})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stn_key(name: str) -> str:
    import re
    return re.sub(r"\s+", "_", name.strip().lower())


def _extract_stn_keys(schedule: list[dict]) -> list[str]:
    """Extract station key names from flat solver output keys (num_pumps_<stn>)."""
    seen: list[str] = []
    for h in schedule:
        for k in h.keys():
            if k.startswith("num_pumps_"):
                stn = k[len("num_pumps_"):]
                if stn and stn not in seen:
                    seen.append(stn)
    return seen


def _csv_response(content: str, filename: str) -> StreamingResponse:
    return StreamingResponse(
        io.BytesIO(content.encode("utf-8-sig")),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# Schedule CSV
# ---------------------------------------------------------------------------

@router.get("/{job_id}/export/schedule-csv")
def export_schedule_csv(
    job_id: str,
    session: Session = Depends(db_session),
    user: User | None = Depends(optional_user),
):
    job = _get_job(job_id, session, user)
    result = _result(job)
    schedule = result.get("mainline_schedule") or []
    if not schedule:
        raise HTTPException(status_code=400, detail="No schedule data in result")

    stn_keys = _extract_stn_keys(schedule)
    rows: list[dict] = []
    for h in schedule:
        row: dict = {
            "Hour": h.get("hour_label", ""),
            "Step": h.get("step", ""),
            "Flow (m3/h)": h.get("FLOW", ""),
            "Total Cost (INR)": h.get("total_cost", ""),
            "Power Cost (INR)": h.get("power_cost", ""),
            "DRA Cost (INR)": h.get("dra_cost", ""),
            "Num Pumps": h.get("num_pumps", ""),
            "Error": "YES" if h.get("error") else "NO",
        }
        for k in stn_keys:
            row[f"{k}_pumps"]    = h.get(f"num_pumps_{k}", "")
            row[f"{k}_eff_pct"]  = h.get(f"efficiency_{k}", "")
            row[f"{k}_dra_ppm"]  = h.get(f"dra_ppm_{k}", "")
            row[f"{k}_sdh_m"]    = h.get(f"sdh_m_{k}", h.get(f"sdh_{k}", ""))
            row[f"{k}_power_kw"] = h.get(f"power_cost_{k}", "")
        rows.append(row)

    if not rows:
        raise HTTPException(status_code=400, detail="Empty schedule")

    cols = list(rows[0].keys())
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(str(r.get(c, "")) for c in cols))
    return _csv_response("\n".join(lines), "schedule.csv")


# ---------------------------------------------------------------------------
# Linefill snapshots CSV
# ---------------------------------------------------------------------------

@router.get("/{job_id}/export/linefill-csv")
def export_linefill_csv(
    job_id: str,
    session: Session = Depends(db_session),
    user: User | None = Depends(optional_user),
):
    job = _get_job(job_id, session, user)
    result = _result(job)
    snaps = result.get("linefill_snaps") or []
    if not snaps:
        raise HTTPException(status_code=400, detail="No linefill snapshot data")

    rows: list[dict] = []
    for hi, snap in enumerate(snaps):
        for bi, batch in enumerate(snap):
            rows.append({
                "Hour": hi,
                "Batch": bi + 1,
                "Product": batch.get("Product", ""),
                "Volume (m3)": batch.get("Volume (m³)", batch.get("Volume (m3)", "")),
                "Viscosity (cSt)": batch.get("Viscosity (cSt)", ""),
                "Density (kg/m3)": batch.get("Density (kg/m³)", batch.get("Density (kg/m3)", "")),
                "Initial DRA (ppm)": batch.get("Initial DRA (ppm)", ""),
            })

    if not rows:
        raise HTTPException(status_code=400, detail="Empty linefill snapshots")
    cols = list(rows[0].keys())
    lines = [",".join(cols)] + [",".join(str(r.get(c, "")) for c in cols) for r in rows]
    return _csv_response("\n".join(lines), "linefill_snapshots.csv")


# ---------------------------------------------------------------------------
# Next-day linefill CSV
# ---------------------------------------------------------------------------

@router.get("/{job_id}/export/nextday-linefill-csv")
def export_nextday_linefill_csv(
    job_id: str,
    session: Session = Depends(db_session),
    user: User | None = Depends(optional_user),
):
    job = _get_job(job_id, session, user)
    result = _result(job)
    final_vol = result.get("final_vol") or []
    if not final_vol:
        raise HTTPException(status_code=400, detail="No next-day linefill data")

    cols = ["Product", "Volume (m3)", "Viscosity (cSt)", "Density (kg/m3)", "Initial DRA (ppm)"]
    lines = [",".join(cols)]
    for b in final_vol:
        lines.append(",".join([
            str(b.get("Product", "")),
            str(b.get("Volume (m³)", b.get("Volume (m3)", ""))),
            str(b.get("Viscosity (cSt)", "")),
            str(b.get("Density (kg/m³)", b.get("Density (kg/m3)", ""))),
            str(b.get("Initial DRA (ppm)", "")),
        ]))
    return _csv_response("\n".join(lines), "nextday_linefill.csv")


# ---------------------------------------------------------------------------
# Segment batches CSV
# ---------------------------------------------------------------------------

@router.get("/{job_id}/export/segments-csv")
def export_segments_csv(
    job_id: str,
    session: Session = Depends(db_session),
    user: User | None = Depends(optional_user),
):
    job = _get_job(job_id, session, user)
    result = _result(job)
    segs = result.get("final_segment_slices") or []
    if not segs:
        raise HTTPException(status_code=400, detail="No segment slice data")

    rows: list[dict] = []
    for seg in segs:
        for bi, sl in enumerate(seg.get("slices", [])):
            rows.append({
                "Segment Idx": seg.get("segment_idx", ""),
                "Station": seg.get("station_name", ""),
                "Batch #": bi + 1,
                "Length (km)": sl.get("length_km", ""),
                "Viscosity (cSt)": sl.get("kv", ""),
                "Density (kg/m3)": sl.get("rho", ""),
            })

    if not rows:
        raise HTTPException(status_code=400, detail="Empty segment slices")
    cols = list(rows[0].keys())
    lines = [",".join(cols)] + [",".join(str(r.get(c, "")) for c in cols) for r in rows]
    return _csv_response("\n".join(lines), "segment_batches.csv")


# ---------------------------------------------------------------------------
# Generic result CSV (single-point flow_rate mode)
# ---------------------------------------------------------------------------

@router.get("/{job_id}/export/result-csv")
def export_result_csv(
    job_id: str,
    session: Session = Depends(db_session),
    user: User | None = Depends(optional_user),
):
    job = _get_job(job_id, session, user)
    result = _result(job)

    flat: dict = {}
    for k, v in result.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            flat[k] = v
    rows = [flat]
    cols = list(flat.keys())
    lines = [",".join(cols)] + [",".join(str(r.get(c, "")) for c in cols) for r in rows]
    return _csv_response("\n".join(lines), "optimization_result.csv")


# ---------------------------------------------------------------------------
# Excel multi-sheet
# ---------------------------------------------------------------------------

@router.get("/{job_id}/export/excel")
def export_excel(
    job_id: str,
    session: Session = Depends(db_session),
    user: User | None = Depends(optional_user),
):
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment
    except ImportError:
        raise HTTPException(status_code=501, detail="openpyxl not installed")

    job = _get_job(job_id, session, user)
    result = _result(job)

    wb = openpyxl.Workbook()
    _hdr = Font(bold=True)
    _fill_blue = PatternFill("solid", fgColor="DBEAFE")
    _fill_gray = PatternFill("solid", fgColor="F3F4F6")

    def _write_sheet(ws, headers: list[str], rows: list[list]):
        for ci, h in enumerate(headers, 1):
            cell = ws.cell(row=1, column=ci, value=h)
            cell.font = _hdr
            cell.fill = _fill_blue
        for ri, row in enumerate(rows, 2):
            fill = _fill_gray if ri % 2 == 0 else None
            for ci, v in enumerate(row, 1):
                c = ws.cell(row=ri, column=ci, value=v)
                if fill:
                    c.fill = fill

    # --- Summary sheet ---
    ws = wb.active
    ws.title = "Summary"
    op_mode = result.get("op_mode", "flow_rate")
    summary_rows = [
        ["op_mode", op_mode],
        ["total_cost", result.get("total_cost", "")],
        ["total_throughput", result.get("total_throughput", "")],
        ["flow", result.get("flow", "")],
        ["generated_at", datetime.utcnow().isoformat()],
    ]
    _write_sheet(ws, ["Key", "Value"], summary_rows)

    # --- Schedule sheet ---
    schedule = result.get("mainline_schedule") or []
    if schedule:
        ws2 = wb.create_sheet("Schedule")
        all_stn_keys = _extract_stn_keys(schedule)
        base_hdrs = ["Hour", "Flow (m3/h)", "Total Cost", "Power Cost", "DRA Cost", "Pumps", "Error"]
        stn_hdrs: list[str] = []
        for k in all_stn_keys:
            stn_hdrs += [f"{k}_pumps", f"{k}_eff%", f"{k}_ppm", f"{k}_sdh"]
        hdrs = base_hdrs + stn_hdrs
        rows2: list[list] = []
        for h in schedule:
            row: list = [
                h.get("hour_label", ""),
                h.get("FLOW", ""),
                h.get("total_cost", ""),
                h.get("power_cost", ""),
                h.get("dra_cost", ""),
                h.get("num_pumps", ""),
                "YES" if h.get("error") else "NO",
            ]
            for k in all_stn_keys:
                row += [
                    h.get(f"num_pumps_{k}", ""),
                    h.get(f"efficiency_{k}", ""),
                    h.get(f"dra_ppm_{k}", ""),
                    h.get(f"sdh_m_{k}", h.get(f"sdh_{k}", "")),
                ]
            rows2.append(row)
        _write_sheet(ws2, hdrs, rows2)

    # --- Linefill sheet ---
    final_vol = result.get("final_vol") or []
    if final_vol:
        ws3 = wb.create_sheet("NextDayLinefill")
        _write_sheet(ws3,
            ["Product", "Volume (m3)", "Viscosity (cSt)", "Density (kg/m3)", "Initial DRA (ppm)"],
            [[b.get("Product",""), b.get("Volume (m³)",b.get("Volume (m3)","")),
              b.get("Viscosity (cSt)",""), b.get("Density (kg/m³)",b.get("Density (kg/m3)","")),
              b.get("Initial DRA (ppm)","")] for b in final_vol],
        )

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="pipeline_report.xlsx"'},
    )


# ---------------------------------------------------------------------------
# PDF report (fpdf2)
# ---------------------------------------------------------------------------

@router.get("/{job_id}/export/pdf")
def export_pdf(
    job_id: str,
    session: Session = Depends(db_session),
    user: User | None = Depends(optional_user),
):
    try:
        from fpdf import FPDF
    except ImportError:
        raise HTTPException(status_code=501, detail="fpdf2 not installed")

    job = _get_job(job_id, session, user)
    result = _result(job)

    pdf = FPDF()
    pdf.set_margins(15, 15, 15)

    try:
        pdf.add_page()

        # Cover
        pdf.set_font("Helvetica", "B", 20)
        pdf.cell(0, 12, "Pipeline Optimization Report", align="C")
        pdf.ln()
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", align="C")
        pdf.ln()
        pdf.cell(0, 7, f"Job ID: {job_id}", align="C")
        pdf.ln()
        pdf.ln(6)

        # Summary table
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Executive Summary")
        pdf.ln()
        pdf.set_font("Helvetica", "", 9)
        op_mode = result.get("op_mode", "flow_rate")
        rows_kv = [
            ("Mode", op_mode),
            ("Total Cost (INR)", f"{float(result.get('total_cost') or 0):,.0f}"),
            ("Total Throughput (m³)", f"{float(result.get('total_throughput') or 0):,.0f}"),
            ("Flow (m³/h)", f"{float(result.get('flow') or result.get('hourly_flow') or 0):,.0f}"),
        ]
        for k, v in rows_kv:
            pdf.cell(60, 7, k, border=1)
            pdf.cell(80, 7, str(v), border=1)
            pdf.ln()
        pdf.ln(4)

        # Schedule table (if available)
        schedule = result.get("mainline_schedule") or []
        if schedule:
            pdf.add_page("L")
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 8, "24-Hour Schedule")
            pdf.ln()
            pdf.set_font("Helvetica", "B", 7)
            hdrs = ["Hour", "Flow", "Total INR", "Power INR", "DRA INR", "Pumps", "OK"]
            col_w = [22, 22, 28, 28, 22, 18, 12]
            for h, w in zip(hdrs, col_w):
                pdf.cell(w, 6, h, border=1, align="C")
            pdf.ln()
            pdf.set_font("Helvetica", "", 7)
            for row in schedule:
                fill = bool(row.get("error", False))
                if fill:
                    pdf.set_fill_color(255, 220, 220)
                vals = [
                    row.get("hour_label", ""),
                    f"{float(row.get('FLOW') or 0):.0f}",
                    f"{float(row.get('total_cost') or 0):,.0f}",
                    f"{float(row.get('power_cost') or 0):,.0f}",
                    f"{float(row.get('dra_cost') or 0):,.0f}",
                    str(row.get("num_pumps", "")),
                    "N" if row.get("error") else "Y",
                ]
                for v, w in zip(vals, col_w):
                    pdf.cell(w, 5, v, border=1, fill=fill, align="C")
                pdf.ln()
                if fill:
                    pdf.set_fill_color(255, 255, 255)

        # Backtrack notes
        bt = result.get("backtrack_notes") or []
        if bt:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 8, "Backtrack / Audit Log")
            pdf.ln()
            pdf.set_font("Helvetica", "", 9)
            for note in bt:
                pdf.multi_cell(0, 6, f"- {note}")

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {exc}") from exc

    buf = io.BytesIO(pdf.output())
    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="pipeline_report.pdf"'},
    )

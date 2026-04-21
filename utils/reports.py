import os
import datetime

REPORTS_DIR = os.path.join('static', 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

def generate_pdf_report(pred: dict, rec: dict, top5: list) -> str:
    """
    Generates a professional PDF clinical report.
    Falls back to a plain-text .txt report if reportlab is not installed.
    """
    out_path_pdf = os.path.join(REPORTS_DIR, f"report_{pred.get('id','0')[:8]}.pdf")
    out_path_txt = out_path_pdf.replace('.pdf', '.txt')

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, HRFlowable)
        from reportlab.lib.enums import TA_CENTER, TA_LEFT

        doc = SimpleDocTemplate(out_path_pdf, pagesize=A4,
                                leftMargin=2*cm, rightMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        story  = []

        # Header
        header_style = ParagraphStyle('header', parent=styles['Heading1'],
                                      fontSize=16, textColor=colors.HexColor('#1a237e'),
                                      alignment=TA_CENTER, spaceAfter=4)
        sub_style = ParagraphStyle('sub', parent=styles['Normal'],
                                   fontSize=10, textColor=colors.grey,
                                   alignment=TA_CENTER, spaceAfter=2)
        normal = styles['Normal']
        bold_style = ParagraphStyle('bold', parent=normal, fontName='Helvetica-Bold')

        story.append(Paragraph("SKINDX — SKIN DISEASE CLASSIFICATION REPORT", header_style))
        story.append(Paragraph("Ghana Communication Technology University · Faculty of Computing & Information Systems", sub_style))
        story.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%d %B %Y, %H:%M')}", sub_style))
        story.append(HRFlowable(width='100%', thickness=2, color=colors.HexColor('#1a237e')))
        story.append(Spacer(1, 0.4*cm))

        # Patient Info
        story.append(Paragraph("PATIENT INFORMATION", bold_style))
        story.append(Spacer(1, 0.2*cm))
        pt_data = [
            ['Patient Name', pred.get('patient_name', 'N/A'),
             'Age', pred.get('patient_age', 'N/A')],
            ['Sex', pred.get('patient_sex', 'N/A'),
             'Date', pred.get('created_at', 'N/A')[:10]],
            ['Examined by', pred.get('full_name', 'N/A'), '', ''],
        ]
        pt_table = Table(pt_data, colWidths=[3.5*cm, 6*cm, 2.5*cm, 5*cm])
        pt_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f5f5f5')),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(pt_table)
        story.append(Spacer(1, 0.4*cm))

        # Prediction result
        urgency_colors = {
            'critical': '#c62828', 'high': '#e53935',
            'medium': '#f57c00',   'low': '#388e3c', 'none': '#1565c0'
        }
        urgency = rec.get('urgency', 'low')
        urg_color = colors.HexColor(urgency_colors.get(urgency, '#388e3c'))

        story.append(Paragraph("CLASSIFICATION RESULT", bold_style))
        story.append(Spacer(1, 0.2*cm))
        res_data = [
            ['Predicted Condition', pred.get('predicted_class', 'N/A')],
            ['Confidence Score', f"{float(pred.get('confidence', 0))*100:.2f}%"],
            ['Severity Level', rec.get('severity', 'N/A')],
            ['Urgency', urgency.upper()],
            ['Recommended Specialist', rec.get('specialist', 'N/A')],
        ]
        res_table = Table(res_data, colWidths=[5*cm, 12*cm])
        res_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8eaf6')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('TEXTCOLOR', (1, 3), (1, 3), urg_color),
            ('FONTNAME', (1, 3), (1, 3), 'Helvetica-Bold'),
        ]))
        story.append(res_table)
        story.append(Spacer(1, 0.4*cm))

        # Clinical description
        story.append(Paragraph("CLINICAL DESCRIPTION", bold_style))
        story.append(Spacer(1, 0.15*cm))
        story.append(Paragraph(rec.get('description', 'N/A'), normal))
        story.append(Spacer(1, 0.4*cm))

        # Recommendations
        story.append(Paragraph("CLINICAL RECOMMENDATIONS", bold_style))
        story.append(Spacer(1, 0.15*cm))
        for i, r in enumerate(rec.get('recommendations', []), 1):
            story.append(Paragraph(f"{i}. {r}", normal))
        story.append(Spacer(1, 0.4*cm))

        # Top 5 predictions
        story.append(Paragraph("TOP-5 DIFFERENTIAL PREDICTIONS", bold_style))
        story.append(Spacer(1, 0.2*cm))
        top5_data = [['Rank', 'Condition', 'Confidence (%)']]
        for i, item in enumerate(top5, 1):
            top5_data.append([str(i), item['class'], f"{item['probability']:.2f}%"])
        t5_table = Table(top5_data, colWidths=[2*cm, 12*cm, 3*cm])
        t5_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a237e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('PADDING', (0, 0), (-1, -1), 7),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('ALIGN', (2, 0), (2, -1), 'CENTER'),
        ]))
        story.append(t5_table)
        story.append(Spacer(1, 0.4*cm))

        # Notes
        if pred.get('notes'):
            story.append(Paragraph("CLINICIAN NOTES", bold_style))
            story.append(Spacer(1, 0.15*cm))
            story.append(Paragraph(pred['notes'], normal))
            story.append(Spacer(1, 0.4*cm))

        # Disclaimer
        story.append(HRFlowable(width='100%', thickness=1, color=colors.lightgrey))
        story.append(Spacer(1, 0.2*cm))
        disc = ParagraphStyle('disc', parent=normal, fontSize=8,
                              textColor=colors.grey, alignment=TA_CENTER)
        story.append(Paragraph(
            "DISCLAIMER: This report is generated by an AI-assisted tool for screening purposes only. "
            "It does not constitute a definitive medical diagnosis. All findings must be reviewed and "
            "confirmed by a qualified healthcare professional before clinical decisions are made.",
            disc))

        doc.build(story)
        return out_path_pdf

    except ImportError:
        # Plain text fallback
        with open(out_path_txt, 'w') as f:
            f.write("SKINDX — SKIN DISEASE CLASSIFICATION REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Patient : {pred.get('patient_name','N/A')}\n")
            f.write(f"Age     : {pred.get('patient_age','N/A')}\n")
            f.write(f"Sex     : {pred.get('patient_sex','N/A')}\n")
            f.write(f"Date    : {pred.get('created_at','N/A')[:10]}\n\n")
            f.write(f"Predicted Condition : {pred.get('predicted_class','N/A')}\n")
            f.write(f"Confidence          : {float(pred.get('confidence',0))*100:.2f}%\n")
            f.write(f"Urgency             : {rec.get('urgency','N/A').upper()}\n\n")
            f.write("RECOMMENDATIONS:\n")
            for r in rec.get('recommendations', []):
                f.write(f"  • {r}\n")
            f.write("\nDISCLAIMER: AI screening tool only. Confirm with qualified clinician.\n")
        return out_path_txt

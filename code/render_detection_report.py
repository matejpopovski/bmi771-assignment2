"""Render the detection section as a PDF for Matej's combined report.

The markdown source, metrics and figures remain the editable source artifacts.
"""
from pathlib import Path
import re
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether

ROOT = Path(__file__).resolve().parent.parent


def inline(text):
    text = escape(text)
    text = re.sub(r'`([^`]+)`', r'<font name="Courier" size="8">\1</font>', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    # Long hashes must wrap instead of running beyond a page margin.
    text = re.sub(r'\b[0-9a-f]{64}\b', lambda m: '<br/>'.join(m[0][i:i+32] for i in (0, 32)), text)
    return text


def main():
    source = ROOT / 'writeup/sean_detection.md'
    output = ROOT / 'writeup/sean_detection.pdf'
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyReport', fontName='Helvetica', fontSize=9.5,
                              leading=13.3, spaceAfter=8))
    styles.add(ParagraphStyle(name='TitleReport', fontName='Helvetica-Bold', fontSize=21,
                              leading=25, textColor=colors.HexColor('#163d53'), spaceAfter=13))
    styles.add(ParagraphStyle(name='SectionReport', fontName='Helvetica-Bold', fontSize=13,
                              leading=17, spaceBefore=13, spaceAfter=7, keepWithNext=True,
                              textColor=colors.HexColor('#163d53')))
    styles.add(ParagraphStyle(name='CaptionReport', fontSize=8, leading=10.5, spaceAfter=12,
                              textColor=colors.HexColor('#425563')))
    story, paragraph, table_rows = [], [], []
    width = letter[0] - 104

    def flush_paragraph():
        if paragraph:
            story.append(Paragraph(inline(' '.join(paragraph)), styles['BodyReport']))
            paragraph.clear()

    def flush_table():
        if not table_rows:
            return
        rows = [[Paragraph(inline(cell), styles['BodyReport']) for cell in row] for row in table_rows]
        widths = [width*.49, width*.17, width*.19, width*.15]
        table = Table(rows, colWidths=widths, repeatRows=1, hAlign='LEFT')
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8f0f5')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 7), ('RIGHTPADDING', (0, 0), (-1, -1), 7),
            ('TOPPADDING', (0, 0), (-1, -1), 7), ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LINEBELOW', (0, 0), (-1, 0), .7, colors.HexColor('#819aaa')),
            ('LINEBELOW', (0, 1), (-1, -1), .3, colors.HexColor('#d6dfe5')),
        ]))
        story.extend([table, Spacer(1, 10)])
        table_rows.clear()

    for line in source.read_text().splitlines():
        if not line.strip():
            flush_paragraph()
            flush_table()
        elif line.startswith('# '):
            flush_paragraph()
            story.append(Paragraph(inline(line[2:]), styles['TitleReport']))
        elif line.startswith('## '):
            flush_paragraph()
            flush_table()
            story.append(Paragraph(inline(line[3:]), styles['SectionReport']))
        elif line.startswith('|'):
            flush_paragraph()
            cells = [cell.strip() for cell in line.strip('|').split('|')]
            if not all(re.fullmatch(r'[-: ]+', cell) for cell in cells):
                table_rows.append(cells)
        elif line.startswith('!['):
            flush_paragraph()
            match = re.fullmatch(r'!\[([^]]+)\]\(([^)]+)\)', line)
            image_path = (source.parent / match[2]).resolve()
            with PILImage.open(image_path) as image:
                iw, ih = image.size
            figure = Image(str(image_path), width=width, height=width*ih/iw)
            caption = Paragraph(inline(match[1].replace('_', ' ')), styles['CaptionReport'])
            story.append(KeepTogether([figure, Spacer(1, 4), caption]))
        else:
            paragraph.append(line.strip())
    flush_paragraph()
    flush_table()

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#526572'))
        canvas.drawString(52, 29, 'Sean Chang | TransformerBlock and FCOS | For team integration')
        canvas.drawRightString(letter[0]-52, 29, str(document.page))
        canvas.restoreState()

    document = SimpleDocTemplate(str(output), pagesize=letter, rightMargin=52,
                                 leftMargin=52, topMargin=45, bottomMargin=49,
                                 title='BMI / CS 771 Assignment 2: TransformerBlock and FCOS',
                                 author='Sean Chang; prepared with AI assistance')
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(output)


if __name__ == '__main__':
    main()

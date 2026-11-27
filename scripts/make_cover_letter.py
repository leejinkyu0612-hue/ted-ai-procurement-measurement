# -*- coding: utf-8 -*-
"""S19 cover letter (docx). Usage: python make_cover_letter.py "<title>" <out.docx>"""
import sys
from docx import Document
from docx.shared import Pt
title, out = sys.argv[1], sys.argv[2]
d = Document(); st = d.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(11)
for line in [
    "To the Program Committee, AI-Biz2027 (Artificial Intelligence of and for Business), JSAI-isAI 2027",
    "",
    "Dear Committee Members,",
    "",
    f"We submit our paper, \"{title}\", for consideration at AI-Biz2027.",
    "The paper studies how AI-related purchases are identified in multilingual public procurement notices and how this measurement choice affects estimates of AI demand and of competition (single bidding). Using 54,160 IT and software award notices from Tenders Electronic Daily (2024-2025), a pre-specified stratified reference sample, and design-based supervised learning, it reports that a multilingual keyword dictionary roughly doubles the recall of the common English keyword approach and that a naive regression association with single bidding is not confirmed after accounting for measurement error. The work fits the workshop topics 'Data Mining, Text Mining and Data Analysis' and 'Empirical Analysis on Managerial Decision Making'.",
    "The manuscript follows the Springer LNCS/LNAI format and has 14 pages including figures and references (limit: 16 pages). It is original, has not been published, and is not under consideration elsewhere. Code, prompts, the analysis plan, annotation logs, and derived notice-level data are publicly available on GitHub.",
    "Large language models were used as research instruments for reference annotation and, with the authors' review, for language editing and internal review; this is disclosed in the paper. Jinkyu Lee is the founder and CEO of HappyAI; the authors declare no other competing interests.",
    "",
    "Sincerely,",
    "Jinkyu Lee and Chang Soo Sung (corresponding author)",
    "Department of Technology Entrepreneurship, Graduate School, Dongguk University, Seoul, Republic of Korea",
    "leejinkyu0612@naver.com; redsun44@dgu.ac.kr",
]:
    d.add_paragraph(line)
d.save(out); print("ok")

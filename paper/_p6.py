import sys
s=open('R3.tex',encoding='utf-8').read()
def rep(a,b):
    global s
    c=s.count(a)
    if c!=1: sys.exit(f'count {c}: {a[:80]}')
    s=s.replace(a,b)
rep("The analysis plan was time-stamped in the project repository before reference annotation (commit 7314e8d, 27 November 2026, 11:12 KST; amendment commit b1eb797, 12:22 KST); it was not externally preregistered.",
    "The analysis plan was time-stamped in the project repository before reference annotation began (commit 7314e8d, 27 November 2026, 11:12 KST); one amendment (commit b1eb797, 12:22 KST) was committed after annotation had ended (12:16 KST) but before any M3 result was computed. Commit times come from the authors' git client, and the plan was not externally preregistered.")
rep(r"\keywords{Public procurement \and AI-relatedness \and Multilingual text classification \and Design-based supervised learning \and Measurement error \and LLM annotation.}",
    r"\keywords{Public procurement \and Business intelligence \and Text mining \and Multilingual text classification \and Design-based supervised learning \and Measurement error.}")
rep("In addition, the authors used ChatGPT (OpenAI) and Claude (Anthropic) to assist with literature search, drafting, language editing, code generation, and structured internal review.",
    "In addition, between November 2026 and January 2027 the authors used ChatGPT (OpenAI, GPT-6.2 Thinking) and Claude (Anthropic, Opus 5.5) to assist with literature search, code generation, structured internal review, and the rewording and copy-editing of author-written drafts; all text was checked and revised by the authors, and no figure was produced with generative AI.")
rep("Future work should obtain expert labels", "For AI suppliers, the design yields a reproducible estimate of the public AI market; for buyers and regulators, it supports competition monitoring with quantified error. Future work should obtain expert labels")
open('R3_S10.tex','w',encoding='utf-8').write(s); print('ok')

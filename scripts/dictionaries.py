# -*- coding: utf-8 -*-
"""Keyword dictionaries (pre-specified, 2026-11-27).
EN list follows the AI-term style of job-posting studies (e.g., Acemoglu et al. 2022) adapted to procurement.
Multilingual list adds core equivalents of 'artificial intelligence', 'machine learning',
'neural network', 'language model' and 'chatbot' for EU languages (stems allow inflection).
"""
import re

EN = [
    r"artificial[\s-]+intelligence", r"machine[\s-]+learning", r"deep[\s-]+learning", r"neural[\s-]+networks?",
    r"natural[\s-]+language[\s-]+processing", r"\bNLP\b", r"computer[\s-]+vision", r"image[\s-]+recognition",
    r"speech[\s-]+recognition", r"speech[\s-]+to[\s-]+text", r"text[\s-]+to[\s-]+speech", r"chat[\s-]?bots?",
    r"virtual[\s-]+assistants?", r"large[\s-]+language[\s-]+models?", r"\bLLMs?\b", r"generative[\s-]+AI", r"\bGenAI\b",
    r"\bGPT\b", r"\bChatGPT\b", r"\bAI\b", r"\bA\.I\.", r"AI[\s-]+(based|powered|driven|enabled)",
]

ML_EXTRA = [
    # German
    r"k(ü|ue)nstlich\w*\s+intelligenz", r"maschinell\w*\s+lern\w*", r"neuronal\w*\s+netz\w*", r"sprachmodell\w*", r"\bKI\b", r"KI-\w+",
    # French
    r"intelligence\s+artificielle", r"apprentissage\s+(automatique|profond)", r"r(é|e)seaux?\s+(de\s+)?neuron\w*", r"\bIA\b",
    # Spanish / Portuguese / Italian / Romanian
    r"inteligencia\s+artificial", r"aprendizaje\s+(autom(á|a)tico|profundo)", r"redes?\s+neuronal\w*",
    r"intelig(ê|e)ncia\s+artificial", r"aprendizagem\s+autom(á|a)tica", r"aprendizado\s+de\s+m(á|a)quina", r"redes?\s+neura\w*",
    r"intelligenza\s+artificiale", r"apprendimento\s+automatico", r"reti?\s+neural\w*",
    r"inteligen(ț|ţ|t)\w*\s+artificial\w*", r"(î|i)nv(ă|a)(ț|ţ|t)are\s+automat\w*",
    # Dutch / Swedish / Danish / Finnish / Estonian
    r"kunstmatige\s+intelligentie", r"neurale?\s+netwerk\w*", r"artificiell\w*\s+intelligens", r"maskininl(ä|a)rning",
    r"kunstig\w*\s+intelligens", r"maskinl(æ|ae)ring", r"teko(ä|a)ly\w*", r"koneoppi\w*", r"tehisintellekt\w*", r"masin(õ|o)pe\w*",
    # Polish / Czech / Slovak / Slovenian / Croatian
    r"sztuczn\w*\s+inteligencj\w*", r"uczeni\w*\s+maszynow\w*", r"sieci\w*\s+neuronow\w*",
    r"um(ě|e)l\w*\s+inteligenc\w*", r"strojov\w*\s+u(č|c)en\w*", r"neuronov\w*\s+s(í|i)t\w*",
    r"umetn\w*\s+inteligenc\w*", r"umjetn\w*\s+inteligencij\w*", r"strojn\w*\s+u(č|c)enj\w*",
    # Baltic / Hungarian / Bulgarian / Greek / Maltese / Irish
    r"m(ā|a)ksl(ī|i)g\w*\s+intelekt\w*", r"dirbtin\w*\s+intelekt\w*", r"ma(š|s)inin\w*\s+mokym\w*",
    r"mesters(é|e)ges\s+intelligenci\w*", r"g(é|e)pi\s+tanul(á|a)s\w*",
    r"изкуствен\w*\s+интелект\w*", r"машинно\s+обучение",
    r"τεχνητ\w*\s+νοημοσ(ύ|υ)ν\w*", r"μηχανικ\w*\s+μ(ά|α)θησ\w*",
    r"intelli(ġ|g)enza\s+artifi(ċ|c)jali", r"intleacht\s+shaorga",
    # chatbot variants
    r"czat\s?bot\w*", r"chatbot\w*", r"\bbot\w*\s+konwersac\w*",
]

_CASE_SENSITIVE = {r"\bKI\b", r"KI-\w+", r"\bIA\b", r"\bAI\b", r"\bNLP\b", r"\bLLMs?\b", r"\bGPT\b", r"\bGenAI\b", r"\bA\.I\."}


def _compile(lst):
    out = []
    for p in lst:
        out.append(re.compile(p) if p in _CASE_SENSITIVE else re.compile(p, re.IGNORECASE))
    return out


EN_RE = _compile(EN)
ML_RE = _compile(EN + ML_EXTRA)


def hit(text, regs):
    return int(any(r.search(text or "") for r in regs))

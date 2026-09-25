#!/usr/bin/env python3
"""
arXiv Digest Web App - Alejandro's Personalized Paper Digest
TF-IDF + optional Moonshot AI LLM-enhanced scoring
"""

import os
import re
import json
import time
import requests
import xml.etree.ElementTree as ET
import numpy as np
from datetime import datetime, timedelta, timezone
from functools import lru_cache

from flask import Flask, render_template, jsonify, request, session
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from sklearn.feature_extraction.text import TfidfVectorizer

# ============================================================
# CONFIG
# ============================================================
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))

# CORS: allow all origins for the API endpoints
# In production, restrict this to your actual domain
allowed_origins = os.environ.get('CORS_ORIGINS', '*')
CORS(app, resources={r"/api/*": {"origins": allowed_origins}})

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# ============================================================
# INSIGHT CATEGORIES — source of truth for topical keywords
# ============================================================
# The digest's DEFAULT_KEYWORDS is automatically computed as the
# union of all category keywords + cross-cutting terms.
# When you add a keyword to a category, it automatically becomes
# part of the digest vocabulary.

DEFAULT_CROSS_CUTTING = [
    "quantum hardware", "quantum algorithm", "error correction", "error mitigation",
    "transpilation", "quantum simulation", "variational quantum algorithms", "VQE",
    "VQA", "hybrid algorithms", "transmon", "neutral atom", "ion trap",
    "ion-trap", "NV center", "nitrogen-vacancy", "quantum noise",
    "hardware noise", "surface", "interface", "interfacial",
    "solvent", "solvated", "quantum phase estimation", "QPE",
    "Trotter", "hardware-efficient", "many-body", "NISQ", "benchmarking",
    "barren plateaus", "solvation", "implicit solvation",
    "tensor networks", "MPS", "MPO", "matrix product state",
    "Bose-Hubbard", "kinetics", "transport",
    "Nonlinear", "nonlinear dynamics", "Krylov",
    "Quantinuum", "IQM", "dissociation", "dissociative ionization",
    "quantum nuclear effects", "polaron", "LiF",
    "fault-tolerant quantum computing", "FTQC",
]

DEFAULT_INSIGHT_CATEGORIES = [
    {
        "name": "Quantum Chemistry",
        "keywords": [
            "quantum chemistry", "ab-initio", "first-principles",
            "variational quantum eigensolver", "DMET",
            "density matrix embedding theory", "DMFT",
            "dynamical mean-field theory", "electronic structure",
            "QSE", "VQSE", "quantum subspace expansion",
            "quantum equations of motion", "QEOM", "embedding",
            "embedding theory", "quantum embedding", "subspace expansion",
            "quantum-selected", "configuration interaction", "SCI", "QSCI",
            "SQD", "Subspace Quantum Diagonalization",
            "fermionic", "fermion", "Hartree-Fock", "ROHF",
            "DFT", "density functional theory",
            "transition metal", "transition metal oxides",
            "transition metal complexes", "strongly correlated",
            "Pauli propagation", "ADAPT-VQE",
            "complete active-space", "self-consistent field",
            "CASSCF", "Slater determinant", "Quantum-HPC", "Ground-State",
            "molecular dynamics", "DMRG", "NEO", "nuclear",
            "Nuclear-Electronic Orbital",
            "nickel oxide", "cobalt oxide", "3d orbital",
            "Iron-Complex", "Cytochrome",
            "non-adiabatic", "pyrazine",
            "coupled cluster", "coupled-cluster", "coupled cluster theory",
            "excited states", "force field", "Lennard-Jones",
            "multiconfiguration", "multiconfigurational", "multiconfigurational dynamics", "multireference",
            "atomistic", "open-shell", "exchange-correlation",
            "Hubbard", "Mott", "mixed quantum-classical", "correlated many-body", "LiF",
            "LiH", "tensor hypercontraction", "P450", "molecular systems",
            "TDDFT", "Spin-Adapted", "RPA", "single-particle basis",
            "6-31G", "STO", "CISD", "CASCI",
            "cc-pVDZ", "cc-pVTZ", "cc-pVQZ",
        ],
        "penalising_keywords": [],
        "prompt": "You are an expert curator for QUANTUM CHEMISTRY. This domain covers electronic structure methods (DFT, coupled cluster, CASSCF), ab-initio calculations, molecular dynamics, embedding theories (DMET, DMFT), computational materials science, AND quantum computing algorithms applied to chemistry (VQE, SQD, circuit compression, ansatz optimization, qubit reduction, quantum-classical hybrid methods for molecules). Favor papers about electronic structure, potential energy surfaces, reaction mechanisms, molecular properties, AND quantum computing approaches to chemical problems (VQE for molecules, SQD, circuit compression for QC, quantum-classical hybrid methods). Papers about VQE, SQD, circuit compression, ansatz optimization, or qubit reduction for molecular problems are HIGHLY relevant. Open quantum systems and dissipation papers belong to the Open Quantum Systems category unless they explicitly study chemical reactions or molecular electronic structure. Quantum battery papers (coherence-based energy storage) belong to OPEN QUANTUM SYSTEMS, not electrochemistry. Pure quantum algorithm papers without chemistry application belong to Quantum PDE Solvers.",
        "summary_prompt": "Summarize today's Quantum Chemistry selection. Note the balance between electronic structure methods (DFT, coupled cluster, CASSCF) and quantum computing approaches (VQE, SQD, circuit compression). Flag days when no true electronic structure papers appear — e.g. if only quantum machine learning or hardware papers were selected.",
    },
    {
        "name": "Electrochemistry",
        "keywords": [
            "electrochemistry", "electrochemical", "electrode", "electrolyte",
            "Butler-Volmer", "battery", "batteries", "fuel cell",
            "cathode", "anode", "lithium", "lithium-ion",
            "Li-ion battery", "LIB",
            "ionic electrolyte", "ion transport", "Na-ion", "Calcium-ion",
            "intercalation", "energy storage",
            "impedance spectroscopy", "battery characterisation",
            "finite-element", "continuum models",
        ],
        "penalising_keywords": [
            "quantum battery", "quantum batteries",
        ],
        "prompt": "You are an expert curator for classical modelling in ELECTROCHEMISTRY (not quantum algorithms for quantum computing). This domain covers electrochemical systems: batteries (Li-ion, Na-ion, solid-state), fuel cells, electrodes, electrolytes, ion transport, intercalation, Butler-Volmer kinetics, and electrochemical characterization. Favor papers about batteries, fuel cells, electrode materials, electrolyte chemistry, and ion transport. Both experimental and simulations are of interest, especially those employing continuum models and multi-scale approaches. Quantum batteries (energy storage via quantum coherence / entanglement) are NOT electrochemical batteries — they belong to OPEN QUANTUM SYSTEMS. Supercapacitors, solar cells, and photocatalysis are relevant only if they have a clear electrochemical component.",
        "summary_prompt": "Summarize today's Electrochemistry selection. Note which battery chemistries (Li-ion, Na-ion, solid-state) or fuel cell types are represented. Flag days when only generic energy storage or materials papers were selected rather than true electrochemical studies.",
    },
    {
        "name": "Open Quantum Systems",
        "keywords": [
            "open quantum systems", "non-equilibrium",
            "dissipative", "dissipation", "Lindblad", "master equation",
            "polaron", "polaron formation", "polaron hopping", "electron-polaron",
            "exciton", "non-Markovian", "static disorder",
            "spectroscopy", "NMR", "NMR spectroscopy", "nanoscale NMR",
            "ultralow-field", "quantum control", "charge transfer",
            "electron transfer", "hole transfer", "charge transport",
            "electron transport", "hole transport",
            "PCET", "proton coupled electron transfer",
            "CIET", "coupled ion-electron transfer", "Newns-Anderson",
            "HEOM", "TEDOPA", "DAMPF", "MCTDH", "vibronic", "bosonic",
            "bosonic quantum computation", "qudits", "electron-phonon",
            "Phonon", "spectral density", "analog quantum simulation",
            "optimal control", "reorganization", "Marcus",
            "non-adiabatic", "vibrational", "Holstein", "Peierls",
            "interfacial", "Born Oppenheimer", "Born-Oppenheimer",
            "long-lived", "long-lived carriers",
            "cavity", "cavities", "qubit-cavity", "QED",
            "cavity QED", "excited states", "optomechanical",
            "relaxation", "reservoir",
            "quantum battery", "quantum batteries",
            "spin battery", "spin batteries",
            "Floquet", "quantum sensor", "energy transfer",
            "quantum dynamics", "electron-proton",
            "self-trapped", "LiF", "noise-assisted",
            "oscillator-qubit", "nonunitary", "Non-Unitary", "qumode",
            "long-time quantum dynamics", "Trotter-Suzuki", "Suzuki-Trotter",
            "quantum harmonic oscillator", "ultrafast", "hot-carrier",
            "non-thermal", "nonzero-temperature", "vibronic spectra",
            "finite-temperature",
            "nonequilibrium materials", "ultrafast electron spectroscopy",
            "nonlinear spectroscopy", "pump-probe", "2D spectroscopy", "ARPES",
        ],
        "prompt": "You are an expert curator for OPEN QUANTUM SYSTEMS. This domain covers non-equilibrium quantum dynamics like long-lived vibronic motion, electron-phonon (vibronic) coupling, dissipation, decoherence, Lindbladian evolution, quantum control and spectroscopy. Favor papers about non-Markovian dynamics and open-system quantum dynamics simulations. Vibronic (electron-phonon) coupling phenomena (e.g. polaron formation) are the core interest here. Quantum dynamics applications to spectroscopy are also specially relevant (e.g. low-field or ultra low-field NMR, non-linear 2D spectroscopy, vibrational spectroscopy, etc). Electrochemical batteries (Li-ion, fuel cells) belong to ELECTROCHEMISTRY, not here. Electronic structure methods (DFT, coupled cluster) belong to QUANTUM CHEMISTRY unless they study open-system effects.",
        "summary_prompt": "Summarize today's Open Quantum Systems selection. Emphasize non-equilibrium vibronic dynamics (electron-phonon coupling, polaron models). Flag days when no explicit dynamics or coupling papers appear — e.g. if only quantum control or generic quantum computing papers were selected.",
    },
    {
        "name": "Quantum PDE Solvers",
        "keywords": [
            "PDE", "non-linear PDE",
            "partial differential equations",
            "non-linear partial differential equations",
            "quantum linear solvers", "VQLS",
            "QNPU", "HHL", "Harrow-Hassidim-Lloyd",
            "Feynman-Kitaev", "Burgers equation",
            "feature map", "differentiable quantum circuits",
            "imaginary time evolution",
            "heat equation", "diffusion equation", "wave equation",
            "Poisson equation", "reaction-diffusion", "fluid dynamics",
            "parameter shift", "ordinary differential equations",
            "Navier-Stokes", "Koopman", "multiscale", "multigrid",
            "QAOA", "quantum machine learning", "combinatorial optimization",
        ],
        "prompt": "You are an expert curator for QUANTUM PDE SOLVERS. This domain covers quantum algorithms for solving partial differential equations, quantum linear systems (HHL, VQLS), quantum machine learning for scientific computing, QAOA, and quantum approaches to fluid dynamics, heat equations, and optimization. Favor papers about quantum algorithms for linear systems, differential equations, optimization, and scientific machine learning. Papers about quantum simulation of physical Hamiltonians belong to Open Quantum Systems unless they explicitly frame it as a PDE-solving algorithm. Barren plateaus, feature maps, and differentiable quantum circuits are relevant here as they relate to quantum machine learning. Pure quantum chemistry methods (VQE for molecules) belong to QUANTUM CHEMISTRY unless the focus is on the algorithmic/PDE-solving aspect.",
        "summary_prompt": "Summarize today's Quantum PDE Solvers selection. Note whether true PDE-solving algorithms (HHL, VQLS, variational quantum linear solvers) were found, or if the selection consists mainly of quantum machine learning or QAOA papers. Flag days when no explicit differential equation or linear system solving papers appear.",
    },
]


# Default expert prompts for each Insight category.
# These guide the LLM to act as a domain-specific expert curator,
# helping it make correct categorization decisions (e.g. quantum battery → OQS, not electrochemistry).
DEFAULT_CATEGORY_PROMPTS = {
    "Quantum Chemistry": """You are an expert curator for QUANTUM CHEMISTRY. This domain covers electronic structure methods (DFT, coupled cluster, CASSCF), ab-initio calculations, molecular dynamics, embedding theories (DMET, DMFT), computational materials science, AND quantum computing algorithms applied to chemistry (VQE, SQD, circuit compression, ansatz optimization, qubit reduction, quantum-classical hybrid methods for molecules).

KEY DISTINCTIONS:
- Favor papers about electronic structure, potential energy surfaces, reaction mechanisms, molecular properties, AND quantum computing approaches to chemical problems (VQE for molecules, SQD, circuit compression for QC, quantum-classical hybrid methods).
- Papers about VQE, SQD (Subspace Quantum Diagonalization), circuit compression, ansatz optimization, or qubit reduction for molecular problems are HIGHLY relevant — they bridge quantum computing and quantum chemistry.
- "Open quantum systems" and "dissipation" papers belong to the Open Quantum Systems category unless they explicitly study chemical reactions or molecular electronic structure.
- "Quantum battery" papers (coherence-based energy storage in quantum systems) belong to OPEN QUANTUM SYSTEMS, not electrochemistry.
- Condensed matter / many-body physics papers without chemical/molecular focus are less relevant here.
- Pure quantum algorithm papers without chemistry application (e.g. general quantum machine learning) belong to Quantum PDE Solvers.""",


    "Electrochemistry": """You are an expert curator for ELECTROCHEMISTRY. This domain covers electrochemical systems: batteries (Li-ion, Na-ion, solid-state), fuel cells, electrodes, electrolytes, ion transport, intercalation, Butler-Volmer kinetics, and electrochemical characterization.

KEY DISTINCTIONS:
- Favor papers about Li-ion batteries, fuel cells, electrode materials, electrolyte chemistry, and ion transport.
- "Quantum batteries" (energy storage via quantum coherence / entanglement) are NOT electrochemical batteries — they belong to OPEN QUANTUM SYSTEMS.
- Supercapacitors, solar cells, and photocatalysis are relevant only if they have a clear electrochemical component.
- Battery papers focused on quantum thermodynamics or quantum information belong to Open Quantum Systems.""",

    "Open Quantum Systems": """You are an expert curator for OPEN QUANTUM SYSTEMS. This domain covers non-equilibrium quantum dynamics like long-lived vibronic motion, electron-phonon (vibronic) coupling, dissipation, decoherence, Lindbladian evolution, quantum control, and quantum thermodynamics (e.g quantum batteries for coherence-based energy storage).

KEY DISTINCTIONS:
- Favor papers about non-Markovian dynamics, quantum master equations, quantum thermodynamics, and open-system quantum dynamics simulations.
- Vibronic (electron-phonon) coupling phenomena (e.g. polarons) are the core interest here.
- Electrochemical batteries (Li-ion, fuel cells) belong to ELECTROCHEMISTRY, not here.
- Electronic structure methods (DFT, coupled cluster) belong to QUANTUM CHEMISTRY unless they study open-system effects.
- Papers on quantum simulation of open systems are highly relevant.""",

    "Quantum PDE Solvers": """You are an expert curator for QUANTUM PDE SOLVERS. This domain covers quantum algorithms for solving partial differential equations, quantum linear systems (HHL, VQLS), quantum machine learning for scientific computing, QAOA, and quantum approaches to fluid dynamics, heat equations, and optimization.

KEY DISTINCTIONS:
- Favor papers about quantum algorithms for linear systems, differential equations, optimization, and scientific machine learning.
- Papers about quantum simulation of physical Hamiltonians belong to Open Quantum Systems unless they explicitly frame it as a PDE-solving algorithm.
- Barren plateaus, feature maps, and differentiable quantum circuits are relevant here as they relate to quantum machine learning.
- Pure quantum chemistry methods (VQE for molecules) belong to QUANTUM CHEMISTRY unless the focus is on the algorithmic/PDE-solving aspect.""",
}


def build_default_keywords(insight_categories=None, cross_cutting=None):
    """Build DEFAULT_KEYWORDS as the union of all category keywords
    plus cross-cutting terms.  This is the single source of truth:
    add a keyword to a category and it automatically enters the digest.
    """
    if insight_categories is None:
        insight_categories = DEFAULT_INSIGHT_CATEGORIES
    if cross_cutting is None:
        cross_cutting = DEFAULT_CROSS_CUTTING
    kw_set = set()
    for cat in insight_categories:
        for kw in cat.get("keywords", []):
            kw_set.add(kw)
    for kw in cross_cutting:
        kw_set.add(kw)
    return sorted(kw_set)


# Default keywords for the initial digest ranking.
# Computed automatically from category keywords + cross-cutting terms.
DEFAULT_KEYWORDS = build_default_keywords()

DEFAULT_PUBLICATIONS = [
    "Optimized auxiliary oscillators for the simulation of general open quantum systems",
    "Dissipation-assisted matrix product factorization",
    "Optimal Energy Transfer in Light-Harvesting Systems",
    "Nonlinear dynamics as a ground-state solution on quantum computers",
    "Probing ultrafast excitation energy transfer of the chlorosome with exciton-phonon variational dynamics",
    "Superradiance at the localization-delocalization crossover in tubular chlorosomes",
    "Driving force and nonequilibrium vibronic dynamics in charge separation of strongly bound electron-hole pairs",
    "Solving partial differential equations using a quantum computer",
    "Dynamics of coherence, localization and excitation transfer in disordered nanorings",
    "Multicolor quantum control for suppressing ground state coherences in two-dimensional electronic spectroscopy",
    "Polaron dynamics of Bloch-Zener oscillations in an extended Holstein model",
    "Simulating Electron Transfer on Noisy Quantum Computers: A Scalable Approach to Open Quantum Systems",
    "Non-Equilibrium Dynamics for Electrochemistry Simulations on Quantum Computers",
    "Scaling Up Dynamics Simulation of Open Quantum Systems on NISQ computers",
    "Quantum algorithms to solve partial differential equations in battery modelling",
    "Simulating Magnetic Pulses for Optimally Controlled NMR in Electrochemistry",
    "Variational quantum eigensolver for nonlinear dynamics",
    "Quantum algorithms for electronic structure and continuum models in electrochemistry",
]

# Umlaut normalization for fuzzy keyword matching
def normalize_umlauts(text):
    """Replace German umlauts and sharp-s with ASCII equivalents for matching.
    Handles both Unicode (ä, ö, ü, ß) and LaTeX-style (\"a, \"o, \"u, \"ss).
    """
    # LaTeX-style umlauts (arXiv RSS uses these)
    text = text.replace('\\"a', 'a').replace('\\"o', 'o').replace('\\"u', 'u').replace('\\"A', 'A').replace('\\"O', 'O').replace('\\"U', 'U').replace('\\"s', 'ss').replace('\\"ss', 'ss')
    # Unicode umlauts
    text = text.replace('ä', 'a').replace('ö', 'o').replace('ü', 'u').replace('ß', 'ss')
    text = text.replace('Ä', 'A').replace('Ö', 'O').replace('Ü', 'U')
    return text


def batch_fill_comments(papers, keywords, publications, api_key, llm_provider='moonshot', model=None):
    """
    Fill in LLM comments for multiple papers in ONE API call.
    Much cheaper than individual calls. Returns list of comment strings aligned with papers.
    """
    if not papers:
        return []

    provider_config = LLM_PROVIDERS.get(llm_provider, LLM_PROVIDERS['moonshot'])
    base_url = provider_config['base_url']
    model = model or provider_config['model']
    fmt = provider_config.get('format', 'openai')

    keyword_summary = ", ".join(keywords[:15])
    pub_summary = "; ".join(publications[:5])

    # Build numbered paper list
    paper_list = "\n".join([
        f"{i+1}. {p['title']} (Categories: {', '.join(p.get('categories', [])[:3])})"
        for i, p in enumerate(papers)
    ])

    prompt = f"""You are an expert academic advisor. A researcher with these interests: Keywords: {keyword_summary}; Publications: {pub_summary}

For each numbered paper below, write ONE SENTENCE explaining why it is relevant (or not relevant) to this researcher. Be specific about techniques, methods, or connections to their keywords.

Respond with ONLY a numbered list matching the paper numbers. Use this exact format:
1. [your one-sentence explanation]
2. [your one-sentence explanation]
...

Papers:
{paper_list}"""

    try:
        if fmt == 'anthropic':
            resp = requests.post(
                f"{base_url}/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2048,
                    "temperature": 0.3,
                },
                timeout=60
            )
        else:
            resp = requests.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 2048
                },
                timeout=60
            )

        if resp.status_code == 200:
            content = _extract_content(resp.json(), fmt)
            if not content:
                return None

            # Parse numbered list: "1. explanation" or "1) explanation" or "1 explanation"
            comments = [''] * len(papers)
            for i in range(len(papers)):
                num = i + 1
                # Try multiple patterns
                patterns = [
                    rf'^{num}\.\s*(.+?)(?=\n\s*{num+1}\.|\Z)',  # "1. ..."
                    rf'\n\s*{num}\.\s*(.+?)(?=\n\s*{num+1}\.|\Z)',  # after newline
                    rf'^{num}\)\s*(.+?)(?=\n\s*{num+1}\)|\Z)',  # "1) ..."
                    rf'\n\s*{num}\)\s*(.+?)(?=\n\s*{num+1}\)|\Z)',
                ]
                for pat in patterns:
                    m = re.search(pat, content, re.DOTALL | re.MULTILINE)
                    if m:
                        comment = m.group(1).strip()
                        # Clean up markdown bold/italic
                        comment = re.sub(r'\*\*?|\*\*?', '', comment)
                        comments[i] = comment
                        break

            return comments
        else:
            return None
    except Exception:
        return None


def _generate_category_summary(cat_name, curated_papers, summary_prompt, llm_api_key, llm_provider='moonshot', llm_model=None):
    """
    v146/v147: Generate a qualitative expert summary for a category's curated papers.
    Uses the per-category summary_prompt to guide what themes/absences to emphasize.
    """
    if not curated_papers or not llm_api_key:
        return ''

    provider_config = LLM_PROVIDERS.get(llm_provider, LLM_PROVIDERS['moonshot'])
    base_url = provider_config['base_url']
    model = llm_model or provider_config['model']
    fmt = provider_config.get('format', 'openai')

    # Build a compact list of selected paper titles (no numbering —
    # prevents the LLM from using incorrect indices in its summary)
    paper_list = "\n".join([
        f"- {p.get('title', 'Untitled')}"
        for p in curated_papers[:15]  # cap at 15 for token budget
    ])

    # v147: Use the user's summary prompt as the core directive, with a generic fallback
    guidance = summary_prompt if summary_prompt else f"Summarize today's {cat_name} selection. Note what themes are well-represented and what is notably absent."

    prompt = f"""You are an expert curator for the category "{cat_name}".

{guidance}

You selected these papers today:
{paper_list}

Write a concise 2-3 sentence summary. Be specific about themes, techniques, and notable absences.
When referring to specific papers, use their titles — do NOT use numbers or indices.
Keep it under 80 words. Write directly — do not use phrases like "I notice" or "It seems"."""

    try:
        if fmt == 'anthropic':
            resp = requests.post(
                f"{base_url}/messages",
                headers={
                    "x-api-key": llm_api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 256,
                    "temperature": 0.4,
                },
                timeout=30
            )
        else:
            resp = requests.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {llm_api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.4,
                    "max_tokens": 256
                },
                timeout=30
            )

        if resp.status_code == 200:
            content = _extract_content(resp.json(), fmt).strip()
            # Clean up: remove quotes if the LLM wrapped the summary
            content = content.strip('"').strip("'")
            return content if content else ''
        return ''
    except Exception:
        return ''


# Available arXiv categories for monitoring
AVAILABLE_CATEGORIES = [
    ('quant-ph', 'Quantum Physics'),
    ('physics.chem-ph', 'Chemical Physics'),
    ('cond-mat', 'Condensed Matter (all)'),
    ('cond-mat.mtrl-sci', 'Materials Science'),
    ('cond-mat.str-el', 'Strongly Correlated Electrons'),
    ('cond-mat.mes-hall', 'Mesoscale & Nanoscale'),
    ('cond-mat.stat-mech', 'Statistical Mechanics'),
    ('physics.comp-ph', 'Computational Physics'),
    ('cs.LG', 'Machine Learning'),
    ('cs.AI', 'Artificial Intelligence'),
    ('cs.SC', 'Scientific Computing'),
    ('math-ph', 'Mathematical Physics'),
]

# Default categories (the ones the user specifically mentioned)
DEFAULT_CATEGORIES = ['quant-ph', 'physics.chem-ph', 'cond-mat']

ARXIV_NS = {
    'atom': 'http://www.w3.org/2005/Atom',
    'arxiv': 'http://arxiv.org/schemas/atom',
    'opensearch': 'http://a9.com/-/spec/opensearch/1.1/'
}

# ============================================================
# USER PROFILE (mutable)
# ============================================================
_user_profile = {
    "name": "Alejandro D. Somoza",
    "scholar_url": "https://scholar.google.de/citations?user=PiQhn5wAAAAJ&hl=de",
    "orcid": "",
    "email": "",
    "keywords": list(DEFAULT_KEYWORDS),
    "publications": list(DEFAULT_PUBLICATIONS),
    "publication_abstracts": [],
    "categories": list(DEFAULT_CATEGORIES),
    "_saved": False,
}

def get_profile():
    """Return the current mutable user profile."""
    return dict(_user_profile)

def update_profile(updates):
    """Update the user profile with new values."""
    global _user_profile
    for key in ["name", "scholar_url", "orcid", "email", "keywords",
                "cross_cutting", "category_keywords",
                "publications", "publication_abstracts", "categories", "_saved"]:
        if key in updates:
            _user_profile[key] = updates[key]
    # If cross_cutting or category_keywords provided, also update the merged keywords
    if "cross_cutting" in updates or "category_keywords" in updates:
        cc = _user_profile.get("cross_cutting", [])
        ck = _user_profile.get("category_keywords", [])
        _user_profile["keywords"] = list(dict.fromkeys(cc + ck))  # merged, deduped
    _user_profile["_saved"] = True
    get_profile_vector.cache_clear()

# ============================================================
# PROFILE & VECTORIZER
# ============================================================
@lru_cache(maxsize=1)
def get_profile_vector(keywords_tuple, publications_tuple, abstracts_tuple=()):
    """Build TF-IDF profile vector from keywords, publications, and optional abstracts."""
    keywords = list(keywords_tuple)
    publications = list(publications_tuple)
    abstracts = list(abstracts_tuple)
    
    corpus = []
    # Keywords
    for kw in keywords:
        corpus.append(f"Research in {kw}")
        corpus.append(kw)
    # Publication titles (heavy weight)
    for pub in publications:
        corpus.append(pub)
        corpus.append(pub)
        corpus.append(pub)
    # Publication abstracts (if provided — much richer signal)
    for ab in abstracts:
        if ab and len(ab) > 20:
            corpus.append(ab)
            corpus.append(ab)
    # Research domain summaries
    corpus.extend([
        "Quantum computing simulation chemical physics electrochemistry open quantum systems",
        "Quantum algorithms partial differential equations PDE battery modeling",
        "Variational quantum eigensolver VQE electronic structure density functional DFT",
        "Open quantum systems non-Markovian dynamics tensor network MPS DMRG HEOM",
        "Electron transfer electrochemistry Butler-Volmer Marcus theory electrode surface",
        "Quantum optimal control NMR spectroscopy error mitigation correction",
        "Quantum embedding DMET DMFT strongly correlated transition metal oxides",
        "Vibronic coupling phonon polaron Holstein Peierls Born-Oppenheimer",
    ])

    vectorizer = TfidfVectorizer(
        lowercase=True, stop_words='english',
        ngram_range=(1, 3), max_features=5000, min_df=1, dtype=np.float32
    )
    profile_matrix = vectorizer.fit_transform(corpus)
    profile_vec = np.asarray(profile_matrix.sum(axis=0)).flatten()
    norm = np.linalg.norm(profile_vec)
    if norm > 0:
        profile_vec = profile_vec / norm
    return vectorizer, profile_vec

# ============================================================
# ARXIV FETCHING
# ============================================================
def _parse_entry(entry):
    """Parse a single arXiv Atom entry into a paper dict."""
    try:
        id_el = entry.find('atom:id', ARXIV_NS)
        if id_el is None:
            return None
        aid = id_el.text.strip().split('/abs/')[-1] if '/abs/' in id_el.text else id_el.text.strip().split('/')[-1]
        clean_id = aid.split('v')[0] if 'v' in aid else aid

        title_el = entry.find('atom:title', ARXIV_NS)
        title = ' '.join(title_el.text.strip().split()) if title_el is not None and title_el.text else ""

        summary_el = entry.find('atom:summary', ARXIV_NS)
        abstract = ' '.join(summary_el.text.strip().split()) if summary_el is not None and summary_el.text else ""

        published_el = entry.find('atom:published', ARXIV_NS)
        pub_date = published_el.text[:10] if published_el is not None and published_el.text else ""

        authors = []
        for a in entry.findall('atom:author', ARXIV_NS):
            name_el = a.find('atom:name', ARXIV_NS)
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())

        cats = []
        pc = entry.find('arxiv:primary_category', ARXIV_NS)
        if pc is not None:
            cats.append(pc.get('term', ''))
        for c in entry.findall('atom:category', ARXIV_NS):
            t = c.get('term', '')
            if t and t not in cats:
                cats.append(t)

        return {
            'arxiv_id': clean_id,
            'title': title,
            'abstract': abstract,
            'published': pub_date,
            'authors': authors,
            'author_str': ', '.join(authors[:4]) + (' et al.' if len(authors) > 4 else ''),
            'categories': cats,
            'primary_cat': cats[0] if cats else '',
            'url': f"https://arxiv.org/abs/{clean_id}"
        }
    except Exception:
        return None


# ============================================================
# RSS FEED FETCHER — for "Today's New Papers" mode
# ============================================================

def fetch_arxiv_rss(categories, target_date=None, date_from=None, date_to=None, include_replacements=False):
    """Fetch papers from arXiv RSS feeds for given categories.

    Args:
        categories: list of category strings (e.g., ['quant-ph', 'cond-mat'])
        target_date: datetime object for single-date mode. If None, uses today.
        date_from: datetime object for date range mode (inclusive).
        date_to: datetime object for date range mode (inclusive).
        include_replacements: If False (default), filters out replacement submissions.

    Returns:
        (list of paper dicts, metadata dict) or ([], error_string)
    """
    if not categories:
        categories = DEFAULT_CATEGORIES

    # Determine target date range for filtering
    # All boundary datetimes must be timezone-aware (UTC) because
    # parsedate_to_datetime() returns timezone-aware datetimes from RSS pubDate.
    if date_from is not None and date_to is not None:
        filter_start = date_from.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        filter_end = date_to.replace(hour=23, minute=59, second=59, microsecond=0, tzinfo=timezone.utc)
        target_date = date_to
    elif target_date is not None:
        if target_date.tzinfo is None:
            target_date = target_date.replace(tzinfo=timezone.utc)
        filter_start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        filter_end = target_date.replace(hour=23, minute=59, second=59, microsecond=0)
    else:
        target_date = datetime.now(timezone.utc)
        filter_start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        filter_end = target_date.replace(hour=23, minute=59, second=59, microsecond=0)

    # arXiv RSS feed URL: multiple categories joined by +
    cat_str = '+'.join(categories)
    rss_url = f"https://rss.arxiv.org/rss/{cat_str}"

    try:
        r = requests.get(rss_url, timeout=30)
        if r.status_code != 200:
            return [], f"RSS feed returned status {r.status_code}"

        # Parse RSS XML
        root = ET.fromstring(r.content)

        # RSS namespaces
        ns = {
            'dc': 'http://purl.org/dc/elements/1.1/',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }

        papers = []
        seen_ids = set()
        filtered_count = 0

        for item in root.findall('.//item'):
            # Extract arXiv ID from <link>
            link_el = item.find('link')
            if link_el is None or not link_el.text:
                continue
            link = link_el.text.strip()
            # Extract ID: https://arxiv.org/abs/2604.20961 -> 2604.20961
            arxiv_id = link.replace('https://arxiv.org/abs/', '').replace('http://arxiv.org/abs/', '')
            # Remove version suffix (e.g., v1)
            arxiv_id = re.sub(r'v\d+$', '', arxiv_id)

            if arxiv_id in seen_ids:
                continue
            seen_ids.add(arxiv_id)

            # Extract announce type
            announce_type = ''
            # Method 1: arxiv:announce_type element
            at_el = item.find('arxiv:announce_type', ns)
            if at_el is not None and at_el.text:
                announce_type = at_el.text.strip().lower()
            else:
                # Method 2: Parse from description text
                desc_el = item.find('description')
                if desc_el is not None and desc_el.text:
                    desc = desc_el.text.strip()
                    m = re.search(r'Announce\s*Type:\s*(\S+)', desc, re.IGNORECASE)
                    if m:
                        announce_type = m.group(1).strip().lower()

            # Filter replacements unless explicitly included
            is_replacement = announce_type.startswith('replace')
            if is_replacement and not include_replacements:
                filtered_count += 1
                continue

            # Extract title
            title_el = item.find('title')
            title = title_el.text.strip() if title_el is not None and title_el.text else 'Unknown Title'
            # Clean up CDATA wrapper if present
            title = title.replace('<![CDATA[', '').replace(']]>', '')

            # Extract abstract from <description>
            desc_el = item.find('description')
            abstract = desc_el.text.strip() if desc_el is not None and desc_el.text else ''
            abstract = abstract.replace('<![CDATA[', '').replace(']]>', '')
            # Strip the announce type prefix from abstract if present
            abstract = re.sub(r'^arXiv:[\d.]+v?\d*\s+Announce\s*Type:\s*\S+\s*', '', abstract, flags=re.IGNORECASE).strip()

            # Extract authors from <dc:creator>
            creator_el = item.find('dc:creator', ns)
            authors = []
            if creator_el is not None and creator_el.text:
                author_str = creator_el.text.strip()
                authors = [a.strip() for a in author_str.split(',') if a.strip()]

            # Extract categories
            categories_found = []
            for cat_el in item.findall('category'):
                if cat_el.text:
                    categories_found.append(cat_el.text.strip())

            # Extract published date from <pubDate> — this is the ANNOUNCEMENT date
            pub_el = item.find('pubDate')
            pub_dt = None
            published = ''
            if pub_el is not None and pub_el.text:
                try:
                    # Format: Fri, 23 Aug 2024 00:00:00 -0400
                    from email.utils import parsedate_to_datetime
                    pub_dt = parsedate_to_datetime(pub_el.text.strip())
                    published = pub_dt.strftime('%Y-%m-%d')
                except Exception:
                    published = pub_el.text.strip()[:10]  # fallback

            # Filter by announcement date range (if specified)
            if pub_dt is not None:
                if pub_dt < filter_start or pub_dt > filter_end:
                    continue  # Skip items outside the target announcement date range
            elif published:
                # Fallback: string comparison for partial dates
                if published < filter_start.strftime('%Y-%m-%d') or published > filter_end.strftime('%Y-%m-%d'):
                    continue

            paper = {
                'title': title,
                'abstract': abstract,
                'authors': authors,
                'author_str': ', '.join(authors),
                'published': published,
                'url': f'https://arxiv.org/abs/{arxiv_id}',
                'arxiv_id': arxiv_id,
                'categories': categories_found,
                'source': 'rss',
                'announce_type': announce_type,
            }
            papers.append(paper)

        metadata = {
            'total_fetched': len(papers) + filtered_count,
            'total_available': len(papers) + filtered_count,
            'total_filtered': filtered_count,
            'replacements_included': include_replacements,
            'date_range_requested': target_date.strftime('%Y-%m-%d'),
            'date_range_covered': target_date.strftime('%Y-%m-%d'),
            'date_mode': 'new',
            'rss_url': rss_url,
            'limited': False,
            'categories_queried': categories,
        }

        return papers, metadata

    except ET.ParseError as e:
        return [], f"RSS XML parse error: {str(e)}"
    except Exception as e:
        return [], f"RSS fetch error: {str(e)}"


def _fetch_arxiv_chunk(date_from, date_to, categories, max_results=2000):
    """Fetch a single chunk of papers from arXiv for a specific date range.
    Returns (list of papers, total_available_for_this_chunk, error_msg)."""
    d0 = date_from.strftime('%Y%m%d%H%M')
    d1 = date_to.strftime('%Y%m%d%H%M')
    cat_q = ' OR '.join([f'cat:{c}' for c in categories])
    query = f"({cat_q}) AND submittedDate:[{d0} TO {d1}]"

    all_papers = []
    seen = set()
    total_available = 0
    page = 0
    batch_size = 500
    absolute_cap = min(max_results, 2000)
    error_msg = None
    last_http_status = None

    while True:
        start = page * batch_size
        chunk_limit = min(batch_size, absolute_cap - len(all_papers))
        if chunk_limit <= 0:
            break
        url = (f"http://export.arxiv.org/api/query"
               f"?search_query={requests.utils.quote(query)}"
               f"&start={start}&max_results={chunk_limit}"
               f"&sortBy=submittedDate&sortOrder=descending")
        try:
            r = requests.get(url, timeout=120)
            last_http_status = r.status_code
            if r.status_code == 429:
                error_msg = "HTTP 429 (rate limited)"
                break
            if r.status_code != 200:
                error_msg = f"HTTP {r.status_code}"
                break
            root = ET.fromstring(r.content)
            if page == 0:
                total_el = root.find('opensearch:totalResults', ARXIV_NS)
                if total_el is not None and total_el.text:
                    try:
                        total_available = int(total_el.text)
                    except ValueError:
                        pass
            entries = root.findall('atom:entry', ARXIV_NS)
            if not entries:
                break
            for entry in entries:
                paper = _parse_entry(entry)
                if paper and paper['arxiv_id'] not in seen:
                    seen.add(paper['arxiv_id'])
                    all_papers.append(paper)
            if len(entries) < chunk_limit:
                break
            page += 1
            time.sleep(4)  # Slightly longer between pages
        except requests.exceptions.Timeout:
            error_msg = f"timeout ({120}s)"
            break
        except requests.exceptions.ConnectionError as e:
            error_msg = f"connection error: {str(e)[:80]}"
            break
        except ET.ParseError as e:
            error_msg = f"XML parse error: {str(e)[:80]}"
            break
        except Exception as e:
            error_msg = f"unexpected: {str(e)[:80]}"
            break

    return all_papers, total_available, error_msg


def fetch_arxiv_papers(date_from=None, date_to=None, days_back=None, categories=None, max_per_chunk=2000, date_mode='submitted', include_replacements=False):
    """Fetch papers from arXiv.

    Supports three modes:
    - 'submitted': date-chunk partitioning by submittedDate (default)
    - 'announcement': widen submission window, filter by published date
    - 'new': use arXiv RSS feeds to get exactly the papers announced today

    Args:
        date_from: datetime object for start date
        date_to: datetime object for end date
        days_back: alternative to date_from — number of days before date_to
        categories: list of category strings
        max_per_chunk: max papers to fetch per date chunk
        date_mode: 'submitted', 'announcement', or 'new'
        include_replacements: for RSS mode, whether to include replacement submissions

    Returns:
        (papers, metadata) or ([], error_string)
    """
    if categories is None:
        categories = DEFAULT_CATEGORIES

    # --- NEW MODE: Use arXiv RSS feeds ---
    if date_mode == 'new':
        if date_to is None:
            date_to = datetime.now()
        return fetch_arxiv_rss(categories, target_date=date_to, include_replacements=include_replacements)

    if date_to is None:
        date_to = datetime.now()
    if date_from is None:
        if days_back is None:
            days_back = 3
        date_from = date_to - timedelta(days=days_back)

    # Clamp: max 30 days per fetch to avoid abuse
    if (date_to - date_from).days > 30:
        date_from = date_to - timedelta(days=30)

    # Save the user-requested range for announcement filtering
    requested_from = date_from
    requested_to = date_to

    # --- ANNOUNCEMENT MODE: Hybrid RSS + API approach ---
    # The arXiv RSS feed only contains the most recent day's announcements
    # and does not support historical queries. For the "today" use case,
    # we use RSS directly (most accurate for announcement date). For any
    # date range that might include historical data, we fall back to the
    # API with a widened submission window and heuristically filter.
    if date_mode == 'announcement':
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        is_today_only = (requested_from.date() == today.date() and 
                         requested_to.date() == today.date())
        is_yesterday = (requested_from.date() == (today - timedelta(days=1)).date() and
                       requested_to.date() == (today - timedelta(days=1)).date())

        if is_today_only or is_yesterday:
            # Use RSS feed — accurate for recent dates
            all_papers = []
            seen = set()
            errors = []

            for cat_set in _chunk_categories(categories):
                papers, result = fetch_arxiv_rss(
                    cat_set,
                    date_from=requested_from,
                    date_to=requested_to,
                    include_replacements=include_replacements
                )
                if isinstance(result, str):
                    errors.append(result)
                else:
                    for p in papers:
                        if p['arxiv_id'] not in seen:
                            seen.add(p['arxiv_id'])
                            all_papers.append(p)

            metadata = {
                'total_fetched': len(all_papers),
                'date_range_requested': f"{requested_from.strftime('%Y-%m-%d')} to {requested_to.strftime('%Y-%m-%d')}",
                'date_mode': date_mode,
                'source': 'rss',
                'categories_queried': categories,
                'chunks': [],
                'chunk_errors': errors,
            }

            if not all_papers:
                return [], {'error': f'No papers announced on {requested_from.strftime("%Y-%m-%d")} yet. ArXiv typically announces between 00:00-08:00 UTC. Try again later or use "Submission Date" mode.', **metadata}

            return all_papers, metadata
        else:
            # Historical date: fall back to API with widened submission window
            # Papers announced on date D were typically submitted D-1 to D-3 business days before
            _log('progress', f"Announcement mode for historical date: widening search to catch submissions from preceding days...")
            date_from = requested_from - timedelta(days=5)
            date_to = requested_to
            # Continue to the submittedDate API path below

    # --- SUBMITTED DATE MODE: Use arXiv API with submittedDate query ---
    # Partition strategy: 2-day chunks for wide ranges, 1-day for narrow
    delta_days = (date_to - date_from).days
    if delta_days <= 3:
        chunk_days = 1
    elif delta_days <= 7:
        chunk_days = 2
    else:
        chunk_days = 3

    all_papers = []
    seen = set()
    chunk_info = []
    chunk_errors = []
    total_available_across_chunks = 0

    current = date_from.replace(hour=0, minute=0, second=0, microsecond=0)
    final_end = date_to.replace(hour=23, minute=59, second=59, microsecond=0)
    while current <= final_end:
        chunk_boundary = min(current + timedelta(days=chunk_days), final_end + timedelta(seconds=1))

        # Retry loop with exponential backoff for transient errors
        chunk_papers, chunk_total, chunk_err = None, 0, None
        for attempt in range(3):
            chunk_papers, chunk_total, chunk_err = _fetch_arxiv_chunk(
                current, chunk_boundary, categories, max_results=max_per_chunk
            )
            if chunk_err is None:
                break  # Success
            if attempt < 2:
                wait = 6 * (attempt + 1)  # 6s, 12s backoff
                time.sleep(wait)

        if chunk_err:
            chunk_errors.append(f"{current.strftime('%Y-%m-%d')}: {chunk_err}")

        total_available_across_chunks += chunk_total
        chunk_info.append({
            'range': f"{current.strftime('%Y-%m-%d')} to {(chunk_boundary - timedelta(seconds=1)).strftime('%Y-%m-%d')}",
            'fetched': len(chunk_papers) if chunk_papers else 0,
            'available': chunk_total,
            'error': chunk_err,
        })

        for p in (chunk_papers or []):
            if p['arxiv_id'] not in seen:
                seen.add(p['arxiv_id'])
                all_papers.append(p)

        current = chunk_boundary
        if current <= final_end:
            time.sleep(6)  # Conservative rate limit between chunks

    # Build metadata
    if all_papers:
        dates = [p['published'] for p in all_papers if p['published']]
        earliest = min(dates) if dates else ''
        latest = max(dates) if dates else ''
    else:
        earliest = latest = ''

    metadata = {
        'total_fetched': len(all_papers),
        'total_available': total_available_across_chunks,
        'date_range_requested': f"{requested_from.strftime('%Y-%m-%d')} to {requested_to.strftime('%Y-%m-%d')}",
        'date_range_covered': f"{earliest} to {latest}" if earliest else "none",
        'date_mode': date_mode,
        'filtered_out': 0,
        'limited': total_available_across_chunks > len(all_papers),
        'categories_queried': categories,
        'chunks': chunk_info,
        'chunk_errors': chunk_errors,
    }

    return all_papers, metadata


# ============================================================
# RANKING
# ============================================================
def _rank_individual(papers, keywords, publications, llm_api_key=None, llm_provider='moonshot', llm_model=None, publication_abstracts=None, max_llm_candidates=50, llm_input_mode='titles_and_abstracts', llm_target='top', tfidf_w=25, kw_w=25, llm_w=50):
    """
    Rank papers using three-way weighted blend of TF-IDF, keyword count, and LLM score.
    
    Args:
        max_llm_candidates: how many top TF-IDF papers get blended LLM scoring
        tfidf_w, kw_w, llm_w: relative weights for the three score components (auto-normalized)
    
    Returns:
        dict with 'main' (blended ranked list), 'llm_rescue_pure', 'llm_rescue_weak'
    """
    if not papers:
        return {'main': [], 'llm_rescue_pure': [], 'llm_rescue_weak': []}
        
    abstracts_tuple = tuple(publication_abstracts) if publication_abstracts else ()
    vectorizer, profile_vec = get_profile_vector(tuple(keywords), tuple(publications), abstracts_tuple)

    # --- Keyword matching (soft, never drops) ---
    kw_patterns = [(kw, re.compile(r'\b' + re.escape(normalize_umlauts(kw.lower())) + r'\b', re.I)) for kw in keywords]
    all_entries = []
    for p in papers:
        text = normalize_umlauts(f"{p['title']} {p['abstract']}".lower())
        matched = [kw for kw, pat in kw_patterns if pat.search(text)]
        all_entries.append((p, len(matched), matched))

    # --- TF-IDF on ALL papers ---
    texts = [f"{p['title']}. {p['abstract']}" for p, _, _ in all_entries]
    pm = vectorizer.transform(texts)
    pd = np.asarray(pm.toarray())
    norms = np.linalg.norm(pd, axis=1, keepdims=True)
    norms[norms == 0] = 1
    pd = pd / norms
    tfidf_scores = pd @ profile_vec

    # Keyword bonus (soft, 0-1)
    mc = np.array([m for _, m, _ in all_entries])
    max_m = mc.max() if mc.max() > 0 else 1
    kw_scores = mc / max_m

    # Pre-compute normalized raw scores for three-way blend
    tfidf_raw_norm = np.zeros_like(tfidf_scores)
    if tfidf_scores.max() > tfidf_scores.min():
        tfidf_raw_norm = (tfidf_scores - tfidf_scores.min()) / (tfidf_scores.max() - tfidf_scores.min())
    kw_raw_norm = kw_scores  # already 0-1
    # llm_raw_norm computed after LLM scoring below

    # --- TRACK A: Blended LLM on top TF-IDF candidates ---
    tfidf_order = np.argsort(tfidf_scores)[::-1]  # descending
    bottom_n = min(max_llm_candidates, len(all_entries))

    if llm_target == 'bottom':
        # Score the LOWEST TF-IDF papers — rescue what the algorithm missed
        top_indices = tfidf_order[-bottom_n:]  # bottom N by TF-IDF
    else:
        # Score the HIGHEST TF-IDF papers — boost what the algorithm found
        top_indices = tfidf_order[:bottom_n]  # top N by TF-IDF
    top_set = set(top_indices)

    llm_scores_raw = np.full(len(all_entries), 0.5)
    llm_comments_raw = [''] * len(all_entries)
    llm_error = None

    if llm_api_key:
        top_papers_for_llm = [all_entries[i][0] for i in top_indices]
        llm_result = score_with_llm(top_papers_for_llm, keywords, publications,
                                     llm_api_key, llm_provider=llm_provider,
                                     model=llm_model if llm_model else None,
                                     input_mode=llm_input_mode,
                                     target=llm_target)
        if llm_result is not None:
            for rank_pos, orig_idx in enumerate(top_indices):
                if rank_pos < len(llm_result['scores']):
                    llm_scores_raw[orig_idx] = llm_result['scores'][rank_pos]
                    llm_comments_raw[orig_idx] = llm_result['comments'][rank_pos] if rank_pos < len(llm_result['comments']) else ''
            llm_error = llm_result.get('error')
            llm_scores = llm_scores_raw
            llm_comments = llm_comments_raw
        else:
            llm_scores = None
            llm_comments = []
    else:
        llm_scores = None
        llm_comments = []
    
    # Normalize LLM scores for live frontend reblending (0.5 default → 0.0)
    llm_raw_norm = (llm_scores_raw - 0.5) / 0.5 if llm_scores_raw is not None else np.zeros(len(all_entries))
    llm_raw_norm = np.clip(llm_raw_norm, 0, 1)

    # Three-way blend: TF-IDF + Keyword + LLM (weights auto-normalize)
    weight_sum = tfidf_w + kw_w + llm_w
    if weight_sum <= 0:
        weight_sum = 1
    final_scores = (tfidf_w * tfidf_raw_norm + kw_w * kw_raw_norm + llm_w * llm_raw_norm) / weight_sum

    if final_scores.max() > final_scores.min():
        final_scores = (final_scores - final_scores.min()) / (final_scores.max() - final_scores.min())
    
    # Alias for backward compatibility in sorting and display
    tfidf_combined = final_scores

    final_order = np.argsort(final_scores)[::-1]

    # Build main track
    main = []
    for rank_pos in final_order:
        p, kw_count, matched_kws = all_entries[rank_pos]
        score = final_scores[rank_pos]
        mentions = {}
        text = normalize_umlauts(f"{p['title']} {p['abstract']}".lower())
        for kw in keywords:
            # Use \b word boundaries to prevent substring matches:
            # "ite" must NOT match "finite", "despite", "limited"
            cnt = len(re.findall(r'\b' + re.escape(normalize_umlauts(kw.lower())) + r'\b', text))
            if cnt > 0:
                mentions[kw] = cnt
        mention_str = " | ".join([f"{kw} ({cnt}x)" for kw, cnt in sorted(mentions.items(), key=lambda x: -x[1])[:6]])
        if not mention_str:
            mention_str = "No direct keyword matches — rescued by TF-IDF/LLM"
        comment = generate_comment(p, mentions, p['categories'])
        entry = dict(p)
        entry['score'] = round(float(score), 3)
        entry['mention_str'] = mention_str
        entry['comment'] = comment
        entry['keyword_matches'] = kw_count
        entry['tfidf_score'] = round(float(tfidf_combined[rank_pos]), 3)
        # Raw scores for live frontend reblending
        entry['_tfidf'] = round(float(tfidf_raw_norm[rank_pos]), 3)
        entry['_kw'] = round(float(kw_raw_norm[rank_pos]), 3)
        entry['_llm'] = round(float(llm_raw_norm[rank_pos]), 3)
        # Only expose llm_score if this paper was actually scored by LLM (avoid 0.500 clutter)
        if llm_scores is not None and rank_pos in top_set:
            entry['llm_score'] = round(float(llm_scores[rank_pos]), 3)
            if llm_comments[rank_pos]:
                entry['llm_comment'] = llm_comments[rank_pos]
        main.append(entry)

    # --- TRACK B: Split Rescue Tier for papers the LLM missed ---
    llm_rescue_pure = []
    llm_rescue_weak = []
    rescue_candidates_count = 0
    if llm_api_key:
        # Candidates: papers NOT scored by Track A's LLM
        unpicked = [
            (i, all_entries[i][0], all_entries[i][1], all_entries[i][2])
            for i in range(len(all_entries))
            if i not in top_set
        ]
        rescue_candidates_count = len(unpicked)
        # Sort by keyword count ascending (lowest first), then by TF-IDF score
        unpicked.sort(key=lambda x: (x[2], tfidf_combined[x[0]]))
        
        # Split into two groups
        pure = [x for x in unpicked if x[2] == 0][:15]
        weak = [x for x in unpicked if 1 <= x[2] <= 2][:15]

        def _build_rescue_group_individual(group_list, label, color_tag):
            if not group_list:
                return []
            group_papers = [p for _, p, _, _ in group_list]
            group_result = score_with_llm(group_papers, keywords, publications,
                                          llm_api_key, llm_provider=llm_provider,
                                          model=llm_model if llm_model else None,
                                          input_mode=llm_input_mode,
                                          target='bottom')
            result = []
            if group_result is not None:
                for rank_pos, (orig_idx, orig_p, orig_kw_count, matched_kws) in enumerate(group_list):
                    if rank_pos < len(group_result['scores']):
                        score = group_result['scores'][rank_pos]
                        comment = group_result['comments'][rank_pos] if rank_pos < len(group_result['comments']) else ''
                        entry = dict(orig_p)
                        entry['score'] = round(float(score), 3)
                        entry['llm_pure_score'] = round(float(score), 3)
                        if matched_kws:
                            entry['mention_str'] = " | ".join([f"{kw} (matched)" for kw in matched_kws[:5]]) + (f" (+{len(matched_kws)-5} more)" if len(matched_kws) > 5 else "")
                        else:
                            entry['mention_str'] = "0 keyword matches — discovered purely by LLM semantics"
                        entry['comment'] = f"LLM rescue ({label}): algorithmic filter missed this — semantic rescue"
                        entry['keyword_matches'] = orig_kw_count
                        entry['llm_comment'] = comment if comment else ""
                        entry['track'] = color_tag
                        # Raw scores for live frontend reblending
                        entry['_tfidf'] = round(float(tfidf_raw_norm[orig_idx]), 3)
                        entry['_kw'] = round(float(kw_raw_norm[orig_idx]), 3)
                        entry['_llm'] = round(float(llm_raw_norm[orig_idx]), 3)
                        result.append(entry)
                result.sort(key=lambda x: x['score'], reverse=True)
            return result

        llm_rescue_pure = _build_rescue_group_individual(pure, 'Pure Semantic', 'llm_rescue_pure')
        llm_rescue_weak = _build_rescue_group_individual(weak, 'Keyword-Weak', 'llm_rescue_weak')

    # --- COMMENT FILL-IN: top main-track papers lacking LLM comments get a lightweight pass ---
    if llm_api_key:
        missing_comment_indices = [
            i for i, entry in enumerate(main[:15])
            if not entry.get('llm_comment')
        ]
        if missing_comment_indices:
            fill_papers = [main[i] for i in missing_comment_indices]
            fill_result = score_with_llm(
                fill_papers, keywords, publications,
                llm_api_key, llm_provider=llm_provider,
                model=llm_model if llm_model else None,
                input_mode='titles_only',  # lightweight — just need comment
                target='top'
            )
            if fill_result is not None:
                for rank_pos, orig_idx in enumerate(missing_comment_indices):
                    if rank_pos < len(fill_result['comments']):
                        comment = fill_result['comments'][rank_pos]
                        if comment:
                            main[orig_idx]['llm_comment'] = comment
                            # Also update score if it was default
                            if rank_pos < len(fill_result['scores']):
                                new_score = fill_result['scores'][rank_pos]
                                # Only update if the paper had a default-ish score
                                if main[orig_idx].get('llm_score', 0.5) == 0.5:
                                    main[orig_idx]['llm_score'] = round(float(new_score), 3)
                                    main[orig_idx]['score'] = round(float(new_score), 3)

    if llm_error and main:
        main[0]['_llm_error'] = llm_error

    return {'main': main, 'llm_rescue_pure': llm_rescue_pure, 'llm_rescue_weak': llm_rescue_weak, 'rescue_candidates_count': rescue_candidates_count}

def _rank_batch_filter(papers, keywords, publications, llm_api_key=None, llm_provider='moonshot', llm_model=None, publication_abstracts=None, max_llm_candidates=100, llm_input_mode='titles_and_abstracts', tfidf_w=25, kw_w=25, llm_w=50, chunked_scoring=False, batch_filter_ratio=0.4):
    """
    Mode B: Batch filter + deep scoring.
    1 call picks candidates from titles, then deep scores only those with abstracts.
    """
    if not papers:
        return {'main': [], 'llm_rescue': [], 'rescue_candidates_count': 0}

    abstracts_tuple = tuple(publication_abstracts) if publication_abstracts else ()
    vectorizer, profile_vec = get_profile_vector(tuple(keywords), tuple(publications), abstracts_tuple)

    # Keyword matching with umlaut normalization
    kw_patterns = [(kw, re.compile(r'\b' + re.escape(normalize_umlauts(kw.lower())) + r'\b', re.I)) for kw in keywords]
    all_entries = []
    for p in papers:
        text = normalize_umlauts(f"{p['title']} {p['abstract']}".lower())
        matched = [kw for kw, pat in kw_patterns if pat.search(text)]
        all_entries.append((p, len(matched), matched))

    # TF-IDF
    texts = [f"{p['title']}. {p['abstract']}" for p, _, _ in all_entries]
    pm = vectorizer.transform(texts)
    pd = np.asarray(pm.toarray())
    norms = np.linalg.norm(pd, axis=1, keepdims=True)
    norms[norms == 0] = 1
    pd = pd / norms
    tfidf_scores = pd @ profile_vec

    mc = np.array([m for _, m, _ in all_entries])
    max_m = mc.max() if mc.max() > 0 else 1
    kw_scores = mc / max_m

    # Pre-compute normalized raw scores for three-way blend
    tfidf_raw_norm = np.zeros_like(tfidf_scores)
    if tfidf_scores.max() > tfidf_scores.min():
        tfidf_raw_norm = (tfidf_scores - tfidf_scores.min()) / (tfidf_scores.max() - tfidf_scores.min())
    kw_raw_norm = kw_scores  # already 0-1
    
    # Base score for sorting and display (TF-IDF + keyword, without LLM)
    base_weight_sum = tfidf_w + kw_w
    if base_weight_sum <= 0:
        base_weight_sum = 1
    tfidf_combined = (tfidf_w * tfidf_raw_norm + kw_w * kw_raw_norm) / base_weight_sum
    if tfidf_combined.max() > tfidf_combined.min():
        tfidf_combined = (tfidf_combined - tfidf_combined.min()) / (tfidf_combined.max() - tfidf_combined.min())

    main = []
    llm_error = None
    llm_rescue_pure = []
    llm_rescue_weak = []
    rescue_candidates_count = 0

    if llm_api_key:
        # Step 1: Select top candidates by TF-IDF for the batch filter
        tfidf_order = np.argsort(tfidf_combined)[::-1]
        candidate_count = min(max_llm_candidates, len(all_entries))
        candidate_indices = tfidf_order[:candidate_count]
        candidate_papers = [all_entries[i][0] for i in candidate_indices]

        # Step 2: Batch filter — 1 call, LLM picks interesting papers by number
        picked_indices = batch_filter_llm(
            candidate_papers, keywords, publications,
            llm_api_key, llm_provider=llm_provider,
            model=llm_model if llm_model else None,
            max_candidates=candidate_count,
            pick_ratio=batch_filter_ratio
        )

        # Map picked indices back to original all_entries indices
        picked_orig_indices = [candidate_indices[i] for i in picked_indices if i < len(candidate_indices)]
        picked_set = set(picked_orig_indices)

        # Track which papers received real deep LLM scoring
        deep_scored_set = set(picked_orig_indices)

        if picked_orig_indices:
            # Step 3: Deep scoring with full abstracts for picked papers only
            deep_papers = [all_entries[i][0] for i in picked_orig_indices]
            if chunked_scoring:
                # v142: Chunked mode — split into batches of 30 for reliability
                BATCH_SIZE = 30
                deep_scores_all = []
                deep_comments_all = []
                llm_errors = []
                for batch_start in range(0, len(picked_orig_indices), BATCH_SIZE):
                    batch_end = min(batch_start + BATCH_SIZE, len(picked_orig_indices))
                    batch_picked = picked_orig_indices[batch_start:batch_end]
                    batch_papers = [all_entries[i][0] for i in batch_picked]
                    batch_result = score_with_llm(
                        batch_papers, keywords, publications,
                        llm_api_key, llm_provider=llm_provider,
                        model=llm_model if llm_model else None,
                        input_mode=llm_input_mode,
                        target='top'
                    )
                    if batch_result is not None:
                        deep_scores_all.extend(batch_result['scores'])
                        deep_comments_all.extend(batch_result['comments'])
                        if batch_result.get('error'):
                            llm_errors.append(batch_result['error'])
                    else:
                        deep_scores_all.extend([0.5] * len(batch_papers))
                        deep_comments_all.extend([''] * len(batch_papers))
                deep_result = {
                    'scores': np.array(deep_scores_all[:len(picked_orig_indices)]),
                    'comments': deep_comments_all[:len(picked_orig_indices)],
                    'error': '; '.join(llm_errors) if llm_errors else None
                }
            else:
                # Default: single call with full context (v140 and prior behavior)
                deep_result = score_with_llm(
                    deep_papers, keywords, publications,
                    llm_api_key, llm_provider=llm_provider,
                    model=llm_model if llm_model else None,
                    input_mode=llm_input_mode,
                    target='top'
                )

            # Build LLM scores for all entries (default 0.5 for unpicked)
            llm_scores = np.full(len(all_entries), 0.5)
            llm_comments = [''] * len(all_entries)
            if deep_result is not None:
                for rank_pos, orig_idx in enumerate(picked_orig_indices):
                    if rank_pos < len(deep_result['scores']):
                        llm_scores[orig_idx] = deep_result['scores'][rank_pos]
                        llm_comments[orig_idx] = deep_result['comments'][rank_pos] if rank_pos < len(deep_result['comments']) else ''
                llm_error = deep_result.get('error')

                # Normalize LLM scores
                llm_raw_norm = (llm_scores - 0.5) / 0.5
                llm_raw_norm = np.clip(llm_raw_norm, 0, 1)
                # v151: floor for papers the batch filter didn't pick for deep scoring.
                # Without this, papers with strong keyword/TF-IDF signals get llm_norm=0
                # and are crushed when LLM weight is high (e.g. 50%). The floor lets
                # them remain competitive without affecting truly scored papers.
                unscored_mask = np.array([i not in deep_scored_set for i in range(len(all_entries))])
                llm_raw_norm[unscored_mask] = np.maximum(llm_raw_norm[unscored_mask], 0.15)

                # Three-way blend: TF-IDF + Keyword + LLM
                weight_sum = tfidf_w + kw_w + llm_w
                if weight_sum <= 0:
                    weight_sum = 1
                final_scores = (tfidf_w * tfidf_raw_norm + kw_w * kw_raw_norm + llm_w * llm_raw_norm) / weight_sum
                if final_scores.max() > final_scores.min():
                    final_scores = (final_scores - final_scores.min()) / (final_scores.max() - final_scores.min())
                
                # Alias for backward compatibility
                tfidf_combined = final_scores

                final_order = np.argsort(final_scores)[::-1]

                for rank_pos in final_order:
                    p, kw_count, matched_kws = all_entries[rank_pos]
                    score = final_scores[rank_pos]
                    mentions = {}
                    text = normalize_umlauts(f"{p['title']} {p['abstract']}".lower())
                    for kw in keywords:
                        cnt = len(re.findall(r'\b' + re.escape(normalize_umlauts(kw.lower())) + r'\b', text))
                        if cnt > 0:
                            mentions[kw] = cnt
                    mention_str = " | ".join([f"{kw} ({cnt}x)" for kw, cnt in sorted(mentions.items(), key=lambda x: -x[1])[:6]])
                    if not mention_str:
                        mention_str = "No direct keyword matches — discovered by batch LLM filter"
                    comment = generate_comment(p, mentions, p['categories'])
                    entry = dict(p)
                    entry['score'] = round(float(score), 3)
                    entry['mention_str'] = mention_str
                    entry['comment'] = comment
                    entry['keyword_matches'] = kw_count
                    entry['tfidf_score'] = round(float(tfidf_combined[rank_pos]), 3)
                    # Raw scores for live frontend reblending
                    entry['_tfidf'] = round(float(tfidf_raw_norm[rank_pos]), 3)
                    entry['_kw'] = round(float(kw_raw_norm[rank_pos]), 3)
                    entry['_llm'] = round(float(llm_raw_norm[rank_pos]), 3)
                    # Only expose llm_score if this paper was actually deep-scored (avoid 0.500 clutter)
                    if rank_pos in deep_scored_set:
                        entry['llm_score'] = round(float(llm_scores[rank_pos]), 3)
                        if llm_comments[rank_pos]:
                            entry['llm_comment'] = llm_comments[rank_pos]
                    main.append(entry)
            else:
                # Deep scoring failed — fall back to TF-IDF
                final_order = np.argsort(tfidf_combined)[::-1]
                for rank_pos in final_order:
                    p, kw_count, matched_kws = all_entries[rank_pos]
                    score = tfidf_combined[rank_pos]
                    mentions = {}
                    text = normalize_umlauts(f"{p['title']} {p['abstract']}".lower())
                    for kw in keywords:
                        cnt = len(re.findall(r'\b' + re.escape(normalize_umlauts(kw.lower())) + r'\b', text))
                        if cnt > 0:
                            mentions[kw] = cnt
                    mention_str = " | ".join([f"{kw} ({cnt}x)" for kw, cnt in sorted(mentions.items(), key=lambda x: -x[1])[:6]])
                    comment = generate_comment(p, mentions, p['categories'])
                    entry = dict(p)
                    entry['score'] = round(float(score), 3)
                    entry['mention_str'] = mention_str
                    entry['comment'] = comment
                    entry['keyword_matches'] = kw_count
                    entry['tfidf_score'] = round(float(score), 3)
                    entry['_tfidf'] = round(float(tfidf_raw_norm[rank_pos]), 3)
                    entry['_kw'] = round(float(kw_raw_norm[rank_pos]), 3)
                    entry['_llm'] = 0.0  # no LLM scoring in this path
                    main.append(entry)
        else:
            # Batch filter picked nothing — fall back to TF-IDF only
            final_order = np.argsort(tfidf_combined)[::-1]
            for rank_pos in final_order:
                p, kw_count, matched_kws = all_entries[rank_pos]
                score = tfidf_combined[rank_pos]
                mentions = {}
                text = normalize_umlauts(f"{p['title']} {p['abstract']}".lower())
                for kw in keywords:
                    cnt = len(re.findall(r'\b' + re.escape(normalize_umlauts(kw.lower())) + r'\b', text))
                    if cnt > 0:
                        mentions[kw] = cnt
                mention_str = " | ".join([f"{kw} ({cnt}x)" for kw, cnt in sorted(mentions.items(), key=lambda x: -x[1])[:6]])
                comment = generate_comment(p, mentions, p['categories'])
                entry = dict(p)
                entry['score'] = round(float(score), 3)
                entry['mention_str'] = mention_str
                entry['comment'] = comment
                entry['keyword_matches'] = kw_count
                entry['tfidf_score'] = round(float(score), 3)
                entry['_tfidf'] = round(float(tfidf_raw_norm[rank_pos]), 3)
                entry['_kw'] = round(float(kw_raw_norm[rank_pos]), 3)
                entry['_llm'] = 0.0  # no LLM scoring in this path
                main.append(entry)

        # --- SPLIT RESCUE TIER: always run when LLM is enabled ---
        if llm_api_key:
            # Exclude top 25 main-track papers from rescue (they already got visibility)
            main_top_urls = set()
            for ent in main[:25]:
                main_top_urls.add(ent.get('url', ''))

            # Candidates: ALL unpicked papers, excluding top 25 main
            unpicked = [
                (i, all_entries[i][0], all_entries[i][1], all_entries[i][2])
                for i in range(len(all_entries))
                if all_entries[i][0].get('url', '') not in main_top_urls and i not in picked_set
            ]
            rescue_candidates_count = len(unpicked)
            
            # Sort by keyword count ascending (lowest first), then by TF-IDF score
            unpicked.sort(key=lambda x: (x[2], tfidf_combined[x[0]]))
            
            # Split into two groups
            pure = [x for x in unpicked if x[2] == 0][:10]      # 0 keyword matches
            weak = [x for x in unpicked if 1 <= x[2] <= 2][:10]  # 1-2 keyword matches

            def _build_rescue_group(group_list, label, color_tag):
                if not group_list:
                    return []
                # Score with LLM
                group_papers = [p for _, p, _, _ in group_list]
                group_scores = {}
                group_comments = {}
                group_result = score_with_llm(
                    group_papers, keywords, publications,
                    llm_api_key, llm_provider=llm_provider,
                    model=llm_model if llm_model else None,
                    input_mode='titles_and_abstracts',
                    target='bottom'
                )
                if group_result is not None:
                    for g_pos, (_, _, _, _) in enumerate(group_list):
                        if g_pos < len(group_result['scores']):
                            group_scores[g_pos] = group_result['scores'][g_pos]
                        if g_pos < len(group_result['comments']):
                            group_comments[g_pos] = group_result['comments'][g_pos]

                result = []
                for g_pos, (orig_idx, orig_p, orig_kw_count, matched_kws) in enumerate(group_list):
                    g_score = group_scores.get(g_pos, 0.5)
                    g_comment = group_comments.get(g_pos, '')
                    r_entry = dict(orig_p)
                    r_entry['score'] = round(float(g_score), 3)
                    r_entry['llm_pure_score'] = round(float(g_score), 3)
                    r_entry['tfidf_score'] = round(float(tfidf_combined[orig_idx]), 3)
                    # Raw scores for live frontend reblending
                    r_entry['_tfidf'] = round(float(tfidf_raw_norm[orig_idx]), 3)
                    r_entry['_kw'] = round(float(kw_raw_norm[orig_idx]), 3)
                    r_entry['_llm'] = round(float((g_score - 0.5) / 0.5), 3)  # normalize rescue LLM score
                    if matched_kws:
                        r_entry['mention_str'] = " | ".join([f"{kw} (matched)" for kw in matched_kws[:5]]) + (f" (+{len(matched_kws)-5} more)" if len(matched_kws) > 5 else "")
                    else:
                        r_entry['mention_str'] = "0 keyword matches — discovered purely by LLM semantics"
                    r_entry['comment'] = f"LLM rescue ({label}): low keyword overlap — semantic relevance check"
                    r_entry['keyword_matches'] = orig_kw_count
                    r_entry['llm_comment'] = g_comment if g_comment else f"Rescue LLM ({label}): no semantic relevance detected"
                    r_entry['track'] = color_tag
                    result.append(r_entry)
                result.sort(key=lambda x: x['score'], reverse=True)
                return result

            llm_rescue_pure = _build_rescue_group(pure, 'Pure Semantic', 'llm_rescue_pure')
            llm_rescue_weak = _build_rescue_group(weak, 'Keyword-Weak', 'llm_rescue_weak')

        # --- COMMENT FILL-IN for main-track papers missing LLM comments ---
        if main and llm_api_key:
            missing_with_idx = [
                (i, ent) for i, ent in enumerate(main)
                if not ent.get('llm_comment') and i < 50  # only top 50 papers get comments
            ]
            if missing_with_idx:
                # BATCHED comment fill-in: one API call for all missing comments
                fill_papers = [ent for _, ent in missing_with_idx]
                batch_comments = batch_fill_comments(
                    fill_papers, keywords, publications,
                    llm_api_key, llm_provider=llm_provider,
                    model=llm_model if llm_model else None
                )
                if batch_comments is not None:
                    for f_pos, (m_idx, _) in enumerate(missing_with_idx):
                        if f_pos < len(batch_comments):
                            fc = batch_comments[f_pos]
                            if fc:
                                main[m_idx]['llm_comment'] = fc
                                # Also update score if meaningful (>0.55 or <0.45)
                                # But batch fill-in only returns comments, not scores

        # --- COMMENT FILL-IN for rescue tier papers missing comments ---
        for r_list, r_name in [(llm_rescue_pure, 'llm_rescue_pure'), (llm_rescue_weak, 'llm_rescue_weak')]:
            if r_list and llm_api_key:
                missing_rescue = [r for r in r_list if not r.get('llm_comment')]
                if missing_rescue:
                    batch_comments = batch_fill_comments(
                        missing_rescue, keywords, publications,
                        llm_api_key, llm_provider=llm_provider,
                        model=llm_model if llm_model else None
                    )
                    if batch_comments is not None:
                        for r_pos, r_ent in enumerate(r_list):
                            if r_pos < len(batch_comments) and not r_ent.get('llm_comment'):
                                rc = batch_comments[r_pos]
                                if rc:
                                    r_ent['llm_comment'] = rc
    else:
        # No LLM key — pure TF-IDF
        final_order = np.argsort(tfidf_combined)[::-1]
        for rank_pos in final_order:
            p, kw_count, matched_kws = all_entries[rank_pos]
            score = tfidf_combined[rank_pos]
            mentions = {}
            text = normalize_umlauts(f"{p['title']} {p['abstract']}".lower())
            for kw in keywords:
                cnt = len(re.findall(r'\b' + re.escape(normalize_umlauts(kw.lower())) + r'\b', text))
                if cnt > 0:
                    mentions[kw] = cnt
            mention_str = " | ".join([f"{kw} ({cnt}x)" for kw, cnt in sorted(mentions.items(), key=lambda x: -x[1])[:6]])
            comment = generate_comment(p, mentions, p['categories'])
            entry = dict(p)
            entry['score'] = round(float(score), 3)
            entry['mention_str'] = mention_str
            entry['comment'] = comment
            entry['keyword_matches'] = kw_count
            entry['tfidf_score'] = round(float(score), 3)
            main.append(entry)

    if llm_error and main:
        main[0]['_llm_error'] = llm_error

    return {'main': main, 'llm_rescue_pure': llm_rescue_pure, 'llm_rescue_weak': llm_rescue_weak, 'rescue_candidates_count': rescue_candidates_count}

def rank_papers(papers, keywords, publications, llm_api_key=None, llm_provider='moonshot', llm_model=None, publication_abstracts=None, max_llm_candidates=50, llm_input_mode='titles_and_abstracts', llm_target='top', llm_mode='individual', tfidf_w=25, kw_w=25, llm_w=50, chunked_scoring=False, batch_filter_ratio=0.4):
    """
    Dispatcher — routes to the appropriate ranking mode.
    
    llm_mode:
        'individual'  → Mode A: individual calls, two tracks + fill-in
        'batch_filter'  → Mode B: 1 batch pick + deep scoring for selected
    tfidf_w, kw_w, llm_w:
        Relative weights for TF-IDF, keyword count, and LLM score (auto-normalized)
    """
    if llm_mode == 'batch_filter':
        return _rank_batch_filter(
            papers, keywords, publications,
            llm_api_key=llm_api_key,
            llm_provider=llm_provider, llm_model=llm_model,
            publication_abstracts=publication_abstracts,
            max_llm_candidates=max_llm_candidates,
            llm_input_mode=llm_input_mode,
            tfidf_w=tfidf_w, kw_w=kw_w, llm_w=llm_w,
            chunked_scoring=chunked_scoring,
            batch_filter_ratio=batch_filter_ratio
        )
    else:
        return _rank_individual(
            papers, keywords, publications,
            llm_api_key=llm_api_key,
            llm_provider=llm_provider, llm_model=llm_model,
            publication_abstracts=publication_abstracts,
            max_llm_candidates=max_llm_candidates,
            llm_input_mode=llm_input_mode,
            llm_target=llm_target,
            tfidf_w=tfidf_w, kw_w=kw_w, llm_w=llm_w
        )


# ============================================================
# LLM SCORING (Moonshot/OpenAI compatible)
# ============================================================
LLM_PROVIDERS = {
    'moonshot': {
        'base_url': 'https://api.moonshot.cn/v1',
        'model': 'moonshot-v1-8k',
        'format': 'openai',
    },
    'openai': {
        'base_url': 'https://api.openai.com/v1',
        'model': 'gpt-4o-mini',
        'format': 'openai',
    },
    'anthropic': {
        'base_url': 'https://api.anthropic.com/v1',
        'model': 'claude-haiku-4-5-20251001',  # Active Haiku (3-5 retired Feb 2026)
        'format': 'anthropic',
    },
}

def _extract_content(resp_json, fmt):
    """Extract text content from OpenAI or Anthropic response."""
    if fmt == 'anthropic':
        # Anthropic: {"content": [{"type": "text", "text": "..."}]}
        for block in resp_json.get('content', []):
            if block.get('type') == 'text':
                return block.get('text', '')
        return ''
    else:
        # OpenAI/Moonshot: {"choices": [{"message": {"content": "..."}}]}
        choices = resp_json.get('choices', [])
        if choices and 'message' in choices[0]:
            return choices[0]['message'].get('content', '')
        return ''


def _batch_filter_single_chunk(chunk_papers, chunk_offset, total_count, keywords, publications, api_key, llm_provider='moonshot', model=None, pick_ratio=0.4):
    """Run batch filter on a single chunk of papers. Returns list of 0-based global indices."""
    provider_config = LLM_PROVIDERS.get(llm_provider, LLM_PROVIDERS['moonshot'])
    base_url = provider_config['base_url']
    model = model or provider_config['model']
    fmt = provider_config.get('format', 'openai')
    keyword_summary = ", ".join(keywords[:20])
    pub_summary = "; ".join(publications[:5])

    paper_list_lines = []
    for idx, p in enumerate(chunk_papers, 1):
        cats = ", ".join(p.get('categories', [])[:3])
        paper_list_lines.append(f"{idx}. {p['title']}  [{cats}]")
    paper_list = "\n".join(paper_list_lines)

    target_picks = max(5, int(len(chunk_papers) * pick_ratio))

    prompt = f"""You are an expert academic advisor. A researcher with this profile is looking for relevant arXiv papers:

Keywords: {keyword_summary}
Representative publications: {pub_summary}

This is chunk {chunk_offset // len(chunk_papers) + 1} of { (total_count + len(chunk_papers) - 1) // len(chunk_papers) }. Please identify which papers are POTENTIALLY RELEVANT.

TARGET: Select approximately {target_picks} papers — roughly {int(pick_ratio * 100)}% of this chunk. Err on the side of INCLUSION. If a paper touches on any topic related to the researcher's work — even tangentially — include it.

Respond ONLY with a comma-separated list of paper numbers, like:
3, 7, 12, 15, 23

If NONE seem relevant, respond with the word: none

Papers:
{paper_list}"""

    try:
        if fmt == 'anthropic':
            resp = requests.post(
                f"{base_url}/messages",
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 512, "temperature": 0.3},
                timeout=60
            )
        else:
            resp = requests.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.3, "max_tokens": 512},
                timeout=60
            )

        if resp.status_code != 200:
            return []

        content = _extract_content(resp.json(), fmt).strip()
        if not content or content.lower() in ('none', 'none.', 'none\n'):
            return []

        numbers = re.findall(r'\b(\d+)\b', content)
        indices = []
        for n in numbers:
            idx = int(n) - 1 + chunk_offset
            if chunk_offset <= idx < chunk_offset + len(chunk_papers):
                indices.append(idx)
        return indices
    except Exception:
        return []


def batch_filter_llm(papers, keywords, publications, api_key, llm_provider='moonshot', model=None, max_candidates=100, pick_ratio=0.4):
    """
    v154: Chunked batch filter. Splits papers into batches of ~50 titles,
    runs the filter on each chunk independently, and unions the results.
    This avoids attention decay on long lists (250+ titles) at high pick ratios.
    """
    import logging
    logger = logging.getLogger(__name__)

    candidate_papers = papers[:max_candidates]
    total = len(candidate_papers)

    # For small sets, use a single call
    if total <= 60:
        return _batch_filter_single_chunk(candidate_papers, 0, total, keywords, publications, api_key, llm_provider, model, pick_ratio)

    # Chunk into batches of ~50
    CHUNK_SIZE = 50
    all_picked = []
    for start in range(0, total, CHUNK_SIZE):
        chunk = candidate_papers[start:start + CHUNK_SIZE]
        picked = _batch_filter_single_chunk(chunk, start, total, keywords, publications, api_key, llm_provider, model, pick_ratio)
        all_picked.extend(picked)
        logger.info(f"Batch filter chunk {start // CHUNK_SIZE + 1}: {len(chunk)} papers, {len(picked)} picked")

    # Deduplicate while preserving order
    seen = set()
    unique_indices = []
    for idx in all_picked:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)

    logger.info(f'Batch filter picked {len(unique_indices)} papers from {len(candidate_papers)}')
    return unique_indices

def score_with_llm(papers, keywords, publications, api_key, llm_provider='moonshot', model=None, input_mode='titles_and_abstracts', target='top'):
    """
    Score papers using LLM API (Moonshot/OpenAI/Anthropic compatible).
    Returns dict {'scores': np.array, 'comments': list} or None on failure.
    Also produces a per-paper relevance comment (one_line_reason).
    """
    provider_config = LLM_PROVIDERS.get(llm_provider, LLM_PROVIDERS['moonshot'])
    base_url = provider_config['base_url']
    model = model or provider_config['model']
    fmt = provider_config.get('format', 'openai')

    import logging
    logger = logging.getLogger(__name__)

    scores = []
    comments = []
    first_error = None
    keyword_summary = ", ".join(keywords[:15])
    pub_summary = "; ".join(publications[:5])

    # Target-specific prompt framing
    if target == 'bottom':
        target_hint = ("An algorithmic filter ranked these papers LOW. "
                       "Your job is to find HIDDEN GEMS — papers that are actually relevant "
                       "despite poor keyword overlap. Be generous if the research direction matches.")
    else:
        target_hint = "Rate how relevant this paper is to the researcher."

    for p in papers:  # Score ALL papers passed in (caller controls count via max_llm_candidates)
        if input_mode == 'titles_only':
            paper_block = f"Title: {p['title']}\nCategories: {', '.join(p['categories'][:3])}"
            token_hint = "You only see the title and categories — judge relevance from these alone."
        else:
            paper_block = f"Title: {p['title']}\nCategories: {', '.join(p['categories'][:3])}\nAbstract: {p['abstract'][:800]}"
            token_hint = ""

        prompt = f"""You are an expert academic advisor specializing in quantum physics, chemistry, and quantum computing.

A researcher with this profile is looking for relevant papers:
- Keywords: {keyword_summary}
- Representative publications: {pub_summary}

{target_hint}
{token_hint}

Your task: Read the paper and respond with TWO lines:
1. A relevance score from 0-100 (where 80+ = highly relevant, 50 = neutral, below 30 = not relevant)
2. ONE SENTENCE explaining specifically why this paper is (or is not) relevant to this researcher. Mention techniques, methods, or keyword connections.

Use this exact format:
SCORE: [number]
REASON: [one sentence explanation]

Paper:
{paper_block}"""

        try:
            if fmt == 'anthropic':
                resp = requests.post(
                    f"{base_url}/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 512,
                        "temperature": 0.2,
                    },
                    timeout=30
                )
            else:
                resp = requests.post(
                    f"{base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.2,
                        "max_tokens": 512
                    },
                    timeout=30
                )

            if resp.status_code == 200:
                content = _extract_content(resp.json(), fmt)
                if content:
                    # Parse SCORE: number REASON: text format
                    score_match = re.search(r'SCORE:\s*(\d+)', content, re.I)
                    reason_match = re.search(r'REASON:\s*(.+?)(?:\n|$)', content, re.I | re.DOTALL)
                    
                    if score_match:
                        score_val = int(score_match.group(1))
                        scores.append(min(100, max(0, score_val)) / 100.0)
                    else:
                        # Fallback: look for any number 0-100
                        fb_match = re.search(r'\b(\d{1,2}|100)\b', content)
                        if fb_match:
                            scores.append(int(fb_match.group(1)) / 100.0)
                        else:
                            scores.append(0.5)
                    
                    if reason_match:
                        comment = reason_match.group(1).strip()
                        # Clean up markdown bold
                        comment = re.sub(r'\*\*?|\*\*?', '', comment)
                        comments.append(comment)
                    else:
                        if not first_error:
                            first_error = f'No REASON found in LLM response. Content: {content[:200]}'
                        comments.append('')
                else:
                    if not first_error:
                        first_error = 'LLM returned empty content'
                    scores.append(0.5)
                    comments.append('')
            else:
                err_detail = ''
                try:
                    err_json = resp.json()
                    err_detail = str(err_json)[:300]
                except Exception:
                    err_detail = resp.text[:300]
                if not first_error:
                    first_error = f'{llm_provider} HTTP {resp.status_code}: {err_detail}'
                scores.append(0.5)
                comments.append('')
            time.sleep(0.5)
        except Exception as e:
            if not first_error:
                first_error = f'{type(e).__name__}: {str(e)[:200]}'
            scores.append(0.5)
            comments.append('')

    # Pad if we didn't score all papers
    while len(scores) < len(papers):
        scores.append(0.5)
        comments.append('')

    result = {'scores': np.array(scores[:len(papers)]), 'comments': comments[:len(papers)]}
    if first_error:
        result['error'] = first_error
        logger.warning(f'LLM scoring error: {first_error}')
    return result


def score_with_llm(papers, keywords, publications, api_key, llm_provider='moonshot', model=None, input_mode='titles_and_abstracts', target='top'):
    """
    Score papers using LLM API — BATCHED version.
    Scores ALL papers in a SINGLE API call for efficiency.
    Returns dict {'scores': np.array, 'comments': list} or None on failure.
    """
    if not papers:
        return {'scores': np.array([]), 'comments': []}

    provider_config = LLM_PROVIDERS.get(llm_provider, LLM_PROVIDERS['moonshot'])
    base_url = provider_config['base_url']
    model = model or provider_config['model']
    fmt = provider_config.get('format', 'openai')

    import logging
    logger = logging.getLogger(__name__)

    keyword_summary = ", ".join(keywords[:15])
    pub_summary = "; ".join(publications[:5])

    if target == 'bottom':
        target_hint = ("An algorithmic filter ranked these papers LOW. "
                       "Your job is to find HIDDEN GEMS — papers that are actually relevant "
                       "despite poor keyword overlap. Be generous if the research direction matches.")
    else:
        target_hint = "Rate how relevant each paper is to the researcher."

    # Build numbered paper list
    if input_mode == 'titles_only':
        paper_list = "\n".join([
            f"{i+1}. {p['title']} [Categories: {', '.join(p.get('categories', [])[:3])}]"
            for i, p in enumerate(papers)
        ])
        token_hint = "You only see titles and categories — judge relevance from these alone."
    else:
        paper_list = "\n".join([
            f"{i+1}. {p['title']} [Categories: {', '.join(p.get('categories', [])[:3])}]\n   Abstract: {p.get('abstract', '')[:600]}"
            for i, p in enumerate(papers)
        ])
        token_hint = ""

    prompt = f"""You are an expert academic advisor specializing in quantum physics, chemistry, and quantum computing.

A researcher with this profile is looking for relevant papers:
- Keywords: {keyword_summary}
- Representative publications: {pub_summary}

{target_hint}
{token_hint}

For each numbered paper below, provide:
1. A relevance score from 0-100 (where 80+ = highly relevant, 50 = neutral, below 30 = not relevant)
2. ONE SENTENCE explaining specifically why this paper is (or is not) relevant to this researcher

Respond with ONLY a numbered list in this exact format:
1. SCORE: [number] | REASON: [one sentence explanation]
2. SCORE: [number] | REASON: [one sentence explanation]
...

Papers:
{paper_list}"""

    try:
        if fmt == 'anthropic':
            resp = requests.post(
                f"{base_url}/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": min(4096, len(papers) * 200),
                    "temperature": 0.2,
                },
                timeout=120
            )
        else:
            resp = requests.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": min(4096, len(papers) * 200)
                },
                timeout=120
            )

        scores = [0.5] * len(papers)
        comments = [''] * len(papers)
        first_error = None

        if resp.status_code == 200:
            content = _extract_content(resp.json(), fmt)
            if content:
                # Parse numbered responses: "1. SCORE: 75 | REASON: explanation"
                for i in range(len(papers)):
                    num = i + 1
                    # Try multiple patterns
                    patterns = [
                        rf'^\s*{num}\.\s*SCORE:\s*(\d+)\s*\|\s*REASON:\s*(.+?)(?=\n\s*{num+1}\.|\Z)',
                        rf'\n\s*{num}\.\s*SCORE:\s*(\d+)\s*\|\s*REASON:\s*(.+?)(?=\n\s*{num+1}\.|\Z)',
                        rf'^\s*{num}\)\s*SCORE:\s*(\d+)\s*\|\s*REASON:\s*(.+?)(?=\n\s*{num+1}\)|\Z)',
                        rf'\n\s*{num}\)\s*SCORE:\s*(\d+)\s*\|\s*REASON:\s*(.+?)(?=\n\s*{num+1}\)|\Z)',
                    ]
                    for pat in patterns:
                        m = re.search(pat, content, re.DOTALL | re.MULTILINE)
                        if m:
                            score_val = int(m.group(1))
                            scores[i] = min(100, max(0, score_val)) / 100.0
                            comment = m.group(2).strip()
                            comment = re.sub(r'\*\*?|\*\*?', '', comment)
                            comments[i] = comment
                            break
                    else:
                        # Fallback: look for just the score
                        score_pat = re.search(rf'{num}\.\D*(\d{{1,2}}|100)\b', content)
                        if score_pat:
                            scores[i] = int(score_pat.group(1)) / 100.0
            else:
                first_error = 'LLM returned empty content'
        else:
            try:
                err_detail = str(resp.json())[:300]
            except Exception:
                err_detail = resp.text[:300]
            first_error = f'{llm_provider} HTTP {resp.status_code}: {err_detail}'

        result = {'scores': np.array(scores[:len(papers)]), 'comments': comments[:len(papers)]}
        if first_error:
            result['error'] = first_error
            logger.warning(f'LLM scoring error: {first_error}')
        return result

    except Exception as e:
        logger.warning(f'LLM scoring exception: {type(e).__name__}: {e}')
        return {'scores': np.full(len(papers), 0.5), 'comments': [''] * len(papers), 'error': f'{type(e).__name__}: {str(e)[:200]}'}


# ============================================================
# COMMENT GENERATION
# ============================================================
def generate_comment(paper, mentions, categories):
    """Generate rule-based relevance comment."""
    kw_set = set(mentions.keys())
    cats = ', '.join(categories[:2]) if isinstance(categories, list) else categories
    abstract = paper.get('abstract', '')
    first_sent = abstract.split('.')[0].strip() if abstract else ""
    if len(first_sent) > 180:
        first_sent = first_sent[:180] + "..."

    areas = []
    if kw_set & {"DMET", "DMFT", "embedding theory", "quantum embedding"}: areas.append("quantum embedding")
    if kw_set & {"quantum chemistry", "density functional", "DFT", "VQE"}: areas.append("electronic structure")
    if kw_set & {"open quantum systems", "Non Markovian", "Markovian", "HEOM", "tensor networks", "MPS", "DMRG"}: areas.append("open quantum systems")
    if kw_set & {"quantum algorithm", "quantum simulation", "quantum hardware", "error correction", "error mitigation"}: areas.append("quantum computing")
    if kw_set & {"quantum control", "optimal control", "NMR"}: areas.append("quantum control")
    if kw_set & {"electrochemistry", "electrode", "electrolyte", "Butler-Volmer", "Marcus theory", "fuel cell", "surface"}: areas.append("electrochemistry")
    if kw_set & {"molecular dynamics", "phonon", "vibrational", "vibronic"}: areas.append("molecular dynamics")
    if kw_set & {"polarons", "Holstein", "Peierls", "Born Oppenheimer"}: areas.append("electron-phonon interactions")
    if kw_set & {"qudits", "quantum hardware", "transpilation", "feature map"}: areas.append("quantum hardware")
    if kw_set & {"partial differential equations", "differential equations", "pde"}: areas.append("PDE methods")
    if kw_set & {"spectroscopy", "charge transfer", "electron transfer"}: areas.append("spectroscopy")

    area_str = "; ".join(areas[:3]) if areas else "interdisciplinary"

    if "quantum embedding" in area_str: return f"Advances quantum embedding theory ({cats}). {first_sent}."
    elif "electronic structure" in area_str: return f"Novel electronic structure methodology ({cats}). {first_sent}."
    elif "open quantum systems" in area_str: return f"Develops open quantum system techniques ({cats}). {first_sent}."
    elif "quantum computing" in area_str: return f"Quantum computing algorithm work ({cats}). {first_sent}."
    elif "quantum control" in area_str: return f"Advances quantum/optimal control ({cats}). {first_sent}."
    elif "electrochemistry" in area_str: return f"Electrochemistry-focused study ({cats}). {first_sent}."
    elif "molecular dynamics" in area_str: return f"Molecular/phonon dynamics study ({cats}). {first_sent}."
    elif "electron-phonon" in area_str: return f"Electron-phonon interaction work ({cats}). {first_sent}."
    elif "quantum hardware" in area_str: return f"Quantum hardware implementation ({cats}). {first_sent}."
    elif "PDE" in area_str: return f"PDE-focused quantum algorithm ({cats}). {first_sent}."
    elif "spectroscopy" in area_str: return f"Spectroscopy/transfer phenomena ({cats}). {first_sent}."
    else: return f"Cross-disciplinary work ({cats}) in {area_str}. {first_sent}."


# ============================================================
# FLASK ROUTES
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/run_digest', methods=['POST'])
@limiter.limit("10 per hour")
def run_digest():
    """Run the digest pipeline."""
    data = request.json or {}
    
    # Use stored profile as defaults, allow override from request
    profile = get_profile()
    days_back = min(data.get('days_back', 3), 7)
    # API key: request JSON overrides environment variable
    llm_api_key = data.get('llm_api_key', '').strip() or os.environ.get('MOONSHOT_API_KEY', '').strip() or os.environ.get('OPENAI_API_KEY', '').strip() or os.environ.get('ANTHROPIC_API_KEY', '').strip() or None
    llm_provider = data.get('llm_provider', 'anthropic')
    llm_model = data.get('llm_model', '')
    llm_input_mode = data.get('llm_input_mode', 'titles_and_abstracts')
    llm_target = data.get('llm_target', 'top')
    llm_mode = data.get('llm_mode', 'batch_filter')
    chunked_scoring = data.get('chunked_scoring', True)
    max_llm_candidates = data.get('max_llm_candidates', 9999)
    date_mode = data.get('date_mode', 'submitted')
    include_replacements = data.get('include_replacements', False)
    # Note: "All fetched" (9999) will be resolved after we know len(papers)
    keywords = data.get('keywords', profile['keywords'])
    publications = data.get('publications', profile['publications'])
    pub_abstracts = data.get('publication_abstracts', profile.get('publication_abstracts', []))
    # Category selection from request or profile
    categories = data.get('categories', profile.get('categories', DEFAULT_CATEGORIES))

    # Date range: explicit from/to takes priority over days_back
    date_from = None
    date_to = None
    if 'date_from' in data and data['date_from']:
        try:
            date_from = datetime.strptime(data['date_from'], '%Y-%m-%d')
        except ValueError:
            pass
    if 'date_to' in data and data['date_to']:
        try:
            date_to = datetime.strptime(data['date_to'], '%Y-%m-%d')
        except ValueError:
            pass

    # Fetch papers with date-chunk partitioning
    papers, metadata = fetch_arxiv_papers(
        date_from=date_from, date_to=date_to, days_back=days_back,
        categories=categories, date_mode=date_mode,
        include_replacements=include_replacements
    )
    if isinstance(metadata, str):
        return jsonify({'error': metadata}), 500

    if not papers:
        return jsonify({
            'error': 'No papers found for the specified period',
            'fetch_meta': metadata
        }), 404

    # Resolve "All fetched" now that we know the paper count
    if max_llm_candidates >= 9999:
        max_llm_candidates = len(papers)
    else:
        max_llm_candidates = min(max_llm_candidates, 150)

    # Rank (three-way weighted blend: TF-IDF + Keyword + LLM)
    scored = rank_papers(papers, keywords, publications, 
                         llm_api_key=llm_api_key,
                         llm_provider=llm_provider, llm_model=llm_model if llm_model else None,
                         publication_abstracts=pub_abstracts,
                         max_llm_candidates=max_llm_candidates,
                         llm_input_mode=llm_input_mode,
                         llm_target=llm_target,
                         llm_mode=llm_mode,
                         tfidf_w=data.get('tfidf_weight', 25),
                         kw_w=data.get('kw_weight', 25),
                         llm_w=data.get('llm_weight', 50),
                         chunked_scoring=chunked_scoring,
                         batch_filter_ratio=float(data.get('batch_filter_ratio', 0.8)))

    if not scored['main']:
        return jsonify({
            'error': 'No papers matched your keywords',
            'fetch_meta': metadata
        }), 404

    # Assign tiers from main track
    main = scored['main']
    tiers = {
        'tier1': main[:3],
        'tier2': main[:5],
        'tier3': main[:10],
        'tier4': main[:25],
        'tier5': main,  # ALL remaining papers — never hide anything
        'llm_rescue_pure': scored.get('llm_rescue_pure', []),
        'llm_rescue_weak': scored.get('llm_rescue_weak', []),
    }

    # Extract LLM error from first main paper if present
    llm_error = main[0].get('_llm_error') if main else None
    for s in main:
        s.pop('_llm_error', None)

    stats = {
        'total_scanned': len(papers),
        'total_matched': len(main),
        'llm_rescue_count': len(tiers.get('llm_rescue_pure', [])) + len(tiers.get('llm_rescue_weak', [])),
        'rescue_candidates': scored.get('rescue_candidates_count', 0),
        'date_range': metadata.get('date_range_covered', ''),
        'date_requested': metadata.get('date_range_requested', ''),
        'date_mode': metadata.get('date_mode', 'submitted'),
        'llm_enabled': llm_api_key is not None,
        'llm_error': llm_error,
        'keywords_used': len(keywords),
        'total_available': metadata.get('total_available', 0),
        'limited': metadata.get('limited', False),
        'categories': categories,
        'replacements_filtered': metadata.get('total_filtered', 0),
        'replacements_included': metadata.get('replacements_included', False),
        'blend_weights': {
            'tfidf': data.get('tfidf_weight', 25),
            'kw': data.get('kw_weight', 25),
            'llm': data.get('llm_weight', 50),
        },
        'llm_model': llm_model or 'default',
        'llm_provider': llm_provider,
        'llm_mode': llm_mode,
        'llm_input_mode': llm_input_mode,
        'llm_target': llm_target,
        'max_llm_candidates': max_llm_candidates,
        'chunked_scoring': chunked_scoring,
        'batch_filter_ratio': float(data.get('batch_filter_ratio', 0.4)),
        'include_replacements': include_replacements,
    }

    return jsonify({'tiers': tiers, 'stats': stats})


@app.route('/api/insights', methods=['POST'])
@limiter.limit("15 per hour")
def generate_insights():
    """Post-process digest to generate topic-wise curated insights.
    
    Expects JSON: {
        'papers': [list of all paper dicts],
        'categories': [
            {'name': 'Quantum Chemistry', 'keywords': ['VQE', 'DMET', '...']},
            ...
        ],
        'llm_api_key': '...',
        'llm_provider': 'moonshot',
        'llm_model': '...',
        'min_papers': 12,
        'max_papers': 24,
        'relative_ratio': 0.1,
        'meaningful_threshold': 1.0,
        'per_category_limits': {
            'Electrochemistry': {'min_papers': 4, 'max_papers': 6},
            'Quantum PDE Solvers': {'min_papers': 4, 'max_papers': 6}
        }
    }
    """
    data = request.json or {}
    papers = data.get('papers', [])
    categories = data.get('categories', [])
    llm_api_key = data.get('llm_api_key')
    llm_provider = data.get('llm_provider', 'moonshot')
    llm_model = data.get('llm_model', '')
    custom_prompt = data.get('custom_prompt')
    per_category_prompts = data.get('per_category_prompts', {})
    per_category_summary_prompts = data.get('per_category_summary_prompts', {})
    global_min = max(1, data.get('min_papers', 5))
    global_max = max(global_min, data.get('max_papers', 10))
    relative_ratio = max(0.1, min(1.0, data.get('relative_ratio', 0.5)))
    meaningful_threshold = max(0.5, min(5.0, data.get('meaningful_threshold', 2.0)))
    match_weights = data.get('match_weights', {})
    l1_weight = max(0.1, match_weights.get('l1', 2.0))
    l3_weight = max(0.1, match_weights.get('l3', 1.5))
    l4_weight = max(0.1, match_weights.get('l4', 1.0))
    per_category_limits = data.get('per_category_limits', {})
    
    if not papers or not categories or not llm_api_key:
        return jsonify({'error': 'Missing papers, categories, or LLM API key'}), 400
    
    insights = {}
    errors = []
    diagnostics = []
    url_cat_km = {}  # (url, cat_name) -> keyword match count for dedup
    url_cat_quality = {}  # (url, cat_name) -> quality_score for dedup and force-include
    
    for cat in categories:
        cat_name = cat.get('name', 'Unknown')
        cat_keywords = cat.get('keywords', [])
        penalising_keywords = cat.get('penalising_keywords', [])
        # Per-category min/max overrides (falls back to global)
        cat_limits = per_category_limits.get(cat_name, {})
        min_papers = max(1, cat_limits.get('min_papers', global_min))
        max_papers = max(min_papers, cat_limits.get('max_papers', global_max))
        
        if not cat_keywords:
            insights[cat_name] = {
                'papers': [],
                'explanation': 'No keywords defined for this category.',
                'max_papers': max_papers
            }
            continue
        
        # Pre-filter: find papers matching category keywords
        # Matching strategy (in order of strictness):
        #   1. Full consecutive phrase match — keyword appears as consecutive words in text
        #   2. Sub-phrase match — a suffix sub-phrase of length >= 2 appears consecutively
        #   3. Word-bag match (fallback) — sufficient individual words are present
        # Hyphen normalization: "non-linear" == "nonlinear" before matching.
        # Returns (count, matched_keywords_list)
        def _s(w):
            """Normalize word for singular/plural tolerant matching."""
            return w.rstrip('s') if len(w) > 3 else w
        
        def _h(w):
            """Normalize hyphens: non-linear -> nonlinear."""
            return w.replace('-', '')
        
        # Acronym expansions: if text contains the expansion, it matches the acronym keyword
        ACRONYM_MAP = {
            'pde': ['partial differential equation', 'differential equation'],
            'pdes': ['partial differential equations', 'differential equations'],
            'ode': ['ordinary differential equation'],
            'odes': ['ordinary differential equations'],
            'vqa': ['variational quantum algorithm', 'variational quantum algorithms'],
            'vqe': ['variational quantum eigensolver'],
            'dmrg': ['density matrix renormalization group'],
            'dft': ['density functional theory'],
        }
        
        def find_flexible_matches(text, keywords, l1_w=3.0, l3_w=1.0, l4_w=0.5):
            """Match keywords against text with 4-level cascade.
            
            Returns: (total_count, matched_keyword_list, quality_score)
            
            quality_score is a weighted sum where:
              L1 (full consecutive phrase): l1_w pts (default 3.0)
              L3 (acronym expansion):       l3_w pts (default 1.0)
              L4 (word-bag fallback):       l4_w pts (default 0.5)
            
            L2 sub-phrase matching REMOVED — it caused phantom matches
            like "time evolution" matching "imaginary time evolution".
            Only L1 (full phrase) can trigger force-include (qs >= 2.0).
            """
            total = 0
            matched_kws = []
            quality_score = 0.0
            
            text_lower = normalize_umlauts(text.lower())
            
            # Build consecutive n-gram sets from the text (normalized)
            # CRITICAL FIX: also extract sub-parts from hyphenated words
            # so "SA-CASSCF" yields both "sacasscf" AND "casscf".
            text_words_raw = re.findall(r"\b[a-z]+(?:-[a-z]+)*\b", text_lower)
            text_subwords = []
            for w in text_words_raw:
                if '-' in w and len(w) > 3:
                    # Split compound hyphenated words into sub-parts
                    parts = w.split('-')
                    # Only add parts that are 3+ chars (skip "SA", "x", etc.)
                    text_subwords.extend(p for p in parts if len(p) >= 3)
            text_words_all = text_words_raw + text_subwords
            text_words_nh = [_h(_s(w)) for w in text_words_all]
            
            text_bigrams = set()
            text_trigrams = set()
            text_fourgrams = set()
            for i in range(len(text_words_nh) - 1):
                text_bigrams.add(text_words_nh[i] + ' ' + text_words_nh[i+1])
            for i in range(len(text_words_nh) - 2):
                text_trigrams.add(text_words_nh[i] + ' ' + text_words_nh[i+1] + ' ' + text_words_nh[i+2])
            for i in range(len(text_words_nh) - 3):
                text_fourgrams.add(text_words_nh[i] + ' ' + text_words_nh[i+1] + ' ' + text_words_nh[i+2] + ' ' + text_words_nh[i+3])
            
            # Also build word-bag for fallback
            text_word_bag = set(text_words_nh)
            
            for kw in keywords:
                kw_lower = normalize_umlauts(kw.lower())
                kw_words = kw_lower.split()
                kw_words_nh = [_h(_s(w)) for w in kw_words]
                match_level = None  # tracks highest level achieved
                
                # --- Level 1: Full consecutive phrase match ---
                # The keyword must appear as consecutive words in the text.
                # This is the ONLY way to get quality_score >= 2.0 (force-include).
                if len(kw_words_nh) >= 2:
                    kw_phrase = ' '.join(kw_words_nh)
                    if len(kw_words_nh) == 2 and kw_phrase in text_bigrams:
                        match_level = 'L1'
                    elif len(kw_words_nh) == 3 and kw_phrase in text_trigrams:
                        match_level = 'L1'
                    elif len(kw_words_nh) == 4 and kw_phrase in text_fourgrams:
                        match_level = 'L1'
                    elif len(kw_words_nh) >= 5:
                        kw_fourgrams = [' '.join(kw_words_nh[i:i+4]) for i in range(len(kw_words_nh)-3)]
                        if any(k4g in text_fourgrams for k4g in kw_fourgrams):
                            match_level = 'L1'
                
                # --- Level 3: Acronym expansion ---
                # Keyword is an acronym (e.g. "PDE"); text contains the expansion
                # ("partial differential equations").  Safe — requires full phrase.
                if match_level is None:
                    for acr, expansions in ACRONYM_MAP.items():
                        if _h(_s(kw_lower)) == acr:
                            for exp in expansions:
                                exp_words = exp.split()
                                matched = False
                                if len(exp_words) == 1:
                                    if _h(_s(exp_words[0])) in text_word_bag:
                                        matched = True
                                elif len(exp_words) == 2:
                                    if _h(_s(exp_words[0])) + ' ' + _h(_s(exp_words[1])) in text_bigrams:
                                        matched = True
                                elif len(exp_words) == 3:
                                    if _h(_s(exp_words[0])) + ' ' + _h(_s(exp_words[1])) + ' ' + _h(_s(exp_words[2])) in text_trigrams:
                                        matched = True
                                if matched:
                                    match_level = 'L3'
                                    break
                        if match_level:
                            break
                
                # --- Level 4: Word-bag fallback ---
                # All individual words must be present (no partial credit for
                # sub-phrases — that was the source of phantom matches like
                # "time evolution" matching "imaginary time evolution").
                if match_level is None:
                    if len(kw_words_nh) == 1:
                        if kw_words_nh[0] in text_word_bag:
                            match_level = 'L4'
                    elif len(kw_words_nh) == 2:
                        if all(w in text_word_bag for w in kw_words_nh):
                            match_level = 'L4'
                    else:
                        word_hits = sum(1 for w in kw_words_nh if w in text_word_bag)
                        threshold = len(kw_words_nh) if len(kw_words_nh) <= 3 else len(kw_words_nh) - 1
                        if word_hits >= threshold:
                            match_level = 'L4'
                
                if match_level:
                    total += 1
                    matched_kws.append(kw)
                    if match_level == 'L1':
                        quality_score += l1_w
                    elif match_level == 'L3':
                        quality_score += l3_w
                    else:  # L4
                        quality_score += l4_w
            
            return total, matched_kws, quality_score
        
        # Pre-compile penalising keyword patterns (full-phrase L1 match only)
        penalising_patterns = []
        if penalising_keywords:
            for pk in penalising_keywords:
                pk_norm = normalize_umlauts(pk.lower())
                pk_dehyphen = pk_norm.replace('-', ' ')
                penalising_patterns.append((pk_norm, pk_dehyphen))
        
        scored = []
        paper_matched_keywords = {}  # url -> list of matched keywords for this category
        for p in papers:
            text = normalize_umlauts(f"{p.get('title','')} {p.get('abstract','')}".lower())
            match_count, matched_kws, quality_score = find_flexible_matches(text, cat_keywords, l1_weight, l3_weight, l4_weight)
            
            # Apply penalising keyword penalties (full-phrase match only, L1 level)
            penalty_applied = False
            if penalising_patterns and match_count > 0:
                for pk_norm, pk_dehyphen in penalising_patterns:
                    if pk_norm in text or pk_dehyphen in text:
                        quality_score -= 10.0  # large penalty to drop below threshold
                        penalty_applied = True
            
            if match_count > 0 and quality_score > 0:
                scored.append((p, match_count, p.get('score', 0), 'keyword'))
                paper_matched_keywords[p.get('url', '')] = matched_kws
                url_cat_km[(p.get('url', ''), cat_name)] = match_count
                url_cat_quality[(p.get('url', ''), cat_name)] = quality_score
        
        # Sort by keyword matches then by digest score
        scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
        # Give LLM a generous pool (2x max) but at least 15 candidates
        candidate_pool_size = max(max_papers * 2, 15)
        candidates = [p for p, _, _, _ in scored[:candidate_pool_size]]
        
        # === CANDIDATE POOL GUARANTEE ===
        # Papers with quality_score >= 2.0 (L1 full-phrase matches) MUST be
        # in the candidate pool.  The pool size cap can exclude keyword-strong
        # papers if many others have higher match counts — this ensures they
        # are never dropped before LLM curation.
        candidate_urls = {p.get('url', '') for p in candidates}
        for p, mc, sc, _ in scored[candidate_pool_size:]:
            url = p.get('url', '')
            if url not in candidate_urls:
                qs = url_cat_quality.get((url, cat_name), 0.0)
                if qs >= 2.0:
                    candidates.append(p)
                    candidate_urls.add(url)
        
        if not candidates:
            insights[cat_name] = {
                'papers': [],
                'explanation': f'No papers matched keywords: {", ".join(cat_keywords[:5])}',
                'max_papers': max_papers
            }
            continue
        
        # LLM curation: pick min-max best and explain
        paper_list = "\n".join([
            f"{i+1}. {p['title']} (arXiv: {p.get('arxiv_id', p['url'].split('/')[-1])}, Score: {p.get('score',0):.3f})\n   Abstract: {p.get('abstract','')[:300]}..."
            for i, p in enumerate(candidates)
        ])
        
        # Build the prompt — uses arxiv IDs for unambiguous matching
        # v133: per-category expert prompts guide the LLM as a domain expert
        expert_guidance = per_category_prompts.get(cat_name) or DEFAULT_CATEGORY_PROMPTS.get(cat_name, "")
        
        # v168: LLM scores ALL candidates (selected + extended) with 0-10 relevance score
        default_prompt_text = f"""{expert_guidance}

You are evaluating {len(candidates)} papers for relevance to "{cat_name}".

For EACH paper, assign a relevance score from 0–10:
  10 = Breakthrough / essential reading for this topic
  7–9 = Highly relevant and important
  4–6 = Moderately relevant
  1–3 = Weakly relevant
  0 = Not relevant

For papers with score >= 5, also write 1–2 sentences explaining WHY it matters.

Respond with a numbered list covering ALL papers in this exact format:
1. [score]/10 arxiv:XXXX.XXXXX — [explanation or "not selected"]
2. [score]/10 arxiv:XXXX.XXXXX — [explanation or "not selected"]
...

Papers:
{paper_list}"""
        
        # v138: per-category prompts now take priority over global custom prompt
        per_cat_prompt = per_category_prompts.get(cat_name, "")
        if per_cat_prompt:
            # User-provided per-category prompt replaces the expert guidance within
            # the full prompt template — it is NOT a standalone template
            prompt = f"""{per_cat_prompt}

You are evaluating {len(candidates)} papers for relevance to "{cat_name}".

For EACH paper, assign a relevance score from 0–10:
  10 = Breakthrough / essential reading for this topic
  7–9 = Highly relevant and important
  4–6 = Moderately relevant
  1–3 = Weakly relevant
  0 = Not relevant

For papers with score >= 5, also write 1–2 sentences explaining WHY it matters.

Respond with a numbered list covering ALL papers in this exact format:
1. [score]/10 arxiv:XXXX.XXXXX — [explanation or "not selected"]
2. [score]/10 arxiv:XXXX.XXXXX — [explanation or "not selected"]
...

Papers:
{paper_list}"""
        elif custom_prompt:
            # Global custom_prompt (backward compat) — only if no per-category prompt
            try:
                prompt = custom_prompt.format(
                    cat_name=cat_name,
                    paper_list=paper_list,
                    min_papers=min_papers,
                    max_papers=max_papers
                )
            except (KeyError, ValueError) as e:
                errors.append(f"{cat_name}: Invalid prompt template — {str(e)}")
                prompt = default_prompt_text
        else:
            prompt = default_prompt_text
        
        provider_config = LLM_PROVIDERS.get(llm_provider, LLM_PROVIDERS['moonshot'])
        base_url = provider_config['base_url']
        model = llm_model or provider_config['model']
        fmt = provider_config.get('format', 'openai')
        
        try:
            if fmt == 'anthropic':
                resp = requests.post(
                    f"{base_url}/messages",
                    headers={
                        "x-api-key": llm_api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 2048,
                        "temperature": 0.3,
                    },
                    timeout=60
                )
            else:
                resp = requests.post(
                    f"{base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {llm_api_key}", "Content-Type": "application/json"},
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3,
                        "max_tokens": 2048
                    },
                    timeout=60
                )
            
            curated = []
            llm_selected_count = 0
            fallback_added = 0
            status = 'unknown'
            raw_response = ''
            
            def _parse_llm_response(text, candidate_list):
                """Parse LLM curation response. PRIMARY: match by arxiv ID.
                FALLBACK: index-based parsing (1-based, with 0-based auto-detect).
                
                v168: Extracts LLM-assigned score (0-10) for each candidate.
                Returns list of (idx, score, explanation).
                
                Using arxiv IDs eliminates ALL ambiguity from index-based parsing
                (0-based vs 1-based confusion, candidate list reordering, etc.).
                """
                # Build lookup maps
                arxiv_map = {}   # arxiv_id → candidate index
                url_map = {}     # url → candidate index
                title_map = {}   # title prefix → candidate index
                for i, c in enumerate(candidate_list):
                    # arxiv ID from URL: https://arxiv.org/abs/2401.12345 → 2401.12345
                    url = c.get('url', '')
                    aid = url.split('/')[-1] if url else ''
                    if aid:
                        arxiv_map[aid] = i
                    if url:
                        url_map[url] = i
                    t = c.get('title', '').lower()
                    if t:
                        title_map[t[:50]] = i
                
                parsed = []
                seen_indices = set()
                seen_urls = set()
                lines = text.replace('\r', '\n').split('\n')
                
                def _extract_score(line_text):
                    """Extract a 0-10 score from the line text."""
                    # Try various score formats: [8]/10, [8.5]/10, 8/10, [8], [8.5]
                    patterns = [
                        r'\[(\d+(?:\.\d+)?)\]\s*/\s*10',  # [8]/10 or [8.5]/10
                        r'\b(\d+(?:\.\d+)?)\s*/\s*10',      # 8/10 or 8.5/10
                        r'\[(\d+(?:\.\d+)?)\]',            # [8] or [8.5]
                    ]
                    for pat in patterns:
                        m = re.search(pat, line_text)
                        if m:
                            val = float(m.group(1))
                            if 0 <= val <= 10:
                                return val
                    return None
                
                def _extract_explanation(line_text, aid):
                    """Extract explanation text after the arxiv ID."""
                    em = re.search(r'[\u2014\u2013\u002d]\s*(.+)', line_text)
                    if em:
                        explanation = em.group(1).strip()
                    else:
                        pos = line_text.lower().find(aid) + len(aid)
                        explanation = line_text[pos:].strip()
                        explanation = re.sub(r'^[\s\-–—]+', '', explanation)
                    # Remove score artifacts and clean up
                    explanation = re.sub(r'\[\d+(?:\.\d+)?\]\s*/?\s*10?', '', explanation).strip()
                    explanation = re.sub(r'^\s*[\-–—]\s*', '', explanation).strip()
                    explanation = re.sub(r'\s*\[\d+\]\s*$', '', explanation).strip()
                    return explanation
                
                # --- Phase 1: arxiv-ID matching (primary) ---
                for line in lines:
                    line_stripped = line.strip()
                    if not line_stripped:
                        continue
                    # Extract arxiv ID: arxiv:2401.12345 or 2401.12345
                    m = re.search(r'arxiv[:\s]*(\d{4}\.\d{4,5})', line_stripped, re.IGNORECASE)
                    if not m:
                        m = re.search(r'\b(\d{4}\.\d{4,5})\b', line_stripped)
                    if not m:
                        continue
                    aid = m.group(1)
                    idx = arxiv_map.get(aid)
                    if idx is None:
                        continue
                    
                    score = _extract_score(line_stripped)
                    explanation = _extract_explanation(line_stripped, aid)
                    # If explanation is "not selected" or similar, treat as no explanation
                    if explanation.lower() in ('not selected', 'not selected.', '-'):
                        explanation = ''
                    
                    url = candidate_list[idx].get('url', '')
                    if idx not in seen_indices and url not in seen_urls:
                        seen_indices.add(idx)
                        seen_urls.add(url)
                        # v168: include score even if explanation is empty
                        parsed.append((idx, score, explanation))
                
                # --- Phase 2: index-based fallback (for old-format responses without arxiv IDs) ---
                # Auto-detect 0-based vs 1-based
                raw_brackets = re.findall(r'\[(\d+)\]', text)
                bracket_nums = [int(x) for x in raw_brackets]
                zero_based = (0 in bracket_nums)
                
                for line in lines:
                    line_stripped = line.strip()
                    if not line_stripped:
                        continue
                    clean = re.sub(r'[*`#>_]', '', line_stripped).strip()
                    if not clean:
                        continue
                    if not (clean[0].isdigit() or clean[0] == '['):
                        continue
                    # Skip lines that already matched by arxiv ID
                    aids_in_line = re.findall(r'\b\d{4}\.\d{4,5}\b', clean)
                    if any(a in arxiv_map and arxiv_map[a] in seen_indices for a in aids_in_line):
                        continue
                    
                    idx = None
                    explanation = ''
                    score = _extract_score(clean)
                    
                    # Format A: "N. [M] — explanation"
                    m = re.search(r'^\d+[.\)]\s*\[(\d+)\]', clean)
                    if m:
                        bracket_val = int(m.group(1))
                        idx = bracket_val if zero_based else bracket_val - 1
                        em = re.search(r'[\u2014\u2013\u002d]\s*(.+)', clean)
                        if em:
                            explanation = em.group(1).strip()
                    
                    # Format B: "[M] — explanation"
                    if idx is None:
                        m = re.search(r'^\[(\d+)\]\s*[\u2014\u2013\u002d]?\s*(.+)', clean)
                        if m:
                            bracket_val = int(m.group(1))
                            idx = bracket_val if zero_based else bracket_val - 1
                            explanation = m.group(2).strip()
                    
                    # Format C: "N. M — explanation"
                    if idx is None:
                        m = re.search(r'^\d+[.\)]\s*(\d+)', clean)
                        if m:
                            idx = int(m.group(1)) - 1
                            em = re.search(r'[\u2014\u2013\u002d]\s*(.+)', clean)
                            if em:
                                explanation = em.group(1).strip()
                    
                    # Format D: "M. explanation"
                    if idx is None:
                        m = re.search(r'^(\d+)[.\)]\s+(.+)', clean)
                        if m:
                            idx = int(m.group(1)) - 1
                            explanation = m.group(2).strip()
                    
                    # Format E: match by title substring
                    if idx is None:
                        cl = clean.lower()
                        for tk, ci in title_map.items():
                            if ci not in seen_indices and tk in cl:
                                idx = ci
                                explanation = clean
                                break
                    
                    if idx is not None and 0 <= idx < len(candidate_list):
                        url = candidate_list[idx].get('url', '')
                        if idx not in seen_indices and url not in seen_urls:
                            seen_indices.add(idx)
                            seen_urls.add(url)
                            explanation = re.sub(r'\s*\[\d+\]\s*$', '', explanation).strip()
                            explanation = re.sub(r'\[\d+(?:\.\d+)?\]\s*/?\s*10?', '', explanation).strip()
                            explanation = re.sub(r'^\s*[\-–—]\s*', '', explanation).strip()
                            if explanation.lower() in ('not selected', 'not selected.', '-'):
                                explanation = ''
                            # v168: include even without explanation (for scoring extended candidates)
                            parsed.append((idx, score, explanation))
                
                return parsed
            
            if resp.status_code == 200:
                resp_json = resp.json()
                
                # Check for API-level errors in the response body (Anthropic sometimes returns 200 with errors)
                if 'error' in resp_json:
                    status = f"llm_api_error: {resp_json['error'].get('message', str(resp_json['error']))[:80]}"
                    errors.append(f"{cat_name}: {status}")
                    raw_response = json.dumps(resp_json)[:500]
                else:
                    content = _extract_content(resp_json, fmt)
                    raw_response = content[:500]
                    
                    if not content or not content.strip():
                        status = 'llm_empty_response'
                        errors.append(f"{cat_name}: LLM returned empty content — check model name and API key")
                    else:
                        status = 'llm_ok'
                        # v168: Parse LLM response — now returns (idx, score, explanation)
                        parsed = _parse_llm_response(content, candidates)
                        
                        # Build maps: idx -> (llm_score, explanation)
                        llm_score_map = {}
                        llm_explanation_map = {}
                        for idx, score, explanation in parsed:
                            llm_score_map[idx] = score
                            llm_explanation_map[idx] = explanation
                        
                        # v168: ALL candidates get llm_expert_score (from LLM or fallback to keyword-based quality_score)
                        for i, c in enumerate(candidates):
                            c['llm_expert_score'] = llm_score_map.get(i, url_cat_quality.get((c.get('url',''), cat_name), 0.0))
                        
                        # Build curated list from papers with explanations (LLM-selected or force-included)
                        for idx, score, explanation in parsed:
                            if not explanation:
                                continue
                            paper = dict(candidates[idx])
                            paper['llm_expert_score'] = score if score is not None else candidates[idx].get('llm_expert_score', 0)
                            paper['insight_explanation'] = explanation
                            matched = paper_matched_keywords.get(paper.get('url', ''), [])
                            paper['matched_keywords'] = matched
                            curated.append(paper)
                        
                        # Force-include: candidates with quality_score >= 2.0 that the LLM missed
                        already_selected_urls = {p['url'] for p in curated}
                        force_added = 0
                        for i, p in enumerate(candidates):
                            url = p.get('url', '')
                            if url not in already_selected_urls:
                                qs = url_cat_quality.get((url, cat_name), 0.0)
                                if qs >= 2.0:
                                    paper = dict(p)
                                    paper['llm_expert_score'] = p.get('llm_expert_score', qs)
                                    paper['insight_explanation'] = f"Strong keyword match for {cat_name} — high relevance."
                                    matched = paper_matched_keywords.get(url, [])
                                    paper['matched_keywords'] = matched
                                    curated.append(paper)
                                    force_added += 1
                        
                        # v168: sort by llm_expert_score descending, then quality_score, then digest_score
                        curated.sort(key=lambda p: (
                            p.get('llm_expert_score', 0),
                            url_cat_quality.get((p.get('url',''), cat_name), 0.0),
                            p.get('score', 0)
                        ), reverse=True)
                        curated = curated[:max_papers]
                        llm_selected_count = sum(1 for p in curated if not p.get('insight_explanation','').startswith('Strong keyword match'))
                        
                        if llm_selected_count == 0 and force_added == 0:
                            status = 'llm_ok_parser_failed'
                            errors.append(f"{cat_name}: LLM responded but 0 papers matched parser — check 'Show LLM raw response'")
                
                # If still fewer than min_papers, fill with top keyword-matched candidates
                fallback_available = 0
                if len(curated) < min_papers:
                    already_selected = {p['url'] for p in curated}
                    fallback_available = sum(1 for p in candidates if p['url'] not in already_selected)
                    for p in candidates:
                        if p['url'] not in already_selected:
                            paper = dict(p)
                            paper['insight_explanation'] = f"High keyword relevance for {cat_name}."
                            matched = paper_matched_keywords.get(paper.get('url', ''), [])
                            paper['matched_keywords'] = matched
                            curated.append(paper)
                            if len(curated) >= min_papers:
                                break
                fallback_added = len(curated) - llm_selected_count
            else:
                status = f'api_error_{resp.status_code}'
                errors.append(f"{cat_name}: API error {resp.status_code}")
                # Fallback: take top min_papers candidates (capped by available)
                fill_count = min(min_papers, len(candidates))
                for i, p in enumerate(candidates[:fill_count]):
                    paper = dict(p)
                    # v168: fallback llm_expert_score = keyword-based quality_score
                    paper['llm_expert_score'] = url_cat_quality.get((p.get('url',''), cat_name), 0.0)
                    paper['insight_explanation'] = f"High keyword relevance for {cat_name}."
                    matched = paper_matched_keywords.get(paper.get('url', ''), [])
                    paper['matched_keywords'] = matched
                    curated.append(paper)
            
            # v168: Build extended candidates list — all candidates not in the
            # final curated list. ALL candidates now include llm_expert_score.
            curated_urls = {p.get('url', '') for p in curated}
            extended = []
            for p in candidates:
                url = p.get('url', '')
                if url not in curated_urls:
                    qs = url_cat_quality.get((url, cat_name), 0.0)
                    km = url_cat_km.get((url, cat_name), 0)
                    matched = paper_matched_keywords.get(url, [])
                    extended.append({
                        'title': p.get('title', 'Untitled'),
                        'url': url,
                        'arxiv_id': p.get('arxiv_id', url.split('/')[-1] if url else ''),
                        'llm_expert_score': round(p.get('llm_expert_score', qs), 2),
                        'quality_score': round(qs, 2),
                        'digest_score': round(p.get('score', 0), 3),
                        'keyword_matches': km,
                        'matched_keywords': matched,
                    })
            # v168: sort extended by llm_expert_score descending, then quality_score, then digest_score
            extended.sort(key=lambda x: (x['llm_expert_score'], x['quality_score'], x['digest_score']), reverse=True)
            
            # Add quality_score and llm_expert_score to curated papers too
            for p in curated:
                url = p.get('url', '')
                p['quality_score'] = round(url_cat_quality.get((url, cat_name), 0.0), 2)
                if 'llm_expert_score' not in p:
                    p['llm_expert_score'] = round(url_cat_quality.get((url, cat_name), 0.0), 2)
            
            insights[cat_name] = {
                'papers': curated,
                'urls': [p.get('url', '') for p in curated],  # v148: URL list for frontend filtering
                'extended_candidates': extended,
                'max_papers': max_papers,  # v169: for dynamic promotion from extended
                'explanation': f"Curated {len(curated)} papers from {len(candidates)} candidates matching {cat_name}."
            }
            diagnostics.append({
                'category': cat_name,
                'candidates': len(candidates),
                'llm_selected': llm_selected_count,
                'force_added': force_added,
                'fallback_added': fallback_added,
                'fallback_available': fallback_available,
                'final_count': len(curated),
                'status': status,
                'min_papers': min_papers,
                'max_papers': max_papers,
                'penalising_keywords': penalising_keywords,
                'raw_response_preview': raw_response
            })
            
        except Exception as e:
            errors.append(f"{cat_name}: {str(e)}")
            fill_count = min(min_papers, len(candidates))
            fallback_papers = []
            for p in candidates[:fill_count]:
                paper = dict(p)
                paper['llm_expert_score'] = url_cat_quality.get((p.get('url',''), cat_name), 0.0)
                paper['quality_score'] = round(url_cat_quality.get((p.get('url',''), cat_name), 0.0), 2)
                fallback_papers.append(paper)
            insights[cat_name] = {
                'papers': fallback_papers,
                'urls': [p.get('url', '') for p in fallback_papers],  # v148
                'max_papers': max_papers,  # v169
                'explanation': f"Error during curation. Showing top {fill_count} keyword matches."
            }
            diagnostics.append({
                'category': cat_name,
                'candidates': len(candidates),
                'llm_selected': 0,
                'fallback_added': fill_count,
                'final_count': fill_count,
                'status': f'exception: {str(e)[:50]}',
                'min_papers': min_papers,
                'max_papers': max_papers
            })
    
    # Cross-category deduplication — v120:
    # A paper stays in a category ONLY if it has meaningful keyword
    # evidence there.  Two criteria:
    #   (1) Absolute: quality_score >= meaningful_threshold in that category
    #   (2) Relative: qs in that category must be at least ratio × the BEST
    #       category's qs.  Prevents noise matches (e.g. "Nonlinear" qs=0.5)
    #       from keeping a paper in a category where it doesn't belong,
    #       when another category has qs=6.0.
    # Cross-disciplinary papers (kept in 2+ categories) are flagged for UI.
    from collections import defaultdict
    
    MEANINGFUL_THRESHOLD = meaningful_threshold  # absolute threshold (configurable)
    RELATIVE_RATIO = relative_ratio      # configurable (default 0.5)
    
    url_to_cats = defaultdict(list)
    for cat_name, cat_data in insights.items():
        for p in cat_data.get('papers', []):
            url = p.get('url', '')
            km = url_cat_km.get((url, cat_name), 0)
            qs = url_cat_quality.get((url, cat_name), 0.0)
            score = p.get('score', 0)
            url_to_cats[url].append((cat_name, km, qs, score))
    
    url_keep_cats = {}
    url_is_cross_disciplinary = set()
    for url, cat_list in url_to_cats.items():
        if len(cat_list) == 1:
            url_keep_cats[url] = {cat_list[0][0]}
            continue
        
        # Find the best category's quality score
        best_qs = max(c[2] for c in cat_list)
        
        # Keep categories that pass BOTH absolute AND relative thresholds
        kept = []
        for c in cat_list:
            cat_name, km, qs, score = c
            abs_ok = qs >= MEANINGFUL_THRESHOLD
            rel_ok = qs >= best_qs * RELATIVE_RATIO  # e.g. ratio=0.5: qs >= best_qs * 0.5
            if abs_ok and rel_ok:
                kept.append(c)
        
        if len(kept) >= 2:
            url_keep_cats[url] = {c[0] for c in kept}
            url_is_cross_disciplinary.add(url)
        elif len(kept) == 1:
            url_keep_cats[url] = {kept[0][0]}
        else:
            # Nothing passed both thresholds — keep the single best category
            best = max(cat_list, key=lambda c: (c[2], c[1], c[3]))
            url_keep_cats[url] = {best[0]}
    
    # Apply dedup filtering + cross-disciplinary flagging
    removed_counts = {}
    for cat_name, cat_data in insights.items():
        filtered = []
        removed = 0
        for p in cat_data.get('papers', []):
            url = p.get('url', '')
            if cat_name in url_keep_cats.get(url, {cat_name}):
                # Flag cross-disciplinary papers
                kept_cats = url_keep_cats.get(url, set())
                if len(kept_cats) >= 2 and url in url_is_cross_disciplinary:
                    p['_cross_disciplinary'] = True
                    p['_appears_in'] = sorted(kept_cats)
                filtered.append(p)
            else:
                removed += 1
        if removed > 0:
            removed_counts[cat_name] = removed
            cat_data['papers'] = filtered
    
    # v146: Generate per-category LLM expert summaries
    # Summaries provide qualitative context about what themes/trends the LLM
    # noticed in each category's curated papers — highlighting variability
    # across days and noting notable absences or shifts.
    diagnostics_dict = {d['category']: d for d in diagnostics}
    for cat_name, cat_data in insights.items():
        curated = cat_data.get('papers', [])
        if not curated:
            cat_data['summary'] = 'No papers were curated for this category today.'
            continue
        # Only summarize if at least one paper was LLM-selected (not pure fallback)
        llm_selected_count = diagnostics_dict.get(cat_name, {}).get('llm_selected', 0)
        if llm_selected_count == 0:
            cat_data['summary'] = 'All papers were selected by keyword matching alone — the LLM did not find any strong matches today.'
            continue
        # v147: Use per-category summary prompt if provided, else fall back to default
        summary_guidance = per_category_summary_prompts.get(cat_name, '')
        if not summary_guidance:
            # Fall back to inline summary_prompt from category definition
            for cat_def in DEFAULT_INSIGHT_CATEGORIES:
                if cat_def['name'] == cat_name:
                    summary_guidance = cat_def.get('summary_prompt', '')
                    break
        summary = _generate_category_summary(
            cat_name, curated, summary_guidance,
            llm_api_key, llm_provider, llm_model
        )
        cat_data['summary'] = summary

    return jsonify({
        'insights': insights,
        'errors': errors,
        'diagnostics': diagnostics,
        'dedup_removed': removed_counts,
        'settings': {
            'relative_ratio': relative_ratio,
            'meaningful_threshold': meaningful_threshold,
            'match_weights': {'l1': l1_weight, 'l3': l3_weight, 'l4': l4_weight},
            'per_category_limits': per_category_limits,
        }
    })


@app.route('/health')
def health():
    """Health check endpoint for deployment monitoring."""
    return jsonify({
        'status': 'ok',
        'version': 'v137',
        'timestamp': datetime.now().isoformat(),
        'features': ['tfidf', 'llm_batch', 'insights', 'category_filter']
    })


@app.route('/api/profile', methods=['GET', 'POST'])
def profile_endpoint():
    """Get or update the user profile."""
    if request.method == 'GET':
        return jsonify(get_profile())
    else:
        data = request.json or {}
        update_profile(data)
        return jsonify({'status': 'ok', 'profile': get_profile()})


@app.route('/api/default_profile', methods=['GET'])
def default_profile():
    # Build category_keywords as union of all insight category keywords
    category_keywords = []
    for cat in DEFAULT_INSIGHT_CATEGORIES:
        category_keywords.extend(cat.get("keywords", []))
    return jsonify({
        'keywords': DEFAULT_KEYWORDS,
        'publications': DEFAULT_PUBLICATIONS,
        'categories': DEFAULT_CATEGORIES,
        'insight_categories': DEFAULT_INSIGHT_CATEGORIES,
        'cross_cutting': DEFAULT_CROSS_CUTTING,
        'category_keywords': category_keywords,
    })


# Per-category min/max limits (optional overrides to global defaults)
_PER_CATEGORY_LIMITS = {}
# Per-category LLM prompts (user overrides to default expert prompts)
_PER_CATEGORY_PROMPTS = {}
# Per-category summary prompts (user overrides to default summary guidance) — v147
_PER_CATEGORY_SUMMARY_PROMPTS = {}

@app.route('/api/insight_categories', methods=['GET', 'POST'])
def insight_categories_endpoint():
    """Get or update the insight category definitions.
    
    GET: Returns the current insight categories (default or user-modified).
    POST: Accepts new category definitions, rebuilds DEFAULT_KEYWORDS from the union.
    """
    global DEFAULT_KEYWORDS, _PER_CATEGORY_LIMITS, _PER_CATEGORY_PROMPTS, _PER_CATEGORY_SUMMARY_PROMPTS
    if request.method == 'GET':
        return jsonify({
            'insight_categories': DEFAULT_INSIGHT_CATEGORIES,
            'cross_cutting': DEFAULT_CROSS_CUTTING,
            'derived_keywords': DEFAULT_KEYWORDS,
            'derived_count': len(DEFAULT_KEYWORDS),
            'per_category_limits': _PER_CATEGORY_LIMITS,
            'per_category_prompts': _PER_CATEGORY_PROMPTS,
            'per_category_summary_prompts': _PER_CATEGORY_SUMMARY_PROMPTS,
        })
    else:
        data = request.json or {}
        new_categories = data.get('insight_categories')
        new_cross = data.get('cross_cutting')
        new_limits = data.get('per_category_limits')
        new_prompts = data.get('per_category_prompts')
        new_summary_prompts = data.get('per_category_summary_prompts')
        if new_categories:
            # Validate structure
            valid = all('name' in c and 'keywords' in c for c in new_categories)
            if not valid:
                return jsonify({'error': 'Each category must have "name" and "keywords"'}), 400
            DEFAULT_INSIGHT_CATEGORIES.clear()
            DEFAULT_INSIGHT_CATEGORIES.extend(new_categories)
        if new_cross:
            DEFAULT_CROSS_CUTTING.clear()
            DEFAULT_CROSS_CUTTING.extend(new_cross)
        if new_limits is not None:
            _PER_CATEGORY_LIMITS = new_limits
        if new_prompts is not None:
            _PER_CATEGORY_PROMPTS = new_prompts
        if new_summary_prompts is not None:
            _PER_CATEGORY_SUMMARY_PROMPTS = new_summary_prompts
        # Rebuild DEFAULT_KEYWORDS from the new categories
        DEFAULT_KEYWORDS = build_default_keywords()
        # Also update the user's profile keywords
        global _user_profile
        _user_profile['keywords'] = list(DEFAULT_KEYWORDS)
        get_profile_vector.cache_clear()
        return jsonify({
            'status': 'ok',
            'insight_categories': DEFAULT_INSIGHT_CATEGORIES,
            'cross_cutting': DEFAULT_CROSS_CUTTING,
            'derived_keywords': DEFAULT_KEYWORDS,
            'derived_count': len(DEFAULT_KEYWORDS),
            'per_category_limits': _PER_CATEGORY_LIMITS,
            'per_category_prompts': _PER_CATEGORY_PROMPTS,
            'per_category_summary_prompts': _PER_CATEGORY_SUMMARY_PROMPTS,
        })


@app.route('/api/clear_results', methods=['POST'])
def clear_results():
    """Clear the current digest results from UI (client-side only, no server cache to clear)."""
    return jsonify({'status': 'ok', 'message': 'Results cleared'})


@app.route('/api/test_llm', methods=['POST'])
@limiter.limit("20 per hour")
def test_llm():
    """Test the LLM API with a single cheap call. Returns raw response for diagnostics."""
    data = request.json or {}
    api_key = data.get('api_key', '').strip()
    provider = data.get('provider', 'moonshot')
    if not api_key:
        return jsonify({'error': 'No API key provided'}), 400

    provider_config = LLM_PROVIDERS.get(provider, LLM_PROVIDERS['moonshot'])
    base_url = provider_config['base_url']
    # Allow model override from request
    model = data.get('model', '') or provider_config['model']
    fmt = provider_config.get('format', 'openai')

    test_prompt = "Respond with ONLY this JSON: {\"score\": 42, \"one_line_reason\": \"Test successful\"}"

    try:
        if fmt == 'anthropic':
            resp = requests.post(
                f"{base_url}/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": test_prompt}],
                    "max_tokens": 128,
                    "temperature": 0.0,
                },
                timeout=20
            )
        else:
            resp = requests.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": test_prompt}],
                    "temperature": 0.0,
                    "max_tokens": 128
                },
                timeout=20
            )

        result = {
            'status_code': resp.status_code,
            'headers': dict(resp.headers),
        }
        try:
            result['json'] = resp.json()
        except Exception:
            result['text'] = resp.text[:500]

        if resp.status_code == 200:
            content = _extract_content(result['json'], fmt)
            result['extracted_content'] = content
            # Try to parse JSON from it
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content)
            if not json_match:
                json_match = re.search(r'\{[^}]+\}', content)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    result['parsed_json'] = parsed
                except Exception as e:
                    result['json_parse_error'] = str(e)
            else:
                result['json_not_found'] = True

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    app.run(debug=debug, host='0.0.0.0', port=port)
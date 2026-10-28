import re
from io import BytesIO

import requests
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ResearchBuddy AI",
    page_icon="📚",
    layout="wide"
)

st.title("📚 ResearchBuddy AI")
st.subheader("Your personal research assistant")

st.write(
    "Upload a research paper, ask a question, and get a concise "
    "AI-generated answer supported by evidence from the document."
)


# --------------------------------------------------
# PDF EXTRACTION AND CHUNKING
# --------------------------------------------------


def extract_chunks(pdf_bytes, chunk_size=180, overlap=40):
    reader = PdfReader(BytesIO(pdf_bytes))
    chunks = []
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        # Fix words split across PDF line breaks
        text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

        # Fix words split by spaces around a hyphen
        text = re.sub(r"\b(\w+)\s+-\s+(\w+)\b", r"\1\2", text)

        # Normalize whitespace after repairing broken words
        text = re.sub(r"\s+", " ", text).strip()
 
        # Preserve the complete page for sentence-level retrieval.
        pages.append({
            "page": page_number,
            "text": text
        })

        words = text.split()
        start = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])

            if chunk_text.strip():
                chunks.append({
                    "page": page_number,
                    "text": chunk_text
                })

            if end == len(words):
                break

            start = end - overlap

    return len(reader.pages), chunks, pages



# --------------------------------------------------
# SENTENCE SPLITTING
# --------------------------------------------------


def split_sentences(text):
    # Join words broken across PDF line breaks.
    text = re.sub(
        r"(\w+)-\s*\n\s*(\w+)",
        r"\1\2",
        text
    )

    # Replace remaining line breaks with spaces.
    text = re.sub(r"\s+", " ", text).strip()

    # Split text at sentence-ending punctuation.
    parts = re.split(
        r'(?<=[.!?])\s+(?=["“‘\']?[A-Z0-9])',
        text
    )

    return [
        sentence.strip()
        for sentence in parts
        if len(sentence.split()) >= 4
    ]


# --------------------------------------------------
# SECTION-AWARE EVIDENCE RETRIEVAL
# --------------------------------------------------




def find_evidence(question, chunks, pages):
    if not question.strip() or not pages:
        return [], []

    question_lower = question.lower()

    # Handle specific paper sections.
    section_patterns = {
        "author contributions": [
            "author contributions",
            "authors' contributions",
            "authors contributions",
            "contributions of the authors",
        ],
        "funding": ["funding", "financial support"],
        "data availability": [
            "data availability",
            "availability of data",
        ],
        "code availability": [
            "code availability",
            "availability of code",
        ],
    }

    target_section = None

    if any(term in question_lower for term in [
        "author contribution",
        "authors' contribution",
        "authors contribution",
        "contribution of the authors",
    ]):
        target_section = "author contributions"
    elif "funding" in question_lower or "financial support" in question_lower:
        target_section = "funding"
    elif "data availability" in question_lower:
        target_section = "data availability"
    elif "code availability" in question_lower:
        target_section = "code availability"

    if target_section:
        for page in pages:
            text = page["text"]
            lower_text = text.lower()

            for heading in section_patterns[target_section]:
                position = lower_text.find(heading)

                if position < 0:
                    continue

                passage = text[position:]

                stop_patterns = [
                    r"\bAuthor contributions\b",
                    r"\bFunding\b",
                    r"\bData availability\b",
                    r"\bCode availability\b",
                    r"\bEthics approval\b",
                    r"\bConflicts? of interest\b",
                    r"\bAcknowledgements?\b",
                    r"\bReferences\b",
                ]

                stops = []

                for pattern in stop_patterns:
                    match = re.search(
                        pattern, passage, re.IGNORECASE
                    )
                    if match and match.start() > 0:
                        stops.append(match.start())

                if stops:
                    passage = passage[:min(stops)]

                passage = " ".join(passage.split()).strip()

                if passage:
                    result = [{
                        "page": page["page"],
                        "text": passage[:1200],
                        "score": 1.0,
                    }]
                    return result, [dict(result[0])]

    # Detect the kind of information being requested.
    asks_challenges = any(term in question_lower for term in [
        "challenge", "challenges", "problem", "problems",
        "difficulty", "difficulties", "limitation", "limitations",
    ])

    asks_datasets = any(term in question_lower for term in [
        "dataset", "datasets", "benchmark", "benchmarks",
        "corpus", "corpora", "test collection",
    ])

    asks_results = any(term in question_lower for term in [
        "result", "results", "performance", "evaluate",
        "evaluation", "metric", "metrics", "ndcg", "map",
        "mrr", "precision", "recall", "improvement",
        "outperform", "compared to",
    ])

    asks_ranking = any(term in question_lower for term in [
        "rank", "ranking", "relevance", "reranking",
        "re-ranking", "most relevant",
    ])

    # Expand the query with terms related to the question.
    retrieval_query = question

    if asks_challenges:
        retrieval_query += (
            " challenges problems difficulties limitations "
            "exact matching vocabulary mismatch semantic matching "
            "context dependent variable length"
        )

    if asks_datasets:
        retrieval_query += (
            " dataset benchmark corpus collection test set "
            "TREC MS MARCO ClueWeb evaluation"
        )

    if asks_results:
        retrieval_query += (
            " experimental results evaluation measures metrics "
            "NDCG MAP MRR precision recall percentage gain "
            "improvement comparison baseline performance"
        )

    if asks_ranking:
        retrieval_query += (
            " ranking ranked documents neural ranking models "
            "relevance BERT reranking query document matching "
            "semantic matching vocabulary mismatch"
        )

    # Build candidates using multiple window sizes.
    # This helps preserve context around datasets and results.
    candidates = []
    seen = set()

    def add_candidate(page_number, text):
        text = " ".join(text.split()).strip()

        if not text:
            return

        key = (page_number, text.lower())

        if key not in seen:
            seen.add(key)
            candidates.append({
                "page": page_number,
                "text": text,
            })

    for page in pages:
        page_text = page["text"]

        if not page_text or not page_text.strip():
            continue

        sentences = split_sentences(page_text)

        # Add individual sentences and short sentence windows.
        for i in range(len(sentences)):
            for window_size in (1, 2, 3):
                window = sentences[i:i + window_size]

                if window:
                    add_candidate(
                        page["page"],
                        " ".join(window)
                    )

        # Also examine overlapping word windows. These can
        # recover useful content when PDF punctuation or tables
        # cause sentence splitting to work poorly.
        words = page_text.split()
        window_size = 110
        stride = 75

        if len(words) <= window_size:
            add_candidate(page["page"], page_text)
        else:
            for start in range(
                0, len(words), stride
            ):
                window = words[start:start + window_size]

                if window:
                    add_candidate(
                        page["page"],
                        " ".join(window)
                    )

    if not candidates:
        return [], []

    candidate_texts = [item["text"] for item in candidates]

    try:
        word_vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=50000,
            sublinear_tf=True,
        )

        word_matrix = word_vectorizer.fit_transform(
            candidate_texts
        )
        question_vector = word_vectorizer.transform(
            [retrieval_query]
        )

        if question_vector.nnz == 0:
            return [], []

        word_scores = cosine_similarity(
            question_vector, word_matrix
        ).flatten()

        char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=50000,
            sublinear_tf=True,
        )

        char_matrix = char_vectorizer.fit_transform(
            candidate_texts
        )
        char_question = char_vectorizer.transform(
            [retrieval_query]
        )

        char_scores = cosine_similarity(
            char_question, char_matrix
        ).flatten()

    except ValueError:
        return [], []

    # Measure coverage of meaningful words in the original question.
    stop_words = {
        "what", "which", "when", "where", "who", "whom",
        "how", "why", "does", "did", "doing", "is", "are",
        "was", "were", "the", "and", "for", "from", "with",
        "that", "this", "these", "those", "paper", "tell",
        "explain", "describe", "identify", "discuss", "main",
        "following", "proposed", "used", "were", "have",
        "has", "been", "based", "their", "they",
    }

    question_terms = set(re.findall(
        r"\b[a-zA-Z][a-zA-Z0-9-]*\b",
        question_lower
    ))

    question_terms = {
        term for term in question_terms
        if len(term) > 2 and term not in stop_words
    }

    coverage_scores = []

    for item in candidates:
        passage_terms = set(re.findall(
            r"\b[a-zA-Z][a-zA-Z0-9-]*\b",
            item["text"].lower()
        ))

        if question_terms:
            coverage = (
                len(question_terms & passage_terms)
                / len(question_terms)
            )
        else:
            coverage = 0.0

        coverage_scores.append(coverage)

    import numpy as np

    coverage_scores = np.array(coverage_scores)

    # Reward passages that contain evidence relevant to the
    # requested information type, not just similar wording.
    intent_boosts = np.zeros(len(candidates))

    for i, item in enumerate(candidates):
        text = item["text"].lower()

        if asks_datasets:
            dataset_signals = [
                "dataset", "datasets", "benchmark", "corpus",
                "trec", "ms marco", "clueweb", "test collection",
                "passage re-ranking", "document ranking",
            ]

            if any(term in text for term in dataset_signals):
                intent_boosts[i] += 0.06

        if asks_results:
            has_number = bool(re.search(
                r"\b\d+(?:\.\d+)?\s*%?"
                r"|\b\d+(?:\.\d+)?\s*"
                r"@\s*\d+",
                text
            ))

            result_signals = [
                "ndcg", "map", "mrr", "precision", "recall",
                "significant improvement", "performance gain",
                "outperform", "compared to", "results show",
                "achieved", "improvement", "evaluation results",
            ]

            if has_number:
                intent_boosts[i] += 0.06

            if any(term in text for term in result_signals):
                intent_boosts[i] += 0.05

    # Combine similarity, question-term coverage and intent signals.
    final_scores = (
        0.65 * word_scores
        + 0.15 * char_scores
        + 0.10 * coverage_scores
        + intent_boosts
    )

    best_score = float(final_scores.max())

    if best_score < 0.05:
        return [], []

    threshold = max(0.07, best_score * 0.40)
    ranked_indices = final_scores.argsort()[::-1]

    selected = []
    selected_token_sets = []
    selected_pages = set()

    for index in ranked_indices:
        score = float(final_scores[index])

        if score < threshold:
            break

        item = candidates[index]
        text = item["text"]

        tokens = set(re.findall(
            r"\b[a-zA-Z][a-zA-Z0-9-]*\b",
            text.lower()
        ))

        # Avoid near-duplicate passages.
                # Avoid near-duplicate and contained passages.
        duplicate = False

        for previous in selected:
            previous_text = previous["text"].lower()

            previous_tokens = set(re.findall(
                r"\b[a-zA-Z][a-zA-Z0-9-]*\b",
                previous_text
            ))

            union = tokens | previous_tokens
            intersection = tokens & previous_tokens

            if not union:
                continue

            jaccard = len(intersection) / len(union)

            smaller_size = min(
                len(tokens), len(previous_tokens)
            )

            containment = (
                len(intersection) / smaller_size
                if smaller_size
                else 0.0
            )

            if jaccard > 0.65 or containment > 0.85:
                duplicate = True
                break

        if duplicate:
            continue


        # Prefer different pages when their scores are comparable.
        if item["page"] in selected_pages:
            if len(selected) >= 2:
                continue

        selected.append({
            "page": item["page"],
            "text": text,
            "score": score,
        })

        selected_token_sets.append(tokens)
        selected_pages.add(item["page"])

        if len(selected) == 3:
            break

    evidence = [dict(item) for item in selected]

    return selected, evidence



    # --------------------------------------------------
    # NORMAL TF-IDF RETRIEVAL
    # --------------------------------------------------

    documents = [chunk["text"] for chunk in chunks]

    try:
        word_vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=30000,
            sublinear_tf=True
        )

        document_matrix = word_vectorizer.fit_transform(documents)
        question_vector = word_vectorizer.transform([question])

    except ValueError:
        return [], []

    if question_vector.nnz == 0:
        return [], []

    chunk_scores = cosine_similarity(
        question_vector,
        document_matrix
    ).flatten()

    # Reject questions with almost no lexical overlap.
    if chunk_scores.max() < 0.05:
        return [], []


    top_indices = chunk_scores.argsort()[::-1][:10]


    candidates = []
    seen = set()

    section_headings = [
        "author contributions",
        "funding",
        "data availability",
        "code availability",
    ]

    # Search sentences from every page, not just top-ranked chunks.
    for page in pages:
        page_number = page["page"]
        page_text = page["text"]

        if not page_text:
            continue

        sentences = split_sentences(page_text)

        filtered_sentences = []

        for sentence in sentences:
            sentence_lower = sentence.lower()

            if any(
                sentence_lower.startswith(heading)
                for heading in section_headings
            ):
                if not any(
                    heading in question_lower
                    for heading in section_headings
                ):
                    continue

            filtered_sentences.append(sentence)

        for sentence_index, sentence in enumerate(filtered_sentences):
            key = (page_number, sentence.lower())

            if key in seen:
                continue

            seen.add(key)

            # Include nearby sentences as context.
            context_start = max(0, sentence_index - 1)
            context_end = min(
                len(filtered_sentences),
                sentence_index + 2
            )

            context = " ".join(
                filtered_sentences[context_start:context_end]
            ).strip()

            if context:
                candidates.append({
                    "page": page_number,
                    "text": context
                })

    if not candidates:
        return [], []



    candidate_texts = [
        item["text"]
        for item in candidates
    ]

    try:
        word_sentence_matrix = word_vectorizer.transform(
            candidate_texts
        )

        char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            max_features=50000,
            sublinear_tf=True
        )

        char_matrix = char_vectorizer.fit_transform(
            candidate_texts
        )

        char_question = char_vectorizer.transform([question])

        word_scores = cosine_similarity(
            question_vector,
            word_sentence_matrix
        ).flatten()

        char_scores = cosine_similarity(
            char_question,
            char_matrix
        ).flatten()

    except ValueError:
        return [], []

    final_scores = (
        0.8 * word_scores
        + 0.2 * char_scores
    )

    best_score = float(final_scores.max())
    threshold = max(0.18, best_score * 0.65)

    ranked_indices = final_scores.argsort()[::-1]

    # --------------------------------------------------
    # SELECT DISTINCT EVIDENCE
    # --------------------------------------------------


    # --------------------------------------------------
    # SELECT RELEVANT, DIVERSE EVIDENCE
    # --------------------------------------------------

    selected = []
    remaining_indices = [
        int(index)
        for index in ranked_indices
        if float(final_scores[index]) >= threshold
    ]

    # Balance relevance against redundancy.
    relevance_weight = 0.75
    max_evidence = 2

    while remaining_indices and len(selected) < max_evidence:
        best_index = None
        best_mmr_score = float("-inf")

        for index in remaining_indices:
            relevance = float(final_scores[index]) / max(
                best_score, 1e-9
            )

            redundancy = 0.0

            if selected:
                similarities = cosine_similarity(
                    word_sentence_matrix[index],
                    word_sentence_matrix[
                        [item["_index"] for item in selected]
                    ]
                ).flatten()

                redundancy = float(similarities.max())

            mmr_score = (
                relevance_weight * relevance
                - (1 - relevance_weight) * redundancy
            )

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_index = index

        if best_index is None:
            break

        item = candidates[best_index]

        selected.append({
            "page": item["page"],
            "text": item["text"],
            "score": float(final_scores[best_index]),
            "_index": best_index
        })

        remaining_indices.remove(best_index)

    # Remove internal ranking metadata before displaying or sending evidence.
    selected = [
        {
            "page": item["page"],
            "text": item["text"],
            "score": item["score"]
        }
        for item in selected
    ]

    # Pass exactly the displayed evidence to Qwen.
    evidence = [dict(item) for item in selected]

    return selected, evidence



# --------------------------------------------------
# OLLAMA / QWEN AI ANSWER GENERATION
# --------------------------------------------------


def generate_answer(question, evidence_items):
    if not evidence_items:
        return (
            "I couldn't find relevant evidence in the uploaded document."
        )

    evidence_text = "\n\n".join(
        f"[Page {item['page']}] {item['text']}"
        for item in evidence_items[:3]
    )

    prompt = f"""
You are ResearchBuddy AI, an assistant that answers questions
using evidence retrieved from a research paper.

STRICT REQUIREMENTS:
- Answer only using the supplied evidence.
- Every factual claim must be explicitly supported by the evidence.
- Do not use background knowledge, assumptions, or outside information.
- First identify the useful information in EACH evidence snippet.
- For every snippet, determine whether it contributes a distinct
  fact that answers the question.
- Include all distinct, relevant facts in the final answer.
- For the question about document ranking, explain both the ranking
  process and the BERT-based second-stage reranking if both are
  supported by the supplied evidence.
- Do not force unrelated evidence into the answer.
- Cite each claim using the page number that supports it.
- Do not repeat the same information in different words.
- After each factual claim, cite its supporting page using
  the format (Page N).
- If a claim is supported by multiple pages, cite them as
  (Pages N, M).
- Use only page numbers provided in the evidence.
- Never invent page numbers, quotations, methods, or findings.

Answer the user's question using only the retrieved evidence.

Rules:
1. Answer the specific question directly.
2. If the question asks for multiple items, list each item supported by the evidence.
3. Write complete sentences. Never return only an introduction followed by a colon.
4. Explain the relevant details from the evidence without repeating unrelated information.
5. Do not invent facts, datasets, metrics, or experimental results.
6. If the evidence partially answers the question, provide the supported information and clearly state what is missing.
7. If the evidence does not answer the question at all, respond exactly:
"The retrieved evidence is insufficient to answer this question."
8. Cite page numbers when they are available in the retrieved evidence.
9. Keep the answer concise but complete.

- If the evidence answers only part of the question, explain
  what can be answered and clearly state the limitation.
- Keep the answer concise, clear, and directly relevant to the question.
- Return only the answer. Do not include a heading or explain your process.

EVIDENCE:
{evidence_text}

QUESTION:
{question}

ANSWER:
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:3b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1
                }
            },
            timeout=180
        )

        response.raise_for_status()

        answer = response.json().get(
            "response",
            ""
        ).strip()

        if not answer:
            return (
                "The AI model returned an empty answer. Please try again."
            )

        return answer

    except requests.ConnectionError:
        return (
            "Could not connect to Ollama. Please make sure Ollama is "
            "running on your computer and try again."
        )

    except requests.Timeout:
        return (
            "The AI model took too long to respond. Please try again."
        )

    except requests.RequestException as error:
        return (
            f"An error occurred while contacting Ollama: {error}"
        )



# --------------------------------------------------
# STREAMLIT USER INTERFACE
# --------------------------------------------------


# --------------------------------------------------
# POLISHED STREAMLIT INTERFACE
# --------------------------------------------------

st.markdown("""
<style>
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 1.8rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #172554, #1d4ed8);
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        color: white;
        font-size: 2.25rem;
        margin-bottom: 0.4rem;
    }

    .hero p {
        color: #dbeafe;
        font-size: 1rem;
        margin-bottom: 0;
    }

    .section-label {
        font-size: 1.15rem;
        font-weight: 650;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    .answer-card {
        padding: 1.25rem;
        border: 1px solid rgba(128, 128, 128, 0.3);
        border-radius: 14px;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }

    .status-note {
        color: #64748b;
        font-size: 0.9rem;
    }

    div[data-testid="stFileUploader"] {
        border: 1px dashed #64748b;
        border-radius: 12px;
        padding: 0.8rem;
    }

    div.stButton > button[kind="primary"] {
        border-radius: 10px;
        min-height: 2.8rem;
        font-weight: 600;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 12px;
        padding: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>📚 ResearchBuddy AI</h1>
    <p>Your research companion. Upload a paper, ask questions,
    and explore answers backed by evidence from the document.</p>
</div>
""", unsafe_allow_html=True)

# -------------------------
# PDF UPLOAD
# -------------------------

st.markdown('<div class="section-label">01 · Upload your paper</div>',
            unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a research paper in PDF format",
    type=["pdf"],
    help="Upload a text-based PDF to get started."
)

if uploaded_file is None:
    st.info(
        "👆 Upload a PDF to begin. Your questions will be answered "
        "using evidence retrieved from that document."
    )

else:
    try:
        total_pages, chunks, pages = extract_chunks(
            uploaded_file.getvalue()
        )

        if not chunks:
            st.warning(
                "No readable text was found. This PDF may contain "
                "scanned pages."
            )

        else:
            st.success(f"Paper ready: {uploaded_file.name}")

            col1, col2, col3 = st.columns(3)

            col1.metric("📄 Pages", total_pages)
            col2.metric("🧩 Text passages", len(chunks))
            col3.metric(
                "🔤 Extracted characters",
                f"{sum(len(chunk['text']) for chunk in chunks):,}"
            )

            with st.expander("Preview extracted text"):
                st.text_area(
                    "Extracted document content",
                    value="\n\n".join(
                        f"Page {chunk['page']}\n{chunk['text']}"
                        for chunk in chunks
                    ),
                    height=250,
                    disabled=True
                )

            st.divider()

            # -------------------------
            # ASK A QUESTION
            # -------------------------

            st.markdown(
                '<div class="section-label">02 · Ask your paper</div>',
                unsafe_allow_html=True
            )

            question = st.text_input(
                "What would you like to understand?",
                placeholder=(
                    "e.g. What are the main findings of this paper?"
                ),
                key="research_question"
            )

            ask_button = st.button(
                "✨ Generate answer",
                type="primary",
                use_container_width=True
            )

            if ask_button:
                if not question.strip():
                    st.warning("Please enter a question first.")

                else:
                    with st.spinner(
                        "Searching the paper for relevant evidence..."
                    ):
                        answers, evidence = find_evidence(
                            question,
                            chunks,
                            pages
                        )

                    if not answers or not evidence:
                        st.warning(
                            "I couldn't find sufficiently relevant evidence. "
                            "Try rephrasing your question or using keywords "
                            "from the paper."
                        )

                    else:
                        with st.spinner(
                            "Qwen is preparing your evidence-based answer..."
                        ):
                            ai_answer = generate_answer(
                                question,
                                evidence
                            )

                        # -------------------------
                        # ANSWER
                        # -------------------------

                        st.divider()

                        st.markdown(
                            '<div class="section-label">'
                            '03 · AI-generated answer</div>',
                            unsafe_allow_html=True
                        )

                        st.markdown(
                            '<div class="answer-card">',
                            unsafe_allow_html=True
                        )
                        st.markdown(ai_answer)
                        st.markdown("</div>", unsafe_allow_html=True)

                        source_pages = sorted({
                            item["page"] for item in evidence
                        })

                        st.caption(
                            "📖 Sources: Page(s) "
                            + ", ".join(map(str, source_pages))
                        )

                        st.caption(
                            "Generated locally using Qwen 2.5 3B via Ollama. "
                            "Verify important claims against the original paper."
                        )

                        # -------------------------
                        # EVIDENCE
                        # -------------------------

                        st.markdown(
                            '<div class="section-label">'
                            '04 · Retrieved evidence</div>',
                            unsafe_allow_html=True
                        )

                        st.caption(
                            "Review the passages used to generate the answer."
                        )

                        for rank, item in enumerate(answers, start=1):
                            with st.container(border=True):
                                st.markdown(
                                    f"**Evidence {rank} · Page "
                                    f"{item['page']}**"
                                )
                                st.write(item["text"])
                                st.caption(
                                    f"Similarity score: {item['score']:.3f}"
                                )

                        with st.expander("Inspect supporting passages"):
                            for rank, item in enumerate(
                                evidence[:3], start=1
                            ):
                                st.markdown(
                                    f"**Passage {rank} · Page "
                                    f"{item['page']}**"
                                )
                                st.write(item["text"])
                                st.caption(
                                    f"Similarity score: {item['score']:.3f}"
                                )

    except Exception as error:
        st.error(f"Could not process this PDF: {error}")

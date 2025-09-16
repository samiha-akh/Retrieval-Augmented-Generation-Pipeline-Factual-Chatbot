# Install dependencies
#!pip install -U langchain langchain-community unstructured python-docx haystack-ai sentence-transformers transformers faiss-cpu
#!pip install -U unstructured[local-inference] pdfminer.six


# Imports

import os
import json
import re
import collections
import spacy
from typing import List, Dict, Optional

from langchain.document_loaders import UnstructuredPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import faiss
import pickle
from google.colab import files

# Utility functions

def save_json_atomic(data, filename: str):
    parent = os.path.dirname(filename)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = filename + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, filename)

def fix_hyphenation(text: str) -> str:
    return re.sub(r'([a-zA-Z]+)-\s+([a-zA-Z]+)', r'\1\2', text)

def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def normalize_quotes(text: str) -> str:
    replacements = {
        '“': '"', '”': '"',
        "‘": "'", "’": "'",
        '—': '-', '–': '-'
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

def load_document(file_path: str):
    if file_path.lower().endswith('.pdf'):
        loader = UnstructuredPDFLoader(file_path)
    elif file_path.lower().endswith('.docx'):
        loader = UnstructuredWordDocumentLoader(file_path)
    elif file_path.lower().endswith('.txt'):
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    return loader.load()

def extract_page_lines(raw_docs) -> List[List[str]]:
    pages = []
    if isinstance(raw_docs, list):
        for doc in raw_docs:
            content = getattr(doc, "page_content", str(doc))
            lines = [l.strip() for l in content.splitlines() if l.strip()]
            pages.append(lines)
    else:
        text = str(raw_docs)
        raw_pages = re.split(r'\n{2,}', text)
        for chunk in raw_pages:
            lines = [l.strip() for l in chunk.splitlines() if l.strip()]
            pages.append(lines)
    return pages

def detect_repeated_lines(pages: List[List[str]], threshold: float = 0.4) -> set:
    line_counts = collections.Counter()
    for page in pages:
        unique = set(page)
        line_counts.update(unique)
    num_pages = len(pages)
    repeated = {line for line, cnt in line_counts.items() if cnt / num_pages >= threshold}
    return repeated

def strip_headers_footers(raw_text: str, repeated_lines: set) -> str:
    filtered = "\n".join([l for l in raw_text.splitlines() if l.strip() not in repeated_lines])
    return filtered

#  spaCy setup
import en_core_web_sm
_nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
if not _nlp.has_pipe("sentencizer"):
    _nlp.add_pipe("sentencizer")

#  Question generation model helpers
QG_MODEL = "valhalla/t5-base-qg-hl"

def load_qg_model(device=None):
    tokenizer = AutoTokenizer.from_pretrained(QG_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(QG_MODEL)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

def make_qg_input(fact: str, context: Optional[str] = None) -> str:
    fact_clean = fact.strip()
    if context:
        if fact_clean in context:
            highlighted = context.replace(fact_clean, f"<hl> {fact_clean} <hl>")
        else:
            highlighted = f"<hl> {fact_clean} <hl> {context}"
    else:
        highlighted = f"<hl> {fact_clean} <hl>"
    return f"generate question: {highlighted}"

def postprocess_question(q: str) -> str:
    q = q.strip()
    if not q.endswith("?"):
        q = q.rstrip(".") + "?"
    if q:
        q = q[0].upper() + q[1:]
    return q

def facts_to_questions(fact_items: List[Dict],
                       fact_key="answer",
                       context_key="context",
                       question_key="question") -> List[Dict]:
    tokenizer, model, device = load_qg_model()
    output = []
    for item in fact_items:
        fact = item.get(fact_key, "").strip()
        if not fact:
            continue
        context = item.get(context_key, None)
        inp = make_qg_input(fact, context)
        inputs = tokenizer.encode(inp, return_tensors="pt", truncation=True, max_length=512).to(device)
        try:
            gen = model.generate(
                inputs,
                max_length=64,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
            question = tokenizer.decode(gen[0], skip_special_tokens=True)
            question = postprocess_question(question)
        except Exception:
            question = postprocess_question(f"What is the key point of: {fact}")
        new_item = item.copy()
        new_item[question_key] = question
        output.append(new_item)
    return output

#  Paraphrase augmentation
PARAPHRASER_MODEL = "ramsrigouthamg/t5_paraphraser"

def generate_paraphrases(question: str, num_return=2, model_name=PARAPHRASER_MODEL, device=None) -> List[str]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    input_text = f"paraphrase: {question} </s>"
    encoding = tokenizer.encode_plus(
        input_text, padding="longest", return_tensors="pt", truncation=True, max_length=256
    ).to(device)
    outputs = model.generate(
        **encoding,
        max_length=64,
        num_beams=5,
        num_return_sequences=num_return,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    paraphrases = set()
    for out in outputs:
        p = tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        paraphrases.add(p.strip())
    return list(paraphrases)

def augment_with_paraphrases(qa_pairs: List[Dict], per_question=1) -> List[Dict]:
    augmented = []
    for pair in qa_pairs:
        augmented.append(pair)
        try:
            paras = generate_paraphrases(pair["question"], num_return=per_question)
            for p in paras:
                new = pair.copy()
                new["question"] = postprocess_question(p)
                augmented.append(new)
        except Exception:

            pass
    return augmented

#  FAISS semantic store builder
def build_and_persist_qa_store(qa_pairs,
                               embedder_name="sentence-transformers/all-MiniLM-L6-v2",
                               index_path="faiss_question.index",
                               store_path="qa_store.pkl"):
    embedder = SentenceTransformer(embedder_name)
    questions = [pair["question"] for pair in qa_pairs]
    question_embeddings = embedder.encode(questions, convert_to_numpy=True, normalize_embeddings=True)

    dim = question_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(question_embeddings)


    faiss.write_index(index, index_path)
    with open(store_path, "wb") as f:
        pickle.dump({
            "qa_pairs": qa_pairs,
            "embedder_name": embedder_name
        }, f)
    print(f"[INFO] Persisted index to {index_path} and store to {store_path}")
    return index, embedder

def load_qa_store(index_path="faiss_question.index", store_path="qa_store.pkl"):
    index = faiss.read_index(index_path)
    with open(store_path, "rb") as f:
        data = pickle.load(f)
    embedder = SentenceTransformer(data["embedder_name"])
    qa_pairs = data["qa_pairs"]
    return index, embedder, qa_pairs

# Retrieval with reranking
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker = CrossEncoder(RERANKER_MODEL)

def retrieve_answer(user_query: str,
                    index,
                    embedder: SentenceTransformer,
                    qa_pairs: List[Dict],
                    top_k=5,
                    similarity_threshold=0.55):
    q_emb = embedder.encode([user_query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = index.search(q_emb, top_k)
    scores = scores[0]
    idxs = idxs[0]

    candidates = []
    for score, idx in zip(scores, idxs):
        if score >= similarity_threshold:
            candidate = qa_pairs[idx].copy()
            candidate["retrieval_score"] = float(score)
            candidates.append(candidate)
    if not candidates:
        return None

    # Rerank
    pair_inputs = [[user_query, c["question"]] for c in candidates]
    rerank_scores = reranker.predict(pair_inputs)
    for c, rs in zip(candidates, rerank_scores):
        c["rerank_score"] = float(rs)

    best = max(candidates, key=lambda x: x["rerank_score"])
    return best

# Hybrid QA generation
def chunk_sentences(sentences: List[Dict], max_words=200):
    chunks = []
    current_chunk = []
    current_len = 0
    for item in sentences:
        word_count = len(item["text"].split())
        if current_len + word_count > max_words and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_len = 0
        current_chunk.append(item)
        current_len += word_count
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def generate_qa_pairs_hybrid(processed_data):
    # Fact extraction summarizer
    fact_node = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)
    qg_node = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl", device=0 if torch.cuda.is_available() else -1)

    chunks = chunk_sentences(processed_data, max_words=200)
    qa_pairs = []

    for chunk in chunks:
        chunk_text = " ".join([item["text"] for item in chunk])

        # Extract a "fact" summary from the chunk
        facts_result = fact_node(chunk_text, max_length=150, min_length=30, do_sample=False)
        facts_text = facts_result[0].get('summary_text', '').strip() if isinstance(facts_result[0], dict) else facts_result[0].strip()

        if not facts_text:
            continue

        # Generate a question from that fact
        # Here we pass the fact as both fact and context for simplicity
        q_input = make_qg_input(facts_text, chunk_text)
        tokenizer, model, device = load_qg_model()
        inputs = tokenizer.encode(q_input, return_tensors="pt", truncation=True, max_length=512).to(device)
        gen = model.generate(
            inputs,
            max_length=64,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
        question = tokenizer.decode(gen[0], skip_special_tokens=True)
        question = postprocess_question(question)

        qa_pairs.append({
            "question": question,
            "answer": facts_text,
            "context": chunk_text,
            "source": {"chunk_id": len(qa_pairs)}
        })

    return qa_pairs

#  Main execution
def process_file_and_build_store(file_path: str, augment_paraphrases: bool = True):
    print(f"[INFO] Processing {file_path}...")
    raw_docs = load_document(file_path)

    # Reconstruct text
    if isinstance(raw_docs, list):
        raw_text = "\n\n".join([getattr(doc, "page_content", str(doc)) for doc in raw_docs])
    else:
        raw_text = str(raw_docs)

    # Normalize
    raw_text = fix_hyphenation(raw_text)
    raw_text = normalize_quotes(raw_text)
    raw_text = normalize_whitespace(raw_text)

    # Header/footer removal
    pages = extract_page_lines(raw_docs)
    repeated = detect_repeated_lines(pages, threshold=0.4)
    cleaned_text = strip_headers_footers(raw_text, repeated)

    save_json_atomic({"raw_text": cleaned_text}, "raw_data.json")

    # Split into sentence-level "processed" entries
    raw_paragraphs = [p.strip() for p in re.split(r'\n{2,}', cleaned_text) if p.strip()]
    processed = []
    global_sentence_id = 0
    for paragraph_id, para in enumerate(raw_paragraphs):
        para_doc = _nlp(para)
        for sent in para_doc.sents:
            sentence_text = sent.text.strip()
            if not sentence_text:
                continue
            entry = {
                "paragraph_id": paragraph_id,
                "sentence_id": global_sentence_id,
                "text": sentence_text
            }
            processed.append(entry)
            global_sentence_id += 1

    save_json_atomic(processed, "processed_data.json")

    # Generate QA via hybrid method
    qa_pairs = generate_qa_pairs_hybrid(processed)

    # Replace or ensure question is present (already generated) and deduplicate
    unique = []
    seen = set()
    for item in qa_pairs:
        key = (item["question"].lower().strip(), item["answer"].lower().strip())
        if key not in seen:
            unique.append(item)
            seen.add(key)
    print(f"[INFO] Generated {len(unique)} unique QA pairs from hybrid pipeline.")

    # Optionally augment with paraphrases
    if augment_paraphrases:
        expanded = augment_with_paraphrases(unique, per_question=1)
        print(f"[INFO] Expanded to {len(expanded)} QA pairs after paraphrasing.")
    else:
        expanded = unique

    # Build semantic store
    index, embedder = build_and_persist_qa_store(expanded)

    # Save final QA file
    save_json_atomic(expanded, "qa_with_generated_questions.json")
    print("[INFO] Pipeline complete. Files saved: raw_data.json, processed_data.json, qa_with_generated_questions.json")

    return index, embedder, expanded



# Upload a file
uploaded = files.upload()
if not uploaded:
    raise RuntimeError("No file uploaded. Please upload a .pdf, .docx, or .txt file.")
last_filename = list(uploaded.keys())[-1]
with open(last_filename, "wb") as f:
    f.write(uploaded[last_filename])

# Process + build index/store
index, embedder, qa_store = process_file_and_build_store(last_filename, augment_paraphrases=True)

# Interactive retrieval loop
print("\n=== Retrieval test: ask questions (type 'exit' to stop) ===")
while True:
    user_q = input("Your question: ").strip()
    if user_q.lower() in {"exit", "quit"}:
        break
    best = retrieve_answer(user_q, index, embedder, qa_store, top_k=5, similarity_threshold=0.5)
    if best:
        print("\nMatched canonical question:", best["question"])
        print("Answer:", best["answer"])
        print("Context snippet:", best["context"][:300].replace("\n", " "), "...")
        print("Source:", best.get("source"))
        print(f"Scores -> retrieval: {best.get('retrieval_score'):.3f}, rerank: {best.get('rerank_score'):.3f}\n")
    else:
        print("No confident match found. Try rephrasing or give more detail.\n")


for fname in ["raw_data.json", "processed_data.json", "qa_with_generated_questions.json", "faiss_question.index", "qa_store.pkl"]:
    if os.path.exists(fname):
        try:
            files.download(fname)
        except Exception:
            print(f"[WARN] Could not auto-download {fname}, but it's present in the environment.")

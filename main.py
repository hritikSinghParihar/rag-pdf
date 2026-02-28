import argparse
import logging

from retrieve import retrieve_relevant_chunks
from generate import generate_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="CLI for Local RAG QA")
    parser.add_argument("question", type=str, help="Question to ask")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k chunks")
    args = parser.parse_args()

    chunks = retrieve_relevant_chunks(args.question, top_k=args.top_k)
    answer = generate_answer(chunks, args.question)
    print("Answer:\n", answer)
    print("\nSources:")
    for ch in chunks:
        print(f"- {ch.get('source')} (page {ch.get('page')})")

if __name__ == "__main__":
    main()

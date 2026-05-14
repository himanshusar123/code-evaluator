"""
RAG System - Simple Human-like Execution
=========================================

This app reads documents, understands questions, and gives smart answers.
It uses a simple approach called RAG (Retrieval Augmented Generation).

STEP BY STEP:
1. Load documents
2. Break into small pieces (chunks)
3. Convert text to numbers (embeddings)
4. Store in searchable format (vector store)
5. When user asks a question:
   - Convert question to numbers
   - Find relevant pieces
   - Give smart answer
"""

import re
import math
from collections import defaultdict

# ============================================================
# STEP 1: LOAD AND PREPARE DOCUMENTS
# ============================================================

def load_documents():
    """Load sample documents for the system."""

    print("📚 Loading documents...")

    documents = [
        {
            "type": "policy",
            "content": """
            REFUND POLICY:
            - 100% refund if requested within 7 days of purchase
            - 50% refund if requested between 7-30 days with less than 30% course completion
            - No refund after 30% completion or after 30 days
            - Refund requests must be submitted via the portal
            - Processing time: 5-7 business days
            """
        },
        {
            "type": "faq",
            "content": """
            FAQ - FREQUENTLY ASKED QUESTIONS:

            Q: How do I reset my password?
            A: Go to login page, click 'Forgot Password', enter your email, and follow the instructions.

            Q: Can I transfer between batches?
            A: Yes, you can request a batch transfer once per course. Submit request 7 days before batch starts.

            Q: How do I download my certificate?
            A: Certificates available after course completion. Go to My Learning > Certificates.
            """
        },
        {
            "type": "notes",
            "content": """
            LECTURE NOTES - Machine Learning:

            Machine Learning is when computers learn from data without being explicitly programmed.

            Types of Machine Learning:
            1. Supervised Learning - Learning with labeled data
               - Classification: Predicting categories (spam/not spam)
               - Regression: Predicting numbers (house prices)

            2. Unsupervised Learning - Learning without labels
               - Clustering: Grouping similar items (customer segments)
               - Dimensionality Reduction: Simplifying complex data

            3. Reinforcement Learning - Learning through trial and error
               - Used in games, robotics, autonomous systems
            """
        }
    ]

    print(f"   ✅ Loaded {len(documents)} documents")
    return documents

# ============================================================
# STEP 2: BREAK DOCUMENTS INTO CHUNKS
# ============================================================

def chunk_documents(documents, chunk_size=100):
    """Break large documents into smaller, searchable pieces."""

    print("\n✂️ Breaking documents into chunks...")

    all_chunks = []

    for doc in documents:
        content = doc["content"]
        doc_type = doc["type"]

        # Simple chunking: split by sentences
        sentences = content.split(". ")

        current_chunk = ""
        chunk_id = 0

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            # Add sentence to current chunk
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                # Save current chunk if not empty
                if current_chunk.strip():
                    all_chunks.append({
                        "content": current_chunk.strip(),
                        "type": doc_type,
                        "chunk_id": f"{doc_type}_{chunk_id}"
                    })
                    chunk_id += 1

                # Start new chunk
                current_chunk = sentence + ". "

        # Don't forget the last chunk
        if current_chunk.strip():
            all_chunks.append({
                "content": current_chunk.strip(),
                "type": doc_type,
                "chunk_id": f"{doc_type}_{chunk_id}"
            })

    print(f"   ✅ Created {len(all_chunks)} chunks")
    return all_chunks

# ============================================================
# STEP 3: CREATE WORD VOCABULARY
# ============================================================

def build_vocabulary(chunks):
    """Build a simple vocabulary from all chunks."""

    print("\n📖 Building vocabulary...")

    word_count = defaultdict(int)
    total_docs = len(chunks)

    # Count how many documents each word appears in
    for chunk in chunks:
        words = extract_words(chunk["content"])
        unique_words = set(words)

        for word in unique_words:
            word_count[word] += 1

    # Keep words that appear in at least 1 document
    vocabulary = {}
    for word, count in word_count.items():
        if count >= 1:
            vocabulary[word] = {
                "count": count,
                "idf": math.log(total_docs / (1 + count))  # Inverse Document Frequency
            }

    # Limit vocabulary size
    vocabulary = dict(list(vocabulary.items())[:500])

    print(f"   ✅ Vocabulary size: {len(vocabulary)} words")
    return vocabulary

def extract_words(text):
    """Extract simple words from text."""
    words = re.findall(r'\b[a-z]+\b', text.lower())
    return words

# ============================================================
# STEP 4: CREATE TEXT EMBEDDINGS (Convert text to numbers)
# ============================================================

def create_embeddings(chunks, vocabulary):
    """Convert text chunks into number vectors."""

    print("\n🔢 Creating embeddings...")

    def text_to_vector(text):
        """Convert a piece of text into a vector of numbers."""
        words = extract_words(text)
        vector = []

        # Simple TF-IDF approach
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1

        for word, stats in vocabulary.items():
            tf = word_freq.get(word, 0) / max(len(words), 1)
            idf = stats.get("idf", 1)
            vector.append(tf * idf)

        return vector

    # Create embeddings for all chunks
    for chunk in chunks:
        chunk["embedding"] = text_to_vector(chunk["content"])

    print(f"   ✅ Created embeddings for {len(chunks)} chunks")
    return chunks

# ============================================================
# STEP 5: SIMPLE SEARCH (Find relevant chunks)
# ============================================================

def cosine_similarity(vec1, vec2):
    """Calculate similarity between two vectors."""

    # Make vectors same length
    max_len = max(len(vec1), len(vec2))
    vec1 = vec1 + [0] * (max_len - len(vec1))
    vec2 = vec2 + [0] * (max_len - len(vec2))

    # Calculate dot product
    dot_product = sum(a * b for a, b in zip(vec1, vec2))

    # Calculate magnitudes
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))

    if mag1 == 0 or mag2 == 0:
        return 0

    return dot_product / (mag1 * mag2)

def search_chunks(query, chunks, top_k=3):
    """Find chunks most relevant to the query."""

    print(f"\n🔍 Searching for: '{query}'")

    # Create query vector
    words = extract_words(query)
    word_freq = defaultdict(int)
    for word in words:
        word_freq[word] += 1

    query_vector = []
    for word, stats in vocabulary.items():
        tf = word_freq.get(word, 0) / max(len(words), 1)
        idf = stats.get("idf", 1)
        query_vector.append(tf * idf)

    # Compare query with all chunks
    results = []
    for chunk in chunks:
        similarity = cosine_similarity(query_vector, chunk["embedding"])
        results.append((chunk, similarity))

    # Sort by similarity
    results.sort(key=lambda x: x[1], reverse=True)

    # Return top results
    top_results = []
    for chunk, score in results[:top_k]:
        if score > 0:
            top_results.append((chunk, score))

    return top_results

# ============================================================
# STEP 6: UNDERSTAND USER QUESTION (Intent Classification)
# ============================================================

def classify_question(question):
    """Figure out what type of question the user is asking."""

    question = question.lower()

    # Check for keywords
    if any(word in question for word in ["refund", "money", "cancel"]):
        return "policy"
    elif any(word in question for word in ["lecture", "learn", "machine learning", "explain"]):
        return "notes"
    elif any(word in question for word in ["how", "what", "can i", "where"]):
        return "faq"
    else:
        return "general"

# ============================================================
# STEP 7: GENERATE ANSWER
# ============================================================

def generate_answer(question, relevant_chunks):
    """Create a helpful answer from relevant chunks."""

    if not relevant_chunks:
        return "I couldn't find any relevant information to answer your question."

    print("\n💬 Generating answer...")

    # Combine relevant information
    answer_parts = []

    answer_parts.append(f"Based on the documents, here's what I found:\n")

    for i, (chunk, score) in enumerate(relevant_chunks, 1):
        answer_parts.append(f"\n[{i}] From {chunk['type']} documents:")
        answer_parts.append(f"   {chunk['content'][:200]}...")

    # Add helpful note
    answer_parts.append("\n\nNote: This answer is based on the information available in our documents.")

    return "\n".join(answer_parts)

# ============================================================
# STEP 8: TOOL SYSTEM (For external data)
# ============================================================

def get_student_status(student_id):
    """Simulate fetching student data from database."""

    print(f"\n🔧 Checking student status for: {student_id}")

    # In real app, this would call an API
    mock_data = {
        "student_id": student_id,
        "enrolled_courses": ["AI Fundamentals", "Data Science Bootcamp"],
        "progress": {
            "AI Fundamentals": "70%",
            "Data Science Bootcamp": "45%"
        },
        "status": "active"
    }

    return mock_data

def get_deadlines(student_id):
    """Simulate fetching assignment deadlines."""

    print(f"\n🔧 Fetching deadlines for: {student_id}")

    mock_deadlines = [
        {"assignment": "Midterm Project", "due": "2024-02-10", "status": "pending"},
        {"assignment": "Data Analysis Report", "due": "2024-02-15", "status": "pending"},
    ]

    return mock_deadlines

def should_use_tools(question):
    """Check if question needs external data (tools)."""

    question = question.lower()

    # Check for personal data indicators
    if any(word in question for word in ["my", "i am", "me"]):
        return True
    if "student" in question and "id" in question:
        return True
    if "deadline" in question or "progress" in question or "enrolled" in question:
        return True

    return False

def extract_student_id(question):
    """Find student ID in the question."""

    # Look for patterns like STU123, ABC123, etc.
    match = re.search(r'\b([A-Z]{2,4}\d{3,})\b', question.upper())
    return match.group(1) if match else None

# ============================================================
# MAIN: PUT IT ALL TOGETHER
# ============================================================

def main():
    """Run the complete RAG system step by step."""

    print("=" * 60)
    print("  RAG SYSTEM - SIMPLE VERSION")
    print("=" * 60)

    # Step 1: Load documents
    documents = load_documents()

    # Step 2: Chunk documents
    chunks = chunk_documents(documents)

    # Step 3: Build vocabulary
    global vocabulary
    vocabulary = build_vocabulary(chunks)

    # Step 4: Create embeddings
    chunks = create_embeddings(chunks, vocabulary)

    print("\n" + "=" * 60)
    print("  SYSTEM READY! Ask questions about courses.")
    print("=" * 60)

    # Demo questions
    demo_questions = [
        "What is the refund policy?",
        "How do I reset my password?",
        "Tell me about machine learning",
        "What's my progress STU123?"
    ]

    for question in demo_questions:
        print(f"\n{'─' * 60}")
        print(f"❓ QUESTION: {question}")
        print(f"{'─' * 60}")

        # Classify question type
        q_type = classify_question(question)
        print(f"📋 Type: {q_type}")

        # Check if tools needed
        use_tools = should_use_tools(question)
        if use_tools:
            student_id = extract_student_id(question)
            if student_id:
                student_data = get_student_status(student_id)
                deadlines = get_deadlines(student_id)
                print(f"📊 Retrieved student data for: {student_id}")

        # Search for relevant chunks
        relevant = search_chunks(question, chunks)

        if relevant:
            print(f"   ✅ Found {len(relevant)} relevant chunks")

        # Generate answer
        answer = generate_answer(question, relevant)

        print(f"\n💬 ANSWER:")
        print(answer)

    print("\n" + "=" * 60)
    print("  DEMO COMPLETE!")
    print("=" * 60)

# Run the system
if __name__ == "__main__":
    main()
"""
RAG Code Evaluator - GitHub Repository Analysis
================================================

This app reads a GitHub repository URL, analyzes the code using RAG,
and awards marks based on various criteria.

STEP BY STEP:
1. Get GitHub repository link
2. Download all code files from repo
3. Break code into chunks (RAG style)
4. When user asks about code → find relevant chunks
5. Analyze each file and award marks
6. Display final grades
"""

import requests
import re
import math
from collections import defaultdict

# ============================================================
# STEP 1: GET REPOSITORY LINK FROM USER
# ============================================================

def get_github_link():
    """Ask user for GitHub repository URL."""
    print("=" * 60)
    print("  RAG CODE EVALUATOR")
    print("=" * 60)
    print()
    print("This tool will:")
    print("  1. Read a GitHub repository")
    print("  2. Analyze all code files")
    print("  3. Award marks based on quality")
    print()

    repo_link = input("Enter GitHub Repository URL: ").strip()
    return repo_link if repo_link else None

# ============================================================
# STEP 2: EXTRACT INFO FROM LINK
# ============================================================

def extract_repo_info(link):
    """Break the link into username and repo name."""

    # Remove .git from end
    link = link.replace(".git", "")

    # Split by github.com/
    parts = link.split("github.com/")

    if len(parts) < 2:
        return None

    path = parts[1].strip("/").split("/")

    if len(path) < 2:
        return None

    return {
        "username": path[0],
        "repo_name": path[1]
    }

# ============================================================
# STEP 3: DOWNLOAD REPOSITORY FILES
# ============================================================

def download_repo_files(username, repo_name):
    """Download all code files from the GitHub repository."""

    print("\n📥 Downloading repository...")
    print(f"   📦 {username}/{repo_name}")

    # Get file tree from GitHub API
    api_url = f"https://api.github.com/repos/{username}/{repo_name}/git/trees/main?recursive=1"

    try:
        response = requests.get(api_url, timeout=30)

        # Try master branch if main doesn't work
        if response.status_code != 200:
            api_url = f"https://api.github.com/repos/{username}/{repo_name}/git/trees/master?recursive=1"
            response = requests.get(api_url, timeout=30)

        if response.status_code != 200:
            print(f"   ❌ Error: Could not access repository")
            return {}

        data = response.json()
        files = {}

        # Go through each file in repository
        for item in data.get("tree", []):
            if item["type"] == "blob":
                file_path = item["path"]

                # Only process code files
                if is_code_file(file_path):
                    file_url = f"https://raw.githubusercontent.com/{username}/{repo_name}/main/{file_path}"

                    try:
                        file_response = requests.get(file_url, timeout=10)

                        if file_response.status_code == 200:
                            files[file_path] = file_response.text
                    except:
                        pass

        print(f"   ✅ Downloaded {len(files)} code files")
        return files

    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return {}

def is_code_file(file_path):
    """Check if file is a code file we should evaluate."""
    code_extensions = [
        ".py", ".js", ".java", ".cpp", ".c", ".h",
        ".cs", ".go", ".rb", ".php", ".swift", ".kt",
        ".ts", ".jsx", ".tsx", ".html", ".css", ".sql"
    ]

    for ext in code_extensions:
        if file_path.endswith(ext):
            return True

    return False

# ============================================================
# STEP 4: CREATE RAG PIPELINE FOR CODE
# ============================================================

def create_code_chunks(files, chunk_size=500):
    """Break code files into searchable chunks (RAG style)."""

    print("\n✂️ Creating code chunks...")

    chunks = []
    chunk_id = 0

    for file_path, content in files.items():
        # Split by function/class definitions
        patterns = [
            r'def\s+\w+',           # Python function
            r'class\s+\w+',         # Class definition
            r'function\s+\w+',      # JavaScript function
            r'public\s+\w+',        # Java public method
            r'void\s+\w+',          # C/C++ void function
            r'import\s+',           # Import statement
            r'from\s+',             # Python import
            r'#\s*===+',           # Section comments
        ]

        # Simple chunking by lines
        lines = content.split('\n')
        current_chunk = ""

        for line in lines:
            # Skip very long lines
            if len(line) > 200:
                continue

            if len(current_chunk) + len(line) < chunk_size:
                current_chunk += line + "\n"
            else:
                if current_chunk.strip():
                    chunks.append({
                        "content": current_chunk.strip(),
                        "file": file_path,
                        "chunk_id": chunk_id
                    })
                    chunk_id += 1

                current_chunk = line + "\n"

        # Don't forget last chunk
        if current_chunk.strip():
            chunks.append({
                "content": current_chunk.strip(),
                "file": file_path,
                "chunk_id": chunk_id
            })
            chunk_id += 1

    print(f"   ✅ Created {len(chunks)} code chunks")
    return chunks

# ============================================================
# STEP 5: BUILD VOCABULARY AND EMBEDDINGS
# ============================================================

def build_index(chunks):
    """Build searchable index for code chunks."""

    print("\n📖 Building search index...")

    # Count words
    word_count = defaultdict(int)
    for chunk in chunks:
        words = extract_words(chunk["content"])
        for word in set(words):
            word_count[word] += 1

    # Build vocabulary
    vocabulary = {}
    for word, count in word_count.items():
        if count >= 1:
            vocabulary[word] = {
                "count": count,
                "idf": math.log(len(chunks) / (1 + count))
            }

    # Limit size
    vocabulary = dict(list(vocabulary.items())[:500])

    print(f"   ✅ Index ready ({len(vocabulary)} words)")
    return vocabulary

def extract_words(text):
    """Extract words from text."""
    words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text.lower())
    return words

def create_embedding(text, vocabulary):
    """Convert text to vector (simple TF-IDF)."""
    words = extract_words(text)
    word_freq = defaultdict(int)

    for word in words:
        word_freq[word] += 1

    vector = []
    for word, stats in vocabulary.items():
        tf = word_freq.get(word, 0) / max(len(words), 1)
        idf = stats.get("idf", 1)
        vector.append(tf * idf)

    return vector

# ============================================================
# STEP 6: SEARCH RELEVANT CODE CHUNKS
# ============================================================

def search_chunks(query, chunks, vocabulary):
    """Find code chunks relevant to a question."""

    # Create query embedding
    query_words = extract_words(query)
    word_freq = defaultdict(int)

    for word in query_words:
        word_freq[word] += 1

    query_vector = []
    for word, stats in vocabulary.items():
        tf = word_freq.get(word, 0) / max(len(query_words), 1)
        idf = stats.get("idf", 1)
        query_vector.append(tf * idf)

    # Calculate similarity with all chunks
    results = []
    for chunk in chunks:
        chunk_vector = create_embedding(chunk["content"], vocabulary)

        # Cosine similarity
        dot = sum(a * b for a, b in zip(query_vector, chunk_vector))
        mag1 = math.sqrt(sum(a * a for a in query_vector))
        mag2 = math.sqrt(sum(b * b for b in chunk_vector))

        if mag1 > 0 and mag2 > 0:
            similarity = dot / (mag1 * mag2)
            results.append((chunk, similarity))

    # Sort by similarity
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:5]  # Top 5

# ============================================================
# STEP 7: EVALUATE AND AWARD MARKS
# ============================================================

def evaluate_code(file_path, content):
    """Analyze code and award marks out of 100."""

    marks = 0
    max_marks = 100
    reasons = []

    print(f"\n📄 Analyzing: {file_path}")

    # === STRUCTURE (20 marks) ===

    # Has functions/classes
    if re.search(r'(def\s+\w+|class\s+\w+|function\s+\w+|void\s+\w+)', content):
        marks += 8
        reasons.append("✓ Has functions/methods")

    # Has proper indentation
    if re.search(r'^\s{2,}', content, re.MULTILINE):
        marks += 6
        reasons.append("✓ Proper indentation")

    # Organized structure (imports, definitions, logic)
    if re.search(r'(import|from|include|#!/usr/bin)', content):
        marks += 6
        reasons.append("✓ Well organized imports")

    # === DOCUMENTATION (20 marks) ===

    # Has comments
    comment_count = len(re.findall(r'(#|//|/\*|\*|<!--)', content))
    if comment_count > 0:
        marks += 10
        reasons.append(f"✓ Has {comment_count} comments")

    # Has docstrings/documents
    if re.search(r'(""".*?"""|\'\'\'.*?\'\'\'|///|<!>)', content, re.DOTALL):
        marks += 10
        reasons.append("✓ Has docstrings/documentation")

    # === CODE QUALITY (25 marks) ===

    # Variable naming (camelCase or snake_case)
    if re.search(r'[a-z][a-z0-9_]*\s*=', content):
        marks += 6
        reasons.append("✓ Good variable naming")

    # Error handling (try/catch/exception)
    if re.search(r'(try\s*{|except\s*:|catch\s*\(|exception)', content):
        marks += 8
        reasons.append("✓ Has error handling")

    # Uses meaningful variable names
    meaningful_patterns = [
        r'\b(is|has|get|set|calculate|process)\w*',
        r'\b(count|total|sum|avg|min|max)\w*',
        r'\b(user|name|email|date|status)\w*'
    ]
    meaningful_count = sum(1 for p in meaningful_patterns if re.search(p, content, re.IGNORECASE))
    if meaningful_count >= 2:
        marks += 6
        reasons.append("✓ Uses descriptive names")
    elif meaningful_count >= 1:
        marks += 4

    # Constants defined
    if re.search(r'(CONST|MAX|MIN|TOTAL|DEFAULT)', content):
        marks += 5
        reasons.append("✓ Has constants defined")

    # === DEPENDENCIES (15 marks) ===

    # Has imports/dependencies
    if re.search(r'(import\s+|require\s*|from\s+|include\s+)', content):
        marks += 8
        reasons.append("✓ Proper module imports")

    # Package management (requirements.txt, package.json, etc.)
    if re.search(r'(requirements|package\.json|pom\.xml|Gemfile)', file_path):
        marks += 7
        reasons.append("✓ Has dependency management")

    # === TESTING (10 marks) ===

    # Has test files
    if 'test' in file_path.lower() or '_test.' in file_path or '.test.' in file_path:
        marks += 10
        reasons.append("✓ Has test files")

    # === SECURITY (10 marks) ===

    # Avoids hardcoded secrets (basic check)
    if not re.search(r'(password\s*=|api_key\s*=|secret\s*=)', content, re.IGNORECASE):
        marks += 5
        reasons.append("✓ No hardcoded secrets")

    # Uses validation/sanitization
    if re.search(r'(validate|sanitize|escape|encode|decode)', content, re.IGNORECASE):
        marks += 5
        reasons.append("✓ Has input validation")

    return marks, reasons

# ============================================================
# STEP 8: DISPLAY FINAL RESULTS
# ============================================================

def display_results(evaluations):
    """Show final evaluation results."""

    print("\n" + "=" * 60)
    print("  📋 EVALUATION RESULTS")
    print("=" * 60)

    total_score = 0
    total_max = 0

    for file_path, (marks, reasons) in evaluations:
        percentage = (marks / 100) * 100
        grade = get_grade(percentage)

        print(f"\n📄 {file_path}")
        print(f"   Score: {marks}/100 ({percentage:.0f}%) - Grade: {grade}")

        for reason in reasons[:5]:
            print(f"   {reason}")

        total_score += marks
        total_max += 100

    # Overall summary
    overall_percentage = (total_score / total_max) * 100 if total_max > 0 else 0
    overall_grade = get_grade(overall_percentage)

    print("\n" + "-" * 60)
    print(f"\n🏆 OVERALL SCORE: {total_score}/{total_max} ({overall_percentage:.1f}%)")
    print(f"📊 OVERALL GRADE: {overall_grade}")
    print(f"📁 FILES EVALUATED: {len(evaluations)}")

    return overall_percentage, overall_grade

def get_grade(percentage):
    """Convert percentage to grade."""
    if percentage >= 90:
        return "A+ (Excellent)"
    elif percentage >= 80:
        return "A (Very Good)"
    elif percentage >= 70:
        return "B+ (Good)"
    elif percentage >= 60:
        return "B (Average)"
    elif percentage >= 50:
        return "C (Below Average)"
    else:
        return "D (Needs Improvement)"

# ============================================================
# STEP 9: RAG-STYLE Q&A ABOUT REPO
# ============================================================

def answer_about_repo(question, chunks, vocabulary):
    """Use RAG to answer questions about the repository."""

    print(f"\n❓ Question: {question}")

    # Search relevant chunks
    results = search_chunks(question, chunks, vocabulary)

    if results:
        print(f"   ✅ Found {len(results)} relevant code sections")
        print("\n   📝 Relevant code:")
        for chunk, score in results[:3]:
            preview = chunk["content"][:150] + "..." if len(chunk["content"]) > 150 else chunk["content"]
            print(f"\n   [{chunk['file']}] (relevance: {score:.2f})")
            print(f"   {preview}")
    else:
        print("   ❌ No relevant code found")

# ============================================================
# MAIN PROGRAM
# ============================================================

def main():
    """Run the complete RAG Code Evaluator."""

    # Step 1: Get repository link
    repo_link = get_github_link()

    if not repo_link or "github.com" not in repo_link:
        print("❌ Invalid GitHub URL!")
        return

    # Step 2: Extract info
    repo_info = extract_repo_info(repo_link)

    if not repo_info:
        print("❌ Could not parse repository URL!")
        return

    print(f"\n📦 Repository: {repo_info['username']}/{repo_info['repo_name']}")

    # Step 3: Download files
    files = download_repo_files(repo_info["username"], repo_info["repo_name"])

    if not files:
        print("❌ No code files found!")
        return

    # Step 4: Create chunks
    chunks = create_code_chunks(files)

    # Step 5: Build index
    vocabulary = build_index(chunks)

    # Step 6: Evaluate all files
    print("\n" + "=" * 60)
    print("  🔍 EVALUATING CODE")
    print("=" * 60)

    evaluations = {}
    for file_path, content in files.items():
        marks, reasons = evaluate_code(file_path, content)
        evaluations[file_path] = (marks, reasons)

    # Step 7: Display results
    display_results(evaluations)

    # Step 8: Demo RAG Q&A
    print("\n" + "=" * 60)
    print("  💬 ASK ABOUT THE CODE (RAG)")
    print("=" * 60)

    demo_questions = [
        "How does the code handle errors?",
        "What is the main functionality?",
        "What dependencies are used?"
    ]

    for question in demo_questions:
        answer_about_repo(question, chunks, vocabulary)

    print("\n" + "=" * 60)
    print("  ✅ EVALUATION COMPLETE!")
    print("=" * 60)

# Run the program
if __name__ == "__main__":
    main()
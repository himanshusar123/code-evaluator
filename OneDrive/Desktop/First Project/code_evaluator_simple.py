"""
Code Evaluator - Simple Human-like Execution
============================================

This app reads a GitHub repository and evaluates code by awarding marks
based on various criteria like structure, documentation, testing, etc.
"""

import requests
import os
import re
from collections import Counter

# ============================================================
# STEP 1: GET REPO LINK FROM USER
# ============================================================

def get_github_link():
    """Ask user for GitHub repository URL."""
    print("=" * 60)
    print("  GITHUB CODE EVALUATOR")
    print("=" * 60)
    print()

    repo_link = input("Enter GitHub Repository URL: ").strip()

    # Validate the link
    if not repo_link:
        print("Error: No URL provided!")
        return None

    if "github.com" not in repo_link:
        print("Error: Please provide a valid GitHub URL!")
        return None

    return repo_link

# ============================================================
# STEP 2: EXTRACT INFO FROM GITHUB LINK
# ============================================================

def extract_repo_info(link):
    """Break the link into username and repo name."""

    # Remove .git from end if present
    link = link.replace(".git", "")

    # Split by github.com/
    parts = link.split("github.com/")

    if len(parts) < 2:
        print("Error: Invalid GitHub URL format!")
        return None

    # Get username/repo
    path = parts[1]
    path_parts = path.strip("/").split("/")

    if len(path_parts) < 2:
        print("Error: Please provide full repository URL!")
        return None

    username = path_parts[0]
    repo_name = path_parts[1]

    print(f"\n📦 Repository: {username}/{repo_name}")

    return {
        "username": username,
        "repo_name": repo_name
    }

# ============================================================
# STEP 3: DOWNLOAD FILES FROM GITHUB
# ============================================================

def download_repo_files(username, repo_name):
    """Download all code files from the repository."""

    print("\n📥 Downloading repository...")

    # API URL to get file tree
    api_url = f"https://api.github.com/repos/{username}/{repo_name}/git/trees/main?recursive=1"

    try:
        response = requests.get(api_url, timeout=30)

        if response.status_code != 200:
            # Try 'master' branch
            api_url = f"https://api.github.com/repos/{username}/{repo_name}/git/trees/master?recursive=1"
            response = requests.get(api_url, timeout=30)

        if response.status_code != 200:
            print(f"Error: Could not access repository (Status: {response.status_code})")
            return {}

        data = response.json()

        if "tree" not in data:
            print("Error: Could not read repository structure!")
            return {}

        files = {}

        # Go through each file in repository
        for item in data["tree"]:
            # Only process code files
            if item["type"] == "blob":
                file_path = item["path"]

                # Check if it's a code file
                if is_code_file(file_path):
                    # Get file content
                    file_url = f"https://raw.githubusercontent.com/{username}/{repo_name}/main/{file_path}"

                    try:
                        file_response = requests.get(file_url, timeout=10)

                        if file_response.status_code == 200:
                            files[file_path] = {
                                "content": file_response.text,
                                "size": len(file_response.text)
                            }
                    except:
                        pass

        print(f"   ✅ Found {len(files)} code files")
        return files

    except Exception as e:
        print(f"Error downloading: {str(e)}")
        return {}

# ============================================================
# STEP 4: CHECK FILE TYPE
# ============================================================

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

    # Also check for common code filenames
    code_names = ["Makefile", "Dockerfile", "README", "package.json"]
    for name in code_names:
        if file_path.endswith(name):
            return True

    return False

# ============================================================
# STEP 5: ANALYZE EACH FILE
# ============================================================

def analyze_file(file_path, content):
    """Analyze a single file and give marks."""

    marks = 0
    reasons = []

    # Check 1: File has content
    if len(content) > 0:
        marks += 2
        reasons.append(f"✓ File has content ({len(content)} chars)")

    # Check 2: Has comments or documentation
    comment_indicators = ["#", "//", "/*", "'''", '"""', "<!--"]
    has_comments = any(indicator in content for indicator in comment_indicators)

    if has_comments:
        marks += 3
        reasons.append("✓ Has comments/documentation")

    # Check 3: Has function/class definitions
    if re.search(r'(def |class |function |public |private |void )', content):
        marks += 3
        reasons.append("✓ Has functions/classes")

    # Check 4: Has proper naming
    if re.search(r'^[a-z][a-z0-9_]*\s*=', content, re.MULTILINE):
        marks += 2
        reasons.append("✓ Uses proper variable naming")

    # Check 5: Has error handling
    if "try" in content or "catch" in content or "exception" in content.lower():
        marks += 3
        reasons.append("✓ Has error handling")

    # Check 6: Has imports/dependencies
    if re.search(r'(import |require |from |include )', content):
        marks += 2
        reasons.append("✓ Has imports/dependencies")

    # Check 7: Check indentation (proper code structure)
    if re.search(r'^\s{4}', content, re.MULTILINE):
        marks += 2
        reasons.append("✓ Proper indentation")

    # Check 8: Has docstrings (for Python)
    if file_path.endswith(".py") and ('"""' in content or "'''" in content):
        marks += 3
        reasons.append("✓ Has docstrings")

    return marks, reasons

# ============================================================
# STEP 6: EVALUATE ENTIRE REPOSITORY
# ============================================================

def evaluate_repository(files):
    """Go through all files and calculate total marks."""

    print("\n📊 EVALUATING CODE...")
    print("-" * 50)

    total_marks = 0
    max_marks = 0
    file_scores = {}

    for file_path, file_data in files.items():
        marks, reasons = analyze_file(file_path, file_data["content"])

        file_scores[file_path] = {
            "marks": marks,
            "reasons": reasons,
            "size": file_data["size"]
        }

        total_marks += marks
        max_marks += 20  # Each file can get max 20 marks

    return file_scores, total_marks, max_marks

# ============================================================
# STEP 7: DISPLAY RESULTS
# ============================================================

def display_results(file_scores, total_marks, max_marks):
    """Show the evaluation results to user."""

    print("\n" + "=" * 60)
    print("  📋 EVALUATION RESULTS")
    print("=" * 60)

    # Sort files by marks
    sorted_files = sorted(file_scores.items(), key=lambda x: x[1]["marks"], reverse=True)

    print("\n📁 FILE-WISE MARKS:\n")

    for file_path, scores in sorted_files:
        grade = get_grade(scores["marks"])
        print(f"  {file_path}")
        print(f"    Marks: {scores['marks']}/20 - Grade: {grade}")

        # Show reasons
        for reason in scores["reasons"][:5]:  # Show top 5 reasons
            print(f"    {reason}")
        print()

    # Overall summary
    print("-" * 50)
    percentage = (total_marks / max_marks * 100) if max_marks > 0 else 0
    overall_grade = get_grade(total_marks / len(file_scores) if file_scores else 0)

    print(f"\n🏆 OVERALL SCORE: {total_marks}/{max_marks} ({percentage:.1f}%)")
    print(f"📊 OVERALL GRADE: {overall_grade}")
    print(f"📁 TOTAL FILES EVALUATED: {len(file_scores)}")

def get_grade(marks):
    """Convert marks to grade."""
    if marks >= 18:
        return "A+ (Excellent)"
    elif marks >= 15:
        return "A (Very Good)"
    elif marks >= 12:
        return "B+ (Good)"
    elif marks >= 9:
        return "B (Average)"
    elif marks >= 6:
        return "C (Below Average)"
    else:
        return "D (Needs Improvement)"

# ============================================================
# STEP 8: MAIN PROGRAM
# ============================================================

def main():
    """Run the entire evaluation process."""

    # Step 1: Get link
    repo_link = get_github_link()

    if repo_link is None:
        return

    # Step 2: Extract info
    repo_info = extract_repo_info(repo_link)

    if repo_info is None:
        return

    # Step 3: Download files
    files = download_repo_files(
        repo_info["username"],
        repo_info["repo_name"]
    )

    if not files:
        print("No code files found in repository!")
        return

    # Step 4: Evaluate
    file_scores, total_marks, max_marks = evaluate_repository(files)

    # Step 5: Show results
    display_results(file_scores, total_marks, max_marks)

    print("\n" + "=" * 60)
    print("✅ Evaluation Complete!")
    print("=" * 60)

# Run the program
if __name__ == "__main__":
    main()
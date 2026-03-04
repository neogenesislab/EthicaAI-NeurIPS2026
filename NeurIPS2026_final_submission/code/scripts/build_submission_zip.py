"""
Build a NeurIPS-compliant submission ZIP archive.
=================================================
- Uses forward-slash paths (/) for cross-platform compatibility
- Single root directory (no duplicate version folders)
- Excludes __pycache__, .pyc, .git, .DS_Store, etc.
- Verifies paper/unified_paper.pdf exists before packaging
"""
import zipfile
import os
import sys
from pathlib import Path

# Configuration
SUBMISSION_DIR = Path(__file__).resolve().parent.parent.parent  # NeurIPS2026_final_submission/
OUTPUT_ZIP = SUBMISSION_DIR.parent / "NeurIPS2026_EthicaAI_Final_Submission.zip"
ROOT_NAME = "NeurIPS2026_final_submission"

# Patterns to exclude
EXCLUDE_PATTERNS = {
    "__pycache__", ".pyc", ".pyo", ".git", ".gitignore",
    ".DS_Store", "Thumbs.db", ".env", "*.egg-info",
    "build_submission_zip.py",  # Don't include self
    "finalize.ps1",  # Build script, not submission content
    "verification_report.txt", "verification_report_utf8.txt",
}

EXCLUDE_DIRS = {"__pycache__", ".git", "node_modules", ".idea", ".vscode"}


def should_exclude(path: Path) -> bool:
    """Check if a file/directory should be excluded."""
    parts = path.parts
    for part in parts:
        if part in EXCLUDE_DIRS:
            return True
    name = path.name
    if name in EXCLUDE_PATTERNS:
        return True
    for pattern in EXCLUDE_PATTERNS:
        if name.endswith(pattern.lstrip("*")):
            return True
    return False


def build_zip():
    # Pre-flight check
    pdf_path = SUBMISSION_DIR / "paper" / "unified_paper.pdf"
    if not pdf_path.exists():
        print(f"[FATAL] PDF not found: {pdf_path}")
        print("  Run pdflatex first!")
        sys.exit(1)
    
    print(f"Building submission ZIP: {OUTPUT_ZIP}")
    print(f"Source: {SUBMISSION_DIR}")
    
    file_count = 0
    total_size = 0
    
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(SUBMISSION_DIR):
            # Filter out excluded directories in-place
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            
            for fname in sorted(files):
                fpath = Path(root) / fname
                rel_path = fpath.relative_to(SUBMISSION_DIR)
                
                if should_exclude(rel_path):
                    continue
                
                # Use forward-slash archive name with single root
                arcname = f"{ROOT_NAME}/{rel_path.as_posix()}"
                zf.write(fpath, arcname)
                
                fsize = fpath.stat().st_size
                file_count += 1
                total_size += fsize
    
    zip_size = OUTPUT_ZIP.stat().st_size
    print(f"\n[OK] ZIP created successfully!")
    print(f"  Files: {file_count}")
    print(f"  Uncompressed: {total_size / 1024 / 1024:.1f} MB")
    print(f"  Compressed: {zip_size / 1024 / 1024:.1f} MB")
    print(f"  Output: {OUTPUT_ZIP}")
    
    # Verify ZIP structure
    print(f"\n--- ZIP Structure Verification ---")
    with zipfile.ZipFile(OUTPUT_ZIP, "r") as zf:
        names = zf.namelist()
        # Check all paths use forward slashes
        backslash_count = sum(1 for n in names if "\\" in n)
        if backslash_count > 0:
            print(f"  [FAIL] {backslash_count} paths contain backslashes!")
        else:
            print(f"  [PASS] All {len(names)} paths use forward slashes")
        
        # Check single root
        roots = set(n.split("/")[0] for n in names if "/" in n)
        if len(roots) == 1 and ROOT_NAME in roots:
            print(f"  [PASS] Single root directory: {ROOT_NAME}")
        else:
            print(f"  [FAIL] Multiple roots detected: {roots}")
        
        # Check no __pycache__
        pycache_count = sum(1 for n in names if "__pycache__" in n)
        if pycache_count == 0:
            print(f"  [PASS] No __pycache__ files")
        else:
            print(f"  [FAIL] {pycache_count} __pycache__ files found")
        
        # Check PDF exists
        pdf_in_zip = any("unified_paper.pdf" in n for n in names)
        if pdf_in_zip:
            print(f"  [PASS] unified_paper.pdf included")
        else:
            print(f"  [FAIL] unified_paper.pdf NOT found in ZIP!")


if __name__ == "__main__":
    build_zip()

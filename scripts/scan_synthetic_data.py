#!/usr/bin/env python3
"""
Scan & Remove Synthetic Data - Implementation of SCAN_AND_REMOVE_SYNTHETIC.md runbook
"""
import os
import re
import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

class SyntheticDataScanner:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.issues_found = []
        
        self.synthetic_patterns = [
            (r'np\.random\.(normal|uniform|randint|choice)', 'Random number generation'),
            (r'random\.(choice|randint|uniform|gauss)', 'Random data generation'),
            (r'sample_data\s*=\s*\[', 'Sample data creation'),
            (r'for.*range.*:\s*.*random', 'Loop with random generation'),
        ]
        
        self.exclude_patterns = [r'test_.*\.py$', r'.*_test\.py$', r'\.git/', r'__pycache__/', r'\.venv/', r'site-packages/', r'scripts/scan_synthetic_data\.py']
    
    def should_scan_file(self, filepath: Path) -> bool:
        if filepath.suffix != '.py':
            return False
        filepath_str = str(filepath).replace('\\', '/')
        # Exclude .venv and site-packages
        if '.venv' in filepath_str or 'site-packages' in filepath_str:
            return False
        for pattern in self.exclude_patterns:
            if re.search(pattern, filepath_str):
                return False
        return True
    
    def scan_file(self, filepath: Path) -> List[Dict]:
        issues = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for line_num, line in enumerate(lines, 1):
                for pattern, description in self.synthetic_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        issues.append({
                            'file': str(filepath.relative_to(self.project_root)),
                            'line': line_num,
                            'code': line.strip(),
                            'issue': description
                        })
        except Exception as e:
            print(f"Error scanning {filepath}: {e}")
        return issues
    
    def scan_project(self) -> List[Dict]:
        print(f"Scanning project: {self.project_root}")
        all_issues = []
        for filepath in self.project_root.rglob('*.py'):
            if self.should_scan_file(filepath):
                issues = self.scan_file(filepath)
                all_issues.extend(issues)
        self.issues_found = all_issues
        return all_issues
    
    def generate_report(self) -> str:
        if not self.issues_found:
            return "SUCCESS: No synthetic data patterns detected!"
        
        report = [f"\nSYNTHETIC DATA DETECTED: {len(self.issues_found)} issues\n"]
        by_file = {}
        for issue in self.issues_found:
            file = issue['file']
            if file not in by_file:
                by_file[file] = []
            by_file[file].append(issue)
        
        for file, issues in sorted(by_file.items()):
            report.append(f"\nFile: {file}")
            for issue in issues:
                report.append(f"  Line {issue['line']}: {issue['issue']}")
                report.append(f"    {issue['code']}")
        return "\n".join(report)

def main():
    import sys
    if sys.platform == 'win32':
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    
    project_root = Path(__file__).parent.parent
    scanner = SyntheticDataScanner(str(project_root))
    issues = scanner.scan_project()
    report = scanner.generate_report()
    print(report)
    
    if issues:
        print(f"\nERROR: Found {len(issues)} synthetic data issues")
        sys.exit(1)
    else:
        print("\nSUCCESS: No synthetic data issues found")
        sys.exit(0)

if __name__ == "__main__":
    main()

import os

file_paths = [
    r"C:\Users\forin\OneDrive\문서\20240820 학적 생활기록 도움 챗봇\files\초중등교육법_시행령.txt",
    r"C:\Users\forin\OneDrive\문서\20240820 학적 생활기록 도움 챗봇\files\2023_학생부_종합지원센터_질의_회신사례집_utf8.txt"
]

for path in file_paths:
    if os.path.exists(path):
        print(f"File exists: {path}")
    else:
        print(f"File does NOT exist: {path}")

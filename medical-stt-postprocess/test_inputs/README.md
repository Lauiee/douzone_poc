# test_inputs

여기에 테스트 원문 `*.txt` 파일을 넣으면 됩니다.

- 파일 1개 = 리포트의 섹션 1개
- 파일명(확장자 제외)이 섹션 제목으로 사용됩니다.  
  예: `type1.txt` -> `## TYPE1`

## 실행

프로젝트 루트에서:

```bash
python3 scripts/folder_before_after_report.py
```

기본 출력 파일:

- `full_before_after_comparison.md`

옵션으로 폴더/출력 경로 지정:

```bash
python3 scripts/folder_before_after_report.py --input-dir test_inputs --output full_before_after_comparison.md
```

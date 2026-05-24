# README Enhancement Implementation Plan

> **For agentic workers:** Execute task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mở rộng README.md từ 78 dòng (basic install/usage) thành README đầy đủ ~300 dòng phục vụ đồ án môn học (methodology, results, references) + GitHub portfolio (badges, screenshots, demo).

**Architecture:** Re-write README.md theo 15 sections, giữ tiếng Việt làm chính (terms kỹ thuật để tiếng Anh). Reuse nội dung từ README cũ + spec + paper SH17. Tạo `docs/images/` cho 3-4 screenshot (user tự chụp).

**Tech Stack:** Markdown, shields.io badges, không thêm dependency.

**Working directory:** `/Users/tranquangtrong/Desktop/ppe-checker/`

**Reference plan mode plan:** `/Users/tranquangtrong/.claude/plans/nh-ng-m-c-c-tham-fluttering-candy.md`

---

## File Structure

```
ppe-checker/                                          (existing)
├── README.md                                         Task 2 (re-write)
└── docs/
    └── images/                                       Task 1 (new folder)
        ├── demo-detection.png                        Task 1 (user-provided)
        ├── ui-sidebar.png                            Task 1 (user-provided)
        └── compliance-table.png                      Task 1 (user-provided)
```

---

## Task 1: Setup docs/images folder + capture screenshots

**Files:**
- Create: `/Users/tranquangtrong/Desktop/ppe-checker/docs/images/.gitkeep`
- Manual (user): 3-4 PNG screenshots

- [ ] **Step 1: Create folder + .gitkeep placeholder**

```bash
cd /Users/tranquangtrong/Desktop/ppe-checker
mkdir -p docs/images
touch docs/images/.gitkeep
```

- [ ] **Step 2: User captures screenshots manually**

Run app:
```bash
streamlit run app.py
```

Capture (Cmd+Shift+4 on macOS):
- `docs/images/demo-detection.png` — Full UI with `samples/sample_01.jpg` uploaded, default settings (HEAD + BODY checked), showing annotated image + compliance table
- `docs/images/ui-sidebar.png` — Close-up sidebar with 2 sliders + 4 checkboxes + About expander open
- `docs/images/compliance-table.png` — Close-up bảng compliance ở góc phải

- [ ] **Step 3: Commit folder structure**

```bash
git add docs/images/.gitkeep
git commit -m "docs: add docs/images/ folder for README screenshots"
```

---

## Task 2: Write new README.md (all 15 sections)

**Files:**
- Modify: `/Users/tranquangtrong/Desktop/ppe-checker/README.md` (full re-write)

- [ ] **Step 1: Write new README content**

(Full content in single Write call — see Task 2 implementation below)

- [ ] **Step 2: Verify markdown renders correctly**

```bash
wc -l README.md
```
Expected: ~250-350 lines

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: comprehensive README with methodology, results, and references"
```

---

## Task 3: Final verification

- [ ] **Step 1: Verify all unit tests still pass**

```bash
cd /Users/tranquangtrong/Desktop/ppe-checker
source .venv/bin/activate
pytest tests/ -v
```
Expected: 14 passed

- [ ] **Step 2: Check git state**

```bash
git log --oneline | head -10
```

---

## Notes

- Originally planned 6 tasks, consolidated to 3 since the README is logically one document and splitting commits per section would fragment git history unnecessarily.
- Author info uses placeholder `[Tên sinh viên]` — user fills in before submission.
- Screenshots reference paths that won't exist until user captures them — README will have broken image links until then. Acceptable trade-off for keeping README structure.

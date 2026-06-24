# Publishing this repository and minting a citable DOI

A one-time walkthrough: get the code onto GitHub cleanly, then archive a release on Zenodo to get a DOI you can cite in the thesis. Commands assume macOS with `git` installed; a web-UI path is given where the CLI is optional.

---

## 0. Prerequisites

- A free [GitHub](https://github.com) account.
- A free [Zenodo](https://zenodo.org) account (sign in **with GitHub** — it makes the link automatic).
- Optional but easiest: the GitHub CLI, `brew install gh` then `gh auth login`.

## 1. Assemble the repo root

Use the **`Master_Thesis_Reproduction_No_TFRecords/`** folder as your repository root — it is self-contained and reproduces every headline number. Copy the scaffold files from `wealth-from-above-github/` into it:

```bash
cd "/Users/danielpazrossebo/Downloads/Master_Thesis_Reproduction_No_TFRecords"
cp ../wealth-from-above-github/README.md .
cp ../wealth-from-above-github/LICENSE .
cp ../wealth-from-above-github/.gitignore .
cp ../wealth-from-above-github/requirements.txt .
cp ../wealth-from-above-github/CITATION.cff .
cp ../wealth-from-above-github/.zenodo.json .
cp ../wealth-from-above-github/DATA.md .
cp ../wealth-from-above-github/PUBLISHING.md .
```

(Your existing `README.txt` can stay — it's useful internal documentation — or delete it now that `README.md` supersedes it.)

## 2. Scrub restricted data — do this BEFORE the first commit

```bash
git init
git add -A
git ls-files | grep -E '^data/|\.csv$' | grep -v '^data/README.md$' || echo "clean — no data staged"
```

If that prints `clean`, continue. If it lists anything, the `.gitignore` isn't catching it — remove it from staging (`git rm --cached <file>`) and re-check. See **DATA.md**. Git history is permanent, so never let restricted files into even the first commit.

## 3. First commit

```bash
git commit -m "Initial public release: Wealth from Above replication code"
git branch -M main
```

## 4. Create the GitHub repo and push

**With the GitHub CLI:**

```bash
gh repo create wealth-from-above --public --source=. --remote=origin \
  --description "Predicting sub-Saharan asset wealth from satellite imagery (MSc thesis replication)" --push
```

**Or via the website:** create a new empty public repo named `wealth-from-above` (no README/license — you already have them), then:

```bash
git remote add origin https://github.com/USERNAME/wealth-from-above.git
git push -u origin main
```

Then fill in the repo **description** and **topics** (e.g. `poverty-mapping`, `remote-sensing`, `deep-learning`, `development-economics`) on the repo page.

## 5. Turn on Zenodo archiving — BEFORE you cut the release

Order matters: Zenodo only archives releases created *after* you enable the repo.

1. Go to [zenodo.org/account/settings/github](https://zenodo.org/account/settings/github) (sign in with GitHub).
2. Find `USERNAME/wealth-from-above` in the list and flip the toggle **ON**. (If it's not listed, click *Sync* / re-authorize.)

## 6. Cut the release → DOI is minted automatically

On the GitHub repo page: **Releases → Draft a new release**. Tag `v1.0.0`, title "v1.0.0 — thesis submission", publish. (CLI: `gh release create v1.0.0 --title "v1.0.0 — thesis submission" --notes "Replication code accompanying the MSc thesis."`)

Within a minute Zenodo creates a record and a DOI. Find it back at [zenodo.org/account/settings/github](https://zenodo.org/account/settings/github) → your repo → the DOI badge.

Zenodo gives you **two** DOIs:

- a **version DOI** (this exact release), and
- a **concept DOI** ("all versions", always resolves to the latest) — **cite the concept DOI in the thesis.**

## 7. Wire the DOI back in

```bash
# In README.md: replace PASTE-YOUR-DOI-HERE (badge URL + link) with your concept DOI.
# In CITATION.cff: uncomment and set `doi:` and set `repository-code:`.
# In .zenodo.json: optional — add a related identifier if you like.
git add README.md CITATION.cff .zenodo.json
git commit -m "Add Zenodo DOI"
git push
```

Then add a line to the thesis (you already have a Data Access Statement in §9.1):

> The full replication code is openly available at `github.com/USERNAME/wealth-from-above` and archived at `https://doi.org/<concept-DOI>`.

## 8. Optional polish

- Add your **ORCID** to `CITATION.cff` and `.zenodo.json`.
- Commit a `requirements.lock.txt` (`pip freeze`) for exact versions.
- Add a short `data/README.md` pointing users to DATA.md for obtaining DHS inputs.

---

**Result:** a public, MIT-licensed, citable repository with a permanent DOI — converting the thesis's "code available on request" into "openly archived and reproducible," which is exactly the open, low-cost, replicable measurement the thesis argues for.

# auto-rec-sys Agent Instructions

Read `program.md` before changing this workspace.

- Use `orchestrator.md` for the principal-investigator loop and `agents/*.md`
  for focused worker briefs.
- Treat `modal_app.py` and `prepare.py` as a starting Modal research workbench.
  Add the smallest new Modal function that answers the next concrete data,
  training, evaluation, or inspection question.
- Use Modal runs, run manifests, dataset IDs, and checkpoint artifacts as
  experiment lineage. Do not use git commit/revert cycles as the research
  ledger.
- Keep raw snapshots immutable. A new data-cleaning hypothesis creates a new
  dataset version and a new result row.
- Keep data execution Modal-native. Codex should receive summaries, small
  samples, metrics, and artifact paths rather than moving raw training data
  locally.
- Do not mutate production DB data. Research DB access is for snapshot reads.
- Gate expensive LLM training on prepared data plus a successful semantic-ID
  assignment result.
- Keep workers focused on one hypothesis and one stage family unless the
  orchestrator explicitly assigns an end-to-end baseline.

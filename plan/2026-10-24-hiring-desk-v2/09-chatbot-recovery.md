# Chatbot recovery after ranking setup

Owner reports chatbot failure after SBERT restoration. Continue current v2 branch.

## Diagnosis and authorized recovery

- Main pyproject/lock and isolated QA/GLiNER runtime configs have no SBERT-install
  changes. The incompatible ML/ONNX installs failed. Restored SBERT/role artifacts
  and inference parts are ranking caches, separate from chatbot model caches.
- Current local Qwen endpoint answers200. Native invented-source QA returns the
  supported degree span; GLiNER returns Python/degree mentions. Keep these logs.
- Live portal runs v1, not v2, so Qwen comparison composition is not attached.
- A separate trimmed-menu UI change leaves a stale remove('Compare candidates')
  on a menu that no longer contains that action; 0/1-candidate reviews can fail.

Add synthetic UI regression showing that failure, guard the conditional removal
without restoring intentionally removed menu items, and verify v2 chat tests.
Run the existing invented-source native walkthrough, then restart only the portal
on its existing port with v2 enabled and Qwen not disabled, preserving its ranking
adapter, pool, accounts, data and database configuration. Do not clear transcripts,
entity caches, accounts or real JDs, and do not claim unfinished ONNX ranking works.

Independent review and final verification are recorded below. No training or
dependency migration is part of this recovery.

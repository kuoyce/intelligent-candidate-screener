# Initial audit and follow-up checklist

Method: source inspection of the current portal modules and tests. This is not a
browser visual audit or an Impeccable tool result. Impeccable was not installed
in the available skill catalog or the inspected skill/plugin directories.

| Finding | Impact | Planned fix |
|---|---|---|
| Job selection is a dropdown and disappears once a session opens | JD context and job switching are hard to discover | Persistent job-card browser, full-JD panel and Review another job |
| Full source JD exists in snapshots but is not central to review | Reviewer sees actions without enough requirements context | Summary and full redacted source above candidate actions |
| Workspace selector exposes synthetic demo in normal use | Competes with recruiter navigation | Remove selector; keep isolated test fixtures |
| Repeated technical disclaimers and long outputs compete with actions | Information hierarchy is weak | Concise source/status labels, service-details panel and readable sections |
| Comparison UI serializes entity values as JSON strings | Dense content is difficult to scan | Human-readable lists, wrapping and category-based comparison |
| Send action is a placeholder; package supports eight CVs | Ten-applicant procedure cannot be demonstrated end-to-end | Versioned ten-CV package, recipient preview and local MIME capture |
| Alert storage has no connected producer | Alerts cannot reflect normal application work | Transactional JD/email events and unread/deep-link UI |
| Footer positioning relies on Streamlit DOM CSS | Visual placement can regress across versions | Verify desktop/mobile/zoom and scope selectors |

After approval, inspect all screens in browser for hierarchy, contrast, font size,
line length, spacing, focus order, target size, colour-only indicators, loading,
disabled actions, empty states and horizontal overflow. Record severity, page,
evidence, change and recheck outcome. Do not mark visual acceptance based solely
on AppTest. Use synthetic documents for screenshots to avoid committing PII.

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

## Implemented audit findings — 24 October

The JD browser now persists above review actions, with search, selected-state
labels and previous/next navigation. Desktop displays multiple cards; narrow
screens display the selected card with the same navigation controls. Full JD text
uses a bounded readable panel. Entity comparisons use readable mention lists.
The production demo selector is removed. Login, saved review, authoring, alerts,
source-unavailable states and capture previews retain explicit labels/actions.

Browser inspection identified chat auto-scroll obscuring the initial JD and an
unanchored sidebar footer. A separately scrollable inline chat and a keyed
sidebar flex container corrected both. The browser then verified JD switching,
authoring/alert navigation, a 390px mobile viewport and 200% root text scaling
without document overflow. Screenshots use invented fixtures and stay ignored:
`logs/hiring-desk-v1-desktop.png`, `hiring-desk-v1-authoring.png`,
`hiring-desk-v1-mobile.png`, and `hiring-desk-v1-text-scaling.png`.
The old shell is captured in `logs/hiring-desk-browser-diagnostic.png`.

Theme colors and caption styles improve contrast; controls use visible labels,
keyboard-native widgets and focus outlines. This is a manual/source/browser audit,
not an Impeccable tool run or certification of complete WCAG conformance.

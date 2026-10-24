# Implementation record — 23 October 2026

Implemented the approved 960-hour provisioning default and maximum. Local-token
principals now expire with their account rather than after one hour. Explicit
shorter account lifetimes, disabled-account checks and version-based revocation
remain in effect. The CLI retains `--hours`.

Extended the existing local reviewer account under its provisioning lock using
an atomic replacement. Its expiry is 2 December 2026, 13:10:54 UTC; every field
other than expiry was verified unchanged. Existing sign-ins must authenticate
again because account-version revocation includes expiry changes.

Removed Sign out from the navigation dropdown. A separate Log out button is
rendered in the sidebar footer before page-specific early returns, with flex
layout to push it to the bottom. It clears the same review/authentication state
and reruns. Browser-level placement has not been visually verified.

## Validation

- Regression run before implementation failed the four changed/new invariants
  covering account/session expiry, explicit 960-hour provisioning and logout.
  Log: `logs/reviewer-auth-before.log`.
- Focused authentication/UI tests: 25 passed.
  Log: `logs/reviewer-auth-focused.log`.
- Full suite: 1,287 passed, 10 skipped, two baseline golden-test failures reporting
  vocabulary/model drift. No baseline/model code was changed by this task.
  Log: `logs/reviewer-auth-pytest.log`.
- The default checks runner selected unrelated records because of pre-existing
  working-tree changes; stopped that run after missing dataset/model errors.
  Log: `logs/reviewer-auth-checks.log`.
- The runner's `affected` dependency selector applied to this task's files alone
  returns `{}`: no committed model/evaluation records depend on these changes.
  Log: `logs/reviewer-auth-check-selection.log`.
- Scoped whitespace check passes. Independent sub-agent review checked the
  design and highlighted revocation and footer placement considerations.

This record documents implementation and validation limits; it does not mark the
plan complete or assert cold reproducibility under D55. No model training,
final-set scoring or research-record changes were needed for this task.

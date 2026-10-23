# Reviewer authentication: 40-day validity

Status: approved 23 October 2026, including a separate logout button at the
bottom of the left sidebar.

## Requirement and design

Keep the reviewer login ID and existing password/access token valid for 40 days.
Remove the separate one-hour authenticated-principal cap so a live Streamlit
session can remain authenticated until the account expires. Browser reconnects
can still require sign-in; this change does not add persistent browser cookies.

Default newly provisioned accounts to 960 hours and allow that duration through
the existing `--hours` interface. Preserve explicit shorter account lifetimes,
credential hashing, department restrictions, sign-out and immediate revocation.
Extend the existing local `reviewer` account to 40 days from implementation time
without changing its token hash or displaying credential material.

## Implementation and validation

1. Update provisioning defaults, duration validation and principal expiry.
2. Add clock-controlled invariants for validity after one hour, expiry at the
   account deadline and the 40-day provisioning default. Demonstrate the new
   invariants fail against the previous implementation.
3. Update the README with account and session lifetime behavior.
4. Run focused authentication tests, the full pytest suite with retained logs,
   and the affected checks runner. Review the diff for unrelated changes.
5. Extend the local reviewer account using an atomic write, preserving all other
   account fields. Report the expiry date without secrets.

## Open questions

- None. The user approved the plan and sidebar logout placement.

## Version control

After approval, create a branch and commit this plan as required by repository
instructions, preserving the existing unrelated working-tree changes.

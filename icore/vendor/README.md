# Included dependencies

These are fixed, reviewed source copies, not generated binary libraries:

- `blasfeo/vendor/blasfeo-source.tar.gz`: linear algebra source.
- `hpipm/hpipm-source.tar.gz`: comparator source.
- `qoco/qoco-source.tar.gz`: comparator source.

The outer archive names are deliberately readable. The original internal upstream
paths, licenses, and version identities are preserved. Build code validates source
content and applies the retained BLASFEO repairs automatically. Do not replace an
archive or remove checks without revalidating the numerical kernels.

No version identifier is a user setting. Normal builds go to ignored `.build/`
folders. See `../../docs/BUILDING.md` relative to the repository root, and the root
`THIRD_PARTY_NOTICES.md` for license locations.

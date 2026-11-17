# Literal text policy for new runs

Training labels and evaluation strings are literal. Preserve case, spaces, punctuation, currency symbols and character order. Checkpoint selection and reporting use complete-string exact count/accuracy only; the earliest epoch wins a tied count. No numeric equivalence, edit-distance tie-break or alphanumeric/case/space normalization is applied.

A same-stem UTF-8 `.txt` file contains one label. Exactly one optional terminal LF or CRLF is record framing, not a label character. Every other character is preserved, including leading/trailing spaces. Embedded newlines, a second terminal newline or a lone CR are rejected for this single-line format. JSON/CSV text fields remain literal; unsupported characters or length must raise rather than be changed or truncated. Set `normalization: none` for new training. Expand the alphabet explicitly when needed.

Existing inference checkpoints and legacy annotation-normalization metadata may still be loaded; output text is not normalized for scoring. This does not retroactively make old weights raw-text-trained. Old training state or cached selection summaries without the current literal policy marker are incompatible with resume: choose a new output directory. No automatic migration of old numeric/normalized/edit-distance best scores occurs.

Historical reports and archived sources keep their original definitions and values. They are not regenerated or relabeled as new literal-only experiments.

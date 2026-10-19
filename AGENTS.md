# Agent notes

> Authorship: agent-written (2026-10-18).

## Comments on the paper

Theo reviews the paper in `latex/review.html`, which Becemder serves at
<http://127.0.0.1:8788/files/bidiagonalization-experiments/latex/review.html>. The page shows `latex/document.pdf`.
He highlights text and comments, and each comment becomes a thread in the page's margin. His comments arrive as a
message that starts with `Comments on the paper`, and each comment in it has an id.

- Reply to every comment with `latex/review_reply.py <id> [--changed]`, with the reply on stdin. The reply shows in the
  comment's thread, which is where Theo reads it. Math in replies is written as `$...$` or `$$...$$`.
- When a comment asks a question or makes a remark and asks for no change, answer it in the reply and change nothing.
- When a comment asks for a change, or Theo asks you to integrate one of your answers, change the LaTeX source, rebuild
  with `latex/build.sh`, say in the reply what you changed, and pass `--changed`.
- The source is `latex/document.tex`, and the tables (`latex/table_*.tex`) and figures (`latex/figures/*.pdf`) that
  `plot_figures.py` draws; change that script and rerun it rather than editing its output.
- Find the passage from the `Section:` line, which names the section and the page, and from the words of the quote. The
  quote comes from the PDF's text, so formulas in it are garbled.
- End the chat reply with one sentence: how many comments you answered, and how many of them changed the paper.
- Do not edit `review.html` or `review-replies.txt` by hand. `review_reply.py` and `build.sh` both touch `review.html`,
  so that Becemder sends Theo's next comments back to the same chat.

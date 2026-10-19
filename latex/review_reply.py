"""Reply to a comment on the paper review page (latex/review.html). The page shows the reply in the comment's thread.

    python3 latex/review_reply.py <id> [--changed] <<'EOF'
    The reply. Math as $...$ or $$...$$.
    EOF

<id> is the comment's id from the message ("id c..."). Pass --changed when the reply changed the paper.
"""

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPLIES = HERE / "review-replies.txt"
PAGE = HERE / "review.html"
HEADER = "# Agent replies to comments on review.html, one JSON object per line, written by review_reply.py.\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("id", help="the comment's id, such as c1a2b3c4d")
    parser.add_argument("--changed", action="store_true", help="the reply changed the paper")
    args = parser.parse_args()
    if not re.fullmatch(r"c[0-9a-z]{4,}", args.id):
        sys.exit(f"{args.id!r} is not a comment id; ids look like c1a2b3c4d")
    text = sys.stdin.read().strip()
    if not text:
        sys.exit("write the reply on stdin")
    entry = {
        "id": args.id,
        "at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "changed": args.changed,
        "text": text,
    }
    with REPLIES.open("a", encoding="utf-8") as f:
        if f.tell() == 0:
            f.write(HEADER)
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    # Becemder sends the page's next comments to the chat that last changed review.html.
    PAGE.touch()
    print(f"Replied to {args.id}.")


if __name__ == "__main__":
    main()

#!/bin/sh
# Builds document.pdf. Touching review.html makes the chat that ran this the one that last updated the review page,
# so Becemder sends the next comments from that page back to the same chat.
set -e
cd "$(dirname "$0")"
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error document.tex
touch review.html

#!/bin/bash
set -e

GITHUB_BLOB="https://github.com/microsoft/qdk-chemistry/blob/main"
GITHUB_TREE="https://github.com/microsoft/qdk-chemistry/tree/main"

sed -E \
    -e "s#\]\(\./([^)]+)\)#](${GITHUB_BLOB}/\1)#g" \
    -e "s#\]\(([A-Z][A-Z_]*\.(md|txt))\)#](${GITHUB_BLOB}/\1)#g" \
    -e "s#\`examples/\`#[\`examples/\`](${GITHUB_TREE}/examples)#g" \
    -e "s#\`cpp/include/\`#[\`cpp/include/\`](${GITHUB_TREE}/cpp/include)#g" \
    -e '/^## Project Structure$/,/^```$/d' \
    README.md > python/README.md

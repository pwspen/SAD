#!/usr/bin/env bash
# Serves the browser pages for local use. It serves the folder *above* this
# repo, so train.html's dataset picker can reach ../comp/data/{kodak,mnist,cifar10} next to it.
#   ./serve.sh            # port 8766
#   PORT=9000 ./serve.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_NAME="$(basename "$REPO_DIR")"
PORT="${PORT:-8766}"
BASE="http://127.0.0.1:$PORT/$REPO_NAME/web"

print_links() {
  echo "  trainer: $BASE/train.html"
  echo "  viewer:  $BASE/index.html"
  echo "  bench:   $BASE/bench.html?image=../test.png"
}

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $PORT is already in use; if that's this server, open:"
  print_links
  exit 0
fi

for f in kodak/validation.parquet mnist/test.parquet cifar10/test.parquet; do
  if [ ! -f "$REPO_DIR/../comp/data/$f" ]; then
    echo "Note: no $(dirname "$REPO_DIR")/comp/data/$f; that dataset's picker will 404 (sources in web/README.md)."
  fi
done

echo "Serving $(dirname "$REPO_DIR") on port $PORT. Open:"
print_links
cd "$REPO_DIR/.."
exec python3 -m http.server "$PORT" --bind 127.0.0.1

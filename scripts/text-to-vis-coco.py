#!/usr/bin/env python3
"""
Convert a TXT list of FlyWire root IDs into:
  (A) a cocoglancer/Neuroglancer scene URL for quick visualization
  (B) optional NBLAST against hemibrain with static 2D plots of top matches

Dependencies:
  pip install fafbseg navis navis-flybrains
  # optional: Saalfeld transforms for navis/flybrains (see flybrains docs)

Optional data (~1.3GB):
  hemibrain_dotprops_jrc2018f.pkl  # from the tutorial
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# Core libs
from fafbseg import flywire
import navis
import flybrains

# --- temporary compat shim (older fafbseg where decode_url lacks 'ret') ---
from fafbseg.flywire import neuroglancer as _ng
import inspect as _inspect
if "ret" not in str(_inspect.signature(_ng.decode_url)):
    _old_decode = _ng.decode_url
    def _decode_url_compat(url, ret=None, format="json"):
        # ignore 'ret' and delegate to older API
        return _old_decode(url, format=format)
    _ng.decode_url = _decode_url_compat
# ---------------------------------------------------------------------------

from urllib.parse import urlparse, unquote


# ------------------------- Utilities ------------------------- #

def setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_ids(txt_path: Path) -> List[int]:
    """Load FlyWire root IDs from text; accepts commas/spaces/newlines."""
    if not txt_path.exists():
        raise FileNotFoundError(f"Missing IDs file: {txt_path}")
    ids: List[int] = []
    for ln in txt_path.read_text().splitlines():
        ln = ln.strip().replace(",", " ")
        if not ln:
            continue
        for tok in ln.split():
            if tok.isdigit():
                ids.append(int(tok))
            else:
                logging.warning("Skipping non-numeric token: %s", tok)
    ids = sorted(set(ids))
    if not ids:
        raise SystemExit("No valid numeric FlyWire root IDs found.")
    logging.info("Loaded %d unique IDs", len(ids))
    return ids


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")
    logging.info("Wrote URL to %s", path.as_posix())


def _multi_unquote(s: str, max_times: int = 3) -> str:
    prev = s
    for _ in range(max_times):
        cur = unquote(prev)
        if cur == prev:
            return cur
        prev = cur
    return prev


def validate_neuroglancer_url(u: str) -> None:
    """
    Validate a Neuroglancer/FlyWire URL:
      - Accepts inline JSON state (…#!{…})
      - Accepts pointer forms (…#!json_url=…, …#!state_url=…, …#!url=…)
      - Raises only for clearly malformed or missing fragments
    """
    if not isinstance(u, str) or not u.strip():
        raise SystemExit("Neuroglancer URL is empty.")

    frag = urlparse(u).fragment
    if not frag:
        raise SystemExit("Neuroglancer URL has no fragment after '#!'.")

    decoded = _multi_unquote(frag.strip())

    if decoded.startswith(("json_url=", "state_url=", "url=")):
        return

    if decoded.startswith("state="):
        decoded = decoded[len("state="):].lstrip()

    if decoded.lstrip().startswith("{"):
        json.loads(decoded)
        return

    maybe = _multi_unquote(decoded)
    if maybe.lstrip().startswith("{"):
        json.loads(maybe)
        return

    if len(decoded.strip()) < 10:
        raise SystemExit(f"Fragment too short to be valid: {decoded!r}")
    return


def encode_cocoglancer_url(
    ids: List[int],
    scene_url: Optional[str],
    dataset: str,
    shorten: bool,
) -> str:
    """Inject IDs into an optional cocoglancer base scene and return a shareable URL."""
    NULL_SCENES = {"", "none", "null", "None", None}

    kwargs = dict(segments=ids, dataset=dataset, shorten=shorten, open=False)
    if scene_url not in NULL_SCENES:
        kwargs["scene"] = str(scene_url).strip()

    url = flywire.encode_url(**kwargs)
    logging.info("Encoded scene with %d segments (dataset=%s)", len(ids), dataset)
    return url


def ensure_dataset(dataset: str) -> None:
    """Set default FlyWire dataset (e.g., 'public' or 'production')."""
    flywire.set_default_dataset(dataset)
    logging.debug('flywire.set_default_dataset("%s") done', dataset)


def maybe_update_roots(ids: List[int], do_update: bool, dataset: str) -> List[int]:
    """Optionally update IDs to current roots (requires FlyWire token)."""
    if not do_update:
        return ids
    logging.info("Updating IDs to latest roots (dataset=%s)...", dataset)
    df = flywire.update_ids(ids, dataset=dataset)
    new_ids = df["new_id"].astype(int).tolist()
    logging.info("Updated %d → %d IDs", len(ids), len(new_ids))
    return new_ids


# ------------------------- NBLAST Pipeline ------------------------- #

def fw_ids_to_dotprops(ids: List[int]):
    logging.info("Fetching L2 dotprops for %d FlyWire IDs…", len(ids))
    dps = flywire.get_l2_dotprops(ids)
    logging.info("Got dotprops for %d neurons", len(dps))
    return dps


def transform_to_jrc2018f(dotprops):
    logging.info("Transforming dotprops to JRC2018F…")
    dps_2018f = navis.xform_brain(dotprops, source="FLYWIRE", target="JRC2018F")
    return dps_2018f


def mirror_one_to_right(dps_2018f, root_id_to_mirror: Optional[int]):
    if root_id_to_mirror is None:
        return dps_2018f
    logging.info("Mirroring root %d to opposite hemisphere…", root_id_to_mirror)
    mirrored = navis.mirror_brain(dps_2018f.idx[root_id_to_mirror], template="JRC2018F")
    keep_ids = [rid for rid in dps_2018f.id if rid != root_id_to_mirror]
    combined = dps_2018f.idx[keep_ids] + mirrored
    return combined


def prune_to_hemibrain_bbox(dps_2018f):
    logging.info("Pruning to hemibrain bounding box…")
    bbox = flybrains.JRCFIB2018Fraw.bbox
    bbox_2018f = navis.xform_brain(bbox, source="JRCFIB2018Fraw", target="JRC2018F")
    dps_final = navis.in_volume(dps_2018f, bbox_2018f)
    return dps_final, bbox_2018f


def load_hemi_dotprops(pkl_path: Path):
    import pickle
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Hemibrain dotprops pickle not found: {pkl_path}\n"
            "Download the ~1.3GB file referenced in the tutorial and set --hemi-dotprops"
        )
    logging.info("Loading hemibrain dotprops from %s …", pkl_path)
    with open(pkl_path, "rb") as f:
        hemi = pickle.load(f)
    logging.info("Hemibrain dotprops loaded: %d neurons", len(hemi))
    return hemi


def run_nblast_smart(query_dps, hemi_dps, t: int = 10, criterion: str = "N"):
    logging.info("Running navis.nblast_smart (t=%d, criterion=%s)…", t, criterion)
    scores = navis.nblast_smart(
        query_dps,
        hemi_dps,
        scores="mean",
        criterion=criterion,
        t=t,
        normalized=True,
    )
    logging.info("NBLAST complete. Scores shape: %s", (scores.shape,))
    return scores


def top_matches(scores: pd.DataFrame, N: int = 3) -> pd.DataFrame:
    ix_srt = np.argsort(scores.values, axis=1)[:, ::-1]
    out = pd.DataFrame(index=scores.index)
    for i in range(N):
        out[f"top_{i+1}"] = scores.columns[ix_srt[:, i]]
        out[f"top_{i+1}_score"] = scores.values[np.arange(scores.shape[0]), ix_srt[:, i]]
    return out


def save_top_match_plots(
    query_dps,
    hemi_dps,
    top_df: pd.DataFrame,
    outdir: Path,
    zoom: float = 1.0,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for rid in top_df.index:
        hb_id = int(top_df.loc[rid, "top_1"])
        fig, ax = navis.plot2d(
            [query_dps.idx[rid], hemi_dps.idx[hb_id]],
            figsize=(10, 10),
            method="3d_complex",
            lw=1.5,
            c=["k", "r"],
        )
        ax.azim = ax.elev = -90
        ax.set_box_aspect(None, zoom=zoom)
        p = outdir / f"query_{rid}_vs_hemi_{hb_id}.png"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        logging.info("Saved: %s", p.as_posix())


# ------------------------- CLI ------------------------- #

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="TXT FlyWire IDs → cocoglancer URL (+ optional hemibrain NBLAST)"
    )
    ap.add_argument("ids_txt", type=Path, help="Text file with FlyWire ROOT IDs")

    # Scene / visualization
    ap.add_argument(
        "--scene",
        default="none",
        help="Base cocoglancer scene URL to inject segments into. Use 'none' to skip.",
    )
    ap.add_argument(
        "--dataset",
        default="public",
        choices=["public", "production", "sandbox", "flat_630"],
        help="FlyWire dataset (use 'public' for 783)",
    )
    ap.add_argument(
        "--shorten",
        action="store_true",
        help="Return a short redirect URL (requires FlyWire token)",
    )
    ap.add_argument(
        "--update-roots",
        action="store_true",
        help="Map IDs to latest roots before use (requires FlyWire token)",
    )

    # NBLAST options
    ap.add_argument(
        "--do-nblast",
        action="store_true",
        help="Run hemibrain NBLAST pipeline & save static plots",
    )
    ap.add_argument(
        "--hemi-dotprops",
        type=Path,
        default=None,
        help="Path to hemibrain_dotprops_jrc2018f.pkl (~1.3GB)",
    )
    ap.add_argument(
        "--mirror-id",
        type=int,
        default=None,
        help="FlyWire root ID to mirror to the opposite hemisphere (optional)",
    )
    ap.add_argument(
        "--prune-hemibrain",
        action="store_true",
        help="Prune query dotprops to hemibrain bounding box before NBLAST",
    )
    ap.add_argument(
        "--nblast-top",
        type=int,
        default=3,
        help="How many top matches to report per query neuron",
    )
    ap.add_argument(
        "--plots-out",
        type=Path,
        default=Path("nblast_plots"),
        help="Directory to save static 2D overlay plots",
    )
    ap.add_argument(
        "--out-url",
        type=Path,
        default=Path("data/cache/scene_url.txt"),
        help="Write the generated Neuroglancer URL to this file",
    )
    ap.add_argument(
        "--no-print-url",
        action="store_true",
        help="Do not print the long URL to stdout (use with --out-url)",
    )

    # Misc
    ap.add_argument("-v", "--verbose", action="count", default=0, help="Verbosity")
    return ap.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logging.info("Starting…")

    # Load & prep IDs
    ids = load_ids(args.ids_txt)
    ensure_dataset(args.dataset)
    ids = maybe_update_roots(ids, args.update_roots, args.dataset)

    # A) Neuroglancer/cocoglancer URL
    url = encode_cocoglancer_url(
        ids=ids, scene_url=args.scene, dataset=args.dataset, shorten=args.shorten
    )

    # Minimal safe patch: validate only when NOT shortening
    if not args.shorten:
        validate_neuroglancer_url(url)

    if args.out_url:
        write_text(args.out_url, url)

    if not args.no_print_url:
        print("\n=== Neuroglancer / cocoglancer URL ===")
        print(url)
        print("Copy into your browser; use the camera icon for a PNG.\n")
    else:
        print(f"\nURL saved to: {args.out_url}\n")

    # B) Optional hemibrain NBLAST pipeline
    if args.do_nblast:
        if args.hemi_dotprops is None:
            raise SystemExit("--do-nblast requires --hemi-dotprops <path-to-pkl>")
        fw_dps = fw_ids_to_dotprops(ids)
        fw_dps_2018f = transform_to_jrc2018f(fw_dps)
        fw_dps_2018f = mirror_one_to_right(fw_dps_2018f, args.mirror_id)
        if args.prune_hemibrain:
            fw_dps_2018f, _ = prune_to_hemibrain_bbox(fw_dps_2018f)
            logging.debug("Pruned to hemibrain; neuron count now: %d", len(fw_dps_2018f))
        hemi_dps_2018f = load_hemi_dotprops(args.hemi_dotprops)
        scores = run_nblast_smart(fw_dps_2018f, hemi_dps_2018f, t=10, criterion="N")
        top_df = top_matches(scores, N=args.nblast_top)
        print("=== Top matches per query neuron ===")
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(top_df)
        save_top_match_plots(
            query_dps=fw_dps_2018f, hemi_dps=hemi_dps_2018f, top_df=top_df, outdir=args.plots_out
        )
        logging.info("NBLAST pipeline complete.")

    logging.info("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
        sys.exit(130)

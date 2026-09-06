#!/usr/bin/env python3
"""Score or diff body-plan classifications (``category_groups.json``).

    # accuracy against a hand truth set ({asset: [acceptable categories]})
    python data_process/tools/eval_category_groups.py \
        --groups dataset/export/truebones/category_groups.json \
        --truth data_process/vlm_caption/eval/truebones_category_truth.json

    # differences between two runs (e.g. two models / prompts)
    python data_process/tools/eval_category_groups.py --groups A.json --other B.json

With ``--review`` (the ``<stem>_review.json`` sibling written by
``classify_category.py``) every miss is printed with its votes and reasons.
"""

import argparse
import json
from collections import Counter

ALIASES = {'quadruped': 'quadrupedal', 'biped': 'bipedal'}


def load_map(path):
    groups = json.load(open(path))
    return {a: ALIASES.get(c, c) for c, members in groups.items() for a in members}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--groups', required=True, help='category_groups.json to score')
    ap.add_argument('--truth', help='{asset: [acceptable categories]} JSON (keys starting with _ ignored)')
    ap.add_argument('--other', help='another category_groups.json to diff against')
    ap.add_argument('--review', help='<stem>_review.json of --groups (votes / reasons per asset)')
    args = ap.parse_args()

    pred = load_map(args.groups)
    review = json.load(open(args.review)) if args.review else {}
    print(f'{args.groups}: {len(pred)} assets, {dict(Counter(pred.values()))}')

    def detail(asset):
        rec = review.get(asset, {})
        if not rec:
            return ''
        reasons = [a.get('reason') for a in rec.get('answers', []) if a.get('reason')]
        return f"  votes={rec.get('votes')} conf={rec.get('confidence')} reasons={reasons[:2]}"

    if args.truth:
        truth = {k: v for k, v in json.load(open(args.truth)).items() if not k.startswith('_')}
        ok = [a for a, t in truth.items() if pred.get(a) in t]
        unc = [a for a in truth if pred.get(a) == 'uncertain']
        miss = [a for a, t in truth.items() if pred.get(a) not in t and pred.get(a) != 'uncertain']
        print(f'truth: {len(truth)} assets | correct {len(ok)} | uncertain {len(unc)} | wrong {len(miss)} '
              f'| missing {sum(1 for a in truth if a not in pred)}')
        for a in miss:
            print(f'  WRONG     {a:36s} got={pred.get(a)!s:18s} truth={truth[a]}{detail(a)}')
        for a in unc:
            print(f'  UNCERTAIN {a:36s} truth={truth[a]}{detail(a)}')

    if args.other:
        other = load_map(args.other)
        common = sorted(set(pred) & set(other))
        diff = [a for a in common if pred[a] != other[a]]
        print(f'vs {args.other}: {len(common)} common assets, {len(common) - len(diff)} agree, {len(diff)} differ')
        pairs = Counter((pred[a], other[a]) for a in diff)
        for (p, o), n in pairs.most_common():
            print(f'  {p:18s} vs {o:18s} x{n}')
        for a in diff[:40]:
            print(f'  {a:36s} {pred[a]:18s} vs {other[a]}{detail(a)}')


if __name__ == '__main__':
    main()

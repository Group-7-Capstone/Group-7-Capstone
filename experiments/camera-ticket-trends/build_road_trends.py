"""Build road-feature shared trends from the unchanged local ticket snapshot."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from data.reader import read_lion_road_attributes
from preprocessing.camera_shared_trends import DEFAULTS, load_inputs, aligned_panels, normalize_baseline
from preprocessing.camera_road_features import ROAD_CONFIG, GROUPS, UNKNOWN, match_camera_roads, road_cohort_ids
from validation.camera_road_features import validate_road_analysis
from validation.camera_shared_trends import describe_matrix
from build_shared_analysis import rows_json

OUT = HERE / 'outputs/road_features'
LABELS = {'lanes': 'Travel lanes', 'width': 'Paved road width', 'direction': 'Traffic direction'}


def run(lion_path=None, reuse_matches=False):
    OUT.mkdir(parents=True, exist_ok=True)
    summary, weekly, audit = load_inputs(HERE / 'outputs')
    meta, primary, _, _ = aligned_panels(summary, weekly, audit)
    if reuse_matches:
        features = pd.read_csv(OUT / 'camera_road_features.csv')
        segments = pd.read_csv(OUT / 'matched_segments.csv', dtype={'rw_type': str})
        lion_receipt = json.loads((OUT / 'lion_receipt.json').read_text(encoding="utf-8"))
    else:
        if lion_path is None:
            raise ValueError('Pass --lion-gdb pointing to the supplied LION geodatabase')
        print('Reading the pinned LION layers.', flush=True)
        roads, aliases, lion_receipt = read_lion_road_attributes(lion_path)
        print(f'Matching {len(summary):,} camera groups to {len(roads):,} LION segments.', flush=True)
        features, segments = match_camera_roads(summary, roads, aliases)
        features.to_csv(OUT / 'camera_road_features.csv', index=False)
        segments.to_csv(OUT / 'matched_segments.csv', index=False)
        (OUT / 'lion_receipt.json').write_text(json.dumps(lion_receipt, indent=2), encoding='utf-8')
    print('Rebuilding road-category shared trends.', flush=True)
    input_hashes = {name: hashlib.sha256((HERE / 'outputs' / name).read_bytes()).hexdigest()
                    for name in ['camera_summary.csv', 'camera_weekly_trends.parquet', 'reconciliation.json']}
    original_snapshot = json.loads((HERE / 'browser/src/data.json').read_text(encoding='utf-8'))
    old_queries = {k: hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()
                   for k, v in original_snapshot['queries'].items() if not k.startswith('road_')}
    curves, counts, members = [], [], []
    with threadpool_limits(limits=2):
        for kind in ['Speed', 'Red Light']:
            for horizon in [26, 52, 104]:
                for policy in ['intersection_only', 'include_street_only']:
                    for baseline_min in [0, 5]:
                        for dimension, groups in GROUPS.items():
                            for number, group in enumerate(groups):
                                ids = road_cohort_ids(features, primary, kind, horizon, dimension, group, policy, baseline_min)
                                common = {'camera_type': kind, 'horizon': horizon, 'policy': policy,
                                          'baseline_min': baseline_min, 'dimension': dimension, 'group': group, 'group_order': number}
                                matrix = primary.loc[ids, 1:horizon]
                                indexed = normalize_baseline(matrix, list(range(1, 14)))
                                index_n = int(indexed.notna().all(axis=1).sum())
                                counts.append({**common, 'cohort_n': len(ids), 'index_n': index_n,
                                               'small_group': len(ids) < ROAD_CONFIG['small_group_n'],
                                               'first_13w_raw_mean': matrix.iloc[:, :13].mean().mean(),
                                               'last_13w_raw_mean': matrix.iloc[:, -13:].mean().mean(),
                                               'last_13w_index_mean': indexed.iloc[:, -13:].mean().mean(),
                                               'index_change_pct': indexed.iloc[:, -13:].mean().mean() - 100,
                                               'low_baseline_n': int(matrix.iloc[:, :13].mean(axis=1).between(0, 5, inclusive='neither').sum())})
                                if matrix.empty:
                                    continue
                                result = describe_matrix(matrix, indexed, kind, 'road_features', DEFAULTS).rename(columns={'time': 'week'})
                                for k, value in common.items():
                                    result[k] = value
                                curves.append(result.drop(columns=['scenario', 'scenario_label']))
                                members.extend({**common, 'camera_id': camera_id} for camera_id in ids)
    trends, coverage, memberships = pd.concat(curves, ignore_index=True), pd.DataFrame(counts), pd.DataFrame(members)
    features['eligible_26w'] = features.camera_id.map(primary.loc[:, 1:26].notna().all(axis=1))
    features['eligible_52w'] = features.camera_id.map(primary.loc[:, 1:52].notna().all(axis=1))
    features['eligible_104w'] = features.camera_id.map(primary.loc[:, 1:104].notna().all(axis=1))
    status = features.groupby(['camera_type', 'county_labels', 'match_status'], dropna=False).size().rename('groups').reset_index()
    checks = validate_road_analysis(summary, primary, features, trends, coverage, memberships, segments)
    if not checks.passed.all():
        raise AssertionError(checks[~checks.passed].to_string())
    tables = {'trends': trends, 'coverage': coverage, 'camera_features': features, 'match_status': status,
              'memberships': memberships, 'validation_checks': checks}
    for name, frame in tables.items():
        frame.to_csv(OUT / f'{name}.csv', index=False)
    generated = datetime.now(timezone.utc).isoformat()
    definitions = {
        'Population': 'Unchanged Eric address/type groups, not verified physical devices; Speed and Red Light remain separate.',
        'Road association': 'The first named street is a context proxy, not a verified enforced approach. Exact normalized name, ticket borough, original coordinate within 60 m, and shared node plus level for intersection addresses. Generic and divided roadbeds are unresolved.',
        'Segment consensus': 'For each feature, every plausible adjoining first-street segment must agree on its category. No nearest-segment tie break or average width is used. A width interval crossing a boundary is unknown.',
        'Travel lanes': 'Number_Travel_Lanes, excluding parking lanes; both directions combined for a two-way road. Categories: 1, 2, 3, 4+.',
        'Paved road width': 'StreetWidth_Min/Max in feet multiplied by 0.3048. Both bounds of every candidate segment must fit <10, 10–<15, 15–<20, or 20+ metres. This is not lane width or intersection crossing distance.',
        'Traffic direction': 'T = Two-way; A/W = One-way. Only ordinary unsplit B roadbeds are assigned to prevent labeling a divided two-way street as one-way.',
        'Cohort': 'Same groups at every relative week; first full Monday–Sunday week on or after unchanged inferred activation. All 26/52/104 weeks must be complete through the last observed ticket.',
        'Normalized index': '100 × group weekly tickets / its own first 13-week mean; zero baselines excluded only from the index. Each group receives equal weight.',
        'Uncertainty': 'Pointwise 95% percentile bootstrap, 500 resamples of whole group histories, seed 42. Does not model shared intersections, spatial or calendar dependence, mapping uncertainty, or historical road changes.',
        'Sensitivities': 'Include street-only matches, require baseline >=5 tickets/week, or change the fixed follow-up window. Unknown curves preserve unresolved and missing-feature groups.',
        'Time interpretation': 'Frozen October 14 ticket extract through August 30, 2026; supplied LION snapshot has unverified release. Road characteristics are not reconstructed at activation. Descriptive associations only, not rates or causal effects.',
    }
    manifest = {'generated_at': generated, 'configuration': ROAD_CONFIG, 'ticket_inputs_sha256': input_hashes,
                'road_match_sha256': hashlib.sha256((OUT/'camera_road_features.csv').read_bytes()).hexdigest(),
                'lion_receipt': 'lion_receipt.json', 'definitions': definitions,
                'groups': len(features), 'strict_matches': int(features.strict_match.sum()),
                'street_only_matches': int(features.match_status.eq('Street-only match').sum()),
                'checks_passed': int(checks.passed.sum()), 'existing_queries_sha256': old_queries}
    (OUT / 'analysis_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    meanings = {'camera_id': 'Unchanged stable address/type identifier', 'strict_match': 'Both street names, borough, distance and connected node matched',
                'any_match': 'Strict match or weaker first-street-and-coordinate match', 'match_distance_m': 'Minimum point-to-candidate-segment distance; not pole-position accuracy',
                'candidate_segments': 'Number of plausible adjoining first-street segments retained', 'group': 'Road category; unknown never means zero',
                'policy': 'intersection_only or include_street_only', 'baseline_min': 'Minimum own first-13-week mean; 0 preserves zero baselines for raw means',
                'horizon': 'Fixed complete relative-week window', 'n': 'Raw contributing groups', 'index_n': 'Positive-baseline contributing groups',
                'index_change_pct': 'Last-13-week mean index minus 100; equals mean of individual proportional changes',
                'raw_mean': 'Equal-group arithmetic mean unique tickets per week', 'index_mean': 'Equal-group mean of own-baseline indices',
                'raw_mean_4w': 'Trailing four complete mean-count weeks; null for first three weeks', 'index_mean_4w': 'Trailing four complete mean-index weeks; null for first three weeks'}
    dictionary = [{'table': table, 'field': column, 'definition': meanings.get(column, definitions.get(LABELS.get(column.replace('_group','')), column.replace('_', ' ').capitalize() + '; see analysis_manifest.json and matched_segments.csv'))} for table, frame in tables.items() for column in frame.columns]
    pd.DataFrame(dictionary).to_csv(OUT/'field_dictionary.csv', index=False)
    source = {'name': 'Road-feature shared trends', 'type': 'file', 'provider': 'Supplied NYC DCP LION + frozen camera ticket aggregates',
              'executedAt': generated, 'files': ['camera_summary.csv', 'camera_weekly_trends.parquet', 'lion.gdb/lion', 'lion.gdb/altnames', 'road_features/analysis_manifest.json'],
              'metricDefinitions': [{'label': k, 'definition': v} for k,v in definitions.items()],
              'evidenceFlow': [{'title': 'Frozen input checksums', 'detail': json.dumps(input_hashes)}, {'title': 'LION snapshot', 'detail': 'Local source files are individually hashed in lion_receipt.json. Release vintage unverified.'}, {'title': 'Reproduction', 'detail': 'uv run python experiments/camera-ticket-trends/build_road_trends.py --lion-gdb <path-to-lion.gdb>'}, {'title': 'Validation', 'detail': f'{int(checks.passed.sum())} reconciliation checks passed. Automated matching is not field verification.'}]}
    for name in ['trends','coverage','camera_features','match_status']:
        original_snapshot['queries']['road_' + name] = {'rows': rows_json(tables[name]), 'source': source}
    original_snapshot['buildStatus'] = 'updating'
    original_snapshot['generatedAt'] = generated
    (HERE/'browser/src/data.json').write_text(json.dumps(original_snapshot,ensure_ascii=False,separators=(',', ':'),allow_nan=False),encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ['groups','strict_matches','street_only_matches','checks_passed']},indent=2),flush=True)
    print(coverage.query("horizon == 52 and policy == 'intersection_only' and baseline_min == 0")[["camera_type","dimension","group","cohort_n","index_n","last_13w_index_mean"]].to_string(index=False),flush=True)
    return tables, manifest


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--lion-gdb',type=Path)
    parser.add_argument('--reuse-matches',action='store_true',help='Reuse this analysis directory\'s pinned match export')
    args=parser.parse_args()
    run(args.lion_gdb,args.reuse_matches)

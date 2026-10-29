"""Compare a migrated experiment to its untouched local reference, without network."""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import pandas as pd
import nbformat


def compare_table(actual_path, expected_path):
    reader = pd.read_parquet if actual_path.suffix == '.parquet' else pd.read_csv
    actual, expected = reader(actual_path), reader(expected_path)
    if list(actual.columns) != list(expected.columns) or len(actual) != len(expected):
        raise AssertionError(f'Table shape differs: {actual_path.name}')
    maximum = 0.0
    for column in expected:
        a, b = actual[column], expected[column]
        if pd.api.types.is_float_dtype(b):
            if not np.allclose(a, b, rtol=1e-10, atol=1e-8, equal_nan=True):
                raise AssertionError(f'Floating values differ: {actual_path.name}/{column}')
            delta = np.abs(a.to_numpy() - b.to_numpy())
            if np.isfinite(delta).any():
                maximum = max(maximum, float(np.nanmax(delta)))
        else:
            pd.testing.assert_series_equal(a, b, check_dtype=False, check_exact=True)
    return {'file': actual_path.name, 'rows': len(actual), 'columns': len(actual.columns),
            'max_float_difference': maximum, 'passed': True}


def compare_experiment(actual, reference):
    """Require unchanged analytical tables; timestamps and QA receipts are separate."""
    actual, reference = Path(actual), Path(reference)
    checks = []
    skip = {'validation_checks.csv', 'field_dictionary.csv', 'source_coverage.csv',
            'group_coverage.csv', 'independent_checks.csv'}
    for path in sorted((reference/'outputs').rglob('*.csv')):
        if path.name in skip or 'qa' in path.relative_to(reference).parts:
            continue
        rel = path.relative_to(reference)
        checks.append({**compare_table(actual/rel, path), 'file': rel.as_posix()})
    rel = Path('outputs/camera_weekly_trends.parquet')
    checks.append({**compare_table(actual/rel, reference/rel), 'file': rel.as_posix()})
    before = json.loads((reference/'outputs/reconciliation.json').read_text(encoding='utf-8'))
    after = json.loads((actual/'outputs/reconciliation.json').read_text(encoding='utf-8'))
    if before != after:
        raise AssertionError('Raw-record reconciliation or historical provenance changed')
    notebooks = []
    for path in sorted(actual.glob('*.ipynb')):
        nb = nbformat.read(path, as_version=4)
        old = nbformat.read(reference/path.name, as_version=4)
        nbformat.validate(nb)
        cells = [cell for cell in nb.cells if cell.cell_type == 'code']
        assert [cell.execution_count for cell in cells] == list(range(1, len(cells)+1))
        assert not [o for c in cells for o in c.outputs if o.output_type == 'error']
        figures = lambda doc: [o['data']['image/png'] for c in doc.cells if c.cell_type == 'code'
                               for o in c.outputs if 'image/png' in o.get('data', {})]
        a, b = figures(nb), figures(old)
        assert len(a) == len(b), path.name
        notebooks.append({'file': path.name, 'code_cells': len(cells), 'embedded_figures': len(a),
                          'identical_embedded_images': sum(x == y for x,y in zip(a,b)), 'error_outputs': 0})
    result = {'tables': checks, 'notebooks': notebooks, 'reconciliation_unchanged': True,
              'float_rtol': 1e-10, 'float_atol': 1e-8}
    dest = actual/'outputs/integration'
    dest.mkdir(exist_ok=True)
    (dest/'result_regression.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'tables_checked': len(checks), 'rows_checked': sum(c['rows'] for c in checks),
                      'max_float_difference': max(c['max_float_difference'] for c in checks),
                      'notebooks': notebooks}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--experiment', type=Path, default=Path(__file__).resolve().parents[1]/'experiments/camera-ticket-trends')
    args = parser.parse_args()
    compare_experiment(args.experiment, args.reference)

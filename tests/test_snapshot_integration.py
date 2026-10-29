"""Shared download/reader contract, including legacy frozen snapshot receipts."""
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import data.download as download
import data.base
from data.reader import iter_camera_ticket_batches, camera_data_receipt, read_camera_data

COLUMNS = ['source_row_id', 'summons_number', 'issue_date', 'street_name']
WHERE = 'violation_code IN (7,36)'


class FakeClient:
    def __init__(self, records, *, expected=None, fail_after=None, changed=False):
        self.records = records
        self.expected = len(records) if expected is None else expected
        self.fail_after = fail_after
        self.changed = changed
        self.pages = 0
        self.metadata_calls = 0

    def get_metadata(self, dataset):
        self.metadata_calls += 1
        return {'name': 'Parking Violations FY2021',
                'rowsUpdatedAt': 2 if self.changed and self.metadata_calls > 1 else 1}

    def get(self, dataset, **kwargs):
        if 'count(*)' in kwargs['select']:
            return [{'year': '2021', 'n': str(self.expected)}]
        self.pages += 1
        if self.fail_after is not None and self.pages > self.fail_after:
            raise ConnectionError('Interrupted')
        query = kwargs['where']
        cursor = query.rsplit(":id > '", 1)[1].split("'")[0] if ':id > ' in query else ''
        return [r.copy() for r in self.records if r['source_row_id'] > cursor][:kwargs['limit']]

    def close(self):
        pass


def records():
    return [{'source_row_id': str(i), 'summons_number': f'00{i}', 'issue_date': '2021-01-04'} for i in range(1, 4)]


def pull(tmp_path, **kwargs):
    return download.fetch_soda('abcd-1234', 'tickets', WHERE, columns=COLUMNS,
        snapshot_dir=tmp_path/'fy2021', partition_field='issue_date', fiscal_year=2021,
        page_size=2, **kwargs)


def test_snapshot_resume_and_offline_legacy_reader(tmp_path, monkeypatch):
    interrupted = FakeClient(records(), fail_after=1)
    monkeypatch.setattr(download, '_client', lambda: interrupted)
    with pytest.raises(ConnectionError):
        pull(tmp_path)
    first = tmp_path/'fy2021/2021/00000.parquet'
    checksum = download.checksum(first)
    resumed = FakeClient(records())
    monkeypatch.setattr(download, '_client', lambda: resumed)
    assert pull(tmp_path) == 3
    assert download.checksum(first) == checksum
    monkeypatch.setattr(download, '_client', lambda: pytest.fail('Offline cache accessed network'))
    assert pull(tmp_path, offline=True) == 3
    frames = list(iter_camera_ticket_batches(tmp_path, batch_size=1))
    assert len(frames) == 3 and all(fy == 2021 for fy, _ in frames)
    raw = pd.concat([df for _, df in frames], ignore_index=True)
    assert raw.summons_number.tolist() == ['001','002','003']
    assert raw.street_name.isna().all()


@pytest.mark.parametrize('problem', ['checksum','cursor','count','missing_page','extra_partition','incomplete','source_change'])
def test_damaged_or_incomplete_cache_is_not_silently_accepted(tmp_path, monkeypatch, problem):
    monkeypatch.setattr(download, '_client', lambda: FakeClient(records()))
    pull(tmp_path)
    folder = tmp_path/'fy2021'
    path = folder/'2021/00000.json'
    receipt = json.loads(path.read_text())
    manifest_path = folder/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if problem == 'checksum':
        receipt['sha256'] = 'wrong'
    elif problem == 'cursor':
        receipt['previous_cursor'] = 'wrong'
    elif problem == 'count':
        receipt['rows'] = 20
    elif problem == 'missing_page':
        (folder/'2021/00000.parquet').unlink()
    elif problem == 'extra_partition':
        (folder/'2022').mkdir()
        pq.write_table(pa.table({'source_row_id':['9']}), folder/'2022/00000.parquet')
    elif problem == 'incomplete':
        manifest['complete'] = False
    else:
        manifest['source_changed_during_pull'] = True
    download.save_json(path, receipt)
    download.save_json(manifest_path, manifest)
    with pytest.raises((ValueError, FileNotFoundError)):
        pull(tmp_path, offline=True)


def test_unexpected_empty_page_and_source_revision(tmp_path, monkeypatch):
    monkeypatch.setattr(download, '_client', lambda: FakeClient(records()[:1], expected=3))
    with pytest.raises(ValueError, match='Empty page'):
        pull(tmp_path)
    other = tmp_path/'changed'
    monkeypatch.setattr(download, '_client', lambda: FakeClient(records(), changed=True))
    with pytest.raises(ValueError, match='Source changed'):
        pull(other)
    assert not json.loads((other/'fy2021/manifest.json').read_text())['complete']


def test_zero_record_partition_is_valid_and_fiscal_identity_is_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(download, '_client', lambda: FakeClient([]))
    assert pull(tmp_path) == 0
    assert pull(tmp_path, offline=True) == 0
    with pytest.raises(ValueError, match='identity'):
        download.fetch_soda('abcd-1234', 'tickets', WHERE, columns=COLUMNS,
            snapshot_dir=tmp_path/'wrong', partition_field='issue_date', fiscal_year=2020)


def test_parameter_change_cannot_reuse_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(download, '_client', lambda: FakeClient(records()))
    pull(tmp_path)
    with pytest.raises(ValueError, match='parameters differ'):
        download.fetch_soda('different', 'tickets', WHERE, columns=COLUMNS,
            snapshot_dir=tmp_path/'fy2021', partition_field='issue_date', fiscal_year=2021, offline=True)


def test_shared_registry_matches_archived_source_without_git():
    registry = read_camera_data('camera_activation_dates')
    assert len(registry) == 4610
    assert camera_data_receipt('camera_activation_dates')['lf_sha256'] == '1244a77a0a35083f43ab3447c60c6af48dbe354e5987eadea39c7fd981ba1062'


def test_flat_projection_and_batch_column_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(data.base, 'DB_PATH', tmp_path/'raw.duckdb')
    monkeypatch.setattr(download, '_client', lambda: FakeClient(records()))
    assert download.fetch_soda('abcd-1234','projected',WHERE,columns=['summons_number']) == 3
    from data.reader import read_table
    assert list(read_table('projected').columns) == ['source_row_id','summons_number']
    assert read_table('projected', columns=['summons_number']).summons_number.tolist() == ['001','002','003']
    assert download.fetch_soda('abcd-1234','projected',WHERE,columns=['summons_number','issue_date']) == 3
    assert 'issue_date' in read_table('projected')
    pull(tmp_path)
    batches = list(iter_camera_ticket_batches(tmp_path, columns=['summons_number']))
    assert all(list(frame.columns) == ['summons_number'] for _, frame in batches)

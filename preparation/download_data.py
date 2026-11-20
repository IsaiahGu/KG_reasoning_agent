"""Download and verify plan inputs only. Never imports a model or runs an experiment."""
import concurrent.futures
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
import urllib.parse
import urllib.request
import zipfile

ROOT = Path('/workspace/hg_reasoning_agent')
DATA = ROOT / 'data'
REPORT = ROOT / 'reports' / 'data_download_manifest.json'
LOCK = threading.Lock()
manifest = {'purpose': 'download-only preparation; no inference, indexing, evaluation or training',
            'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'files': [], 'archives': [], 'notes': [], 'errors': []}

def save():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    tmp = REPORT.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(REPORT)

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)

def hf_tasks(repo, names, subdir):
    info = get_json('https://huggingface.co/api/datasets/' + repo)
    revision = info['sha']
    tree = get_json('https://huggingface.co/api/datasets/' + repo + '/tree/' + revision + '?recursive=true')
    tree = {x['path']: x for x in tree}
    tasks = []
    for name in names:
        row = tree[name]
        tasks.append({'url': 'https://huggingface.co/datasets/' + repo + '/resolve/' + revision + '/' + name,
                      'relative_path': subdir + '/' + name, 'source_repo': repo,
                      'source_revision': revision, 'expected_bytes': row['size'],
                      'expected_sha256': row.get('lfs', {}).get('oid')})
    return tasks

def resolve_drive(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        if 'text/html' not in r.headers.get('Content-Type', ''):
            return url
        text = r.read().decode()
    action = re.search(r'<form[^>]*action="([^"]+)"', text)
    fields = dict(re.findall(r'<input[^>]*type="hidden"[^>]*name="([^"]+)"[^>]*value="([^"]*)"', text))
    if not action or not fields.get('id') or not fields.get('confirm'):
        raise RuntimeError('Google Drive returned no public download confirmation form')
    return html.unescape(action.group(1)) + '?' + urllib.parse.urlencode(fields)

def drive_range_download(task, target):
    """Preserve a contiguous prefix and fetch exact, independently retryable ranges."""
    expected = task['expected_bytes']
    partial = target.with_name(target.name + '.part')
    prefix = partial.stat().st_size if partial.is_file() else 0
    if prefix > expected:
        raise RuntimeError('Existing Google Drive prefix is larger than its source')
    piece_dir = target.with_name(target.name + '.ranges')
    piece_dir.mkdir(exist_ok=True)
    direct = resolve_drive(task['url'])
    chunk_size = 8 * 1024 * 1024
    ranges = [(start, min(start + chunk_size - 1, expected - 1))
              for start in range(prefix, expected, chunk_size)]
    print('RESUME_RANGES', task['relative_path'], 'prefix', prefix,
          'total', expected, 'chunks', len(ranges), flush=True)

    def fetch(pair):
        start, end = pair
        piece = piece_dir / f'{start}-{end}'
        length = end - start + 1
        if piece.is_file() and piece.stat().st_size == length:
            return piece
        for attempt in range(8):
            try:
                req = urllib.request.Request(direct, headers={'Range': f'bytes={start}-{end}'})
                with urllib.request.urlopen(req, timeout=120) as r:
                    if r.status != 206 or r.headers.get('Content-Range') != f'bytes {start}-{end}/{expected}':
                        raise RuntimeError('Google Drive did not honor the exact requested byte range')
                    with piece.open('wb') as f:
                        shutil.copyfileobj(r, f, length=1024 * 1024)
                if piece.stat().st_size != length:
                    raise RuntimeError('Google Drive range was truncated')
                print('RANGE_READY', task['relative_path'], start, end, flush=True)
                return piece
            except Exception as e:
                print('RANGE_RETRY', task['relative_path'], start, attempt + 1,
                      type(e).__name__, str(e)[:160], flush=True)
                if attempt == 7:
                    raise
                time.sleep(2)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pieces = list(pool.map(fetch, ranges))
    with partial.open('ab') as dst:
        for piece in pieces:
            with piece.open('rb') as src:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
    if partial.stat().st_size != expected:
        raise RuntimeError('Assembled Google Drive file size mismatch')
    partial.replace(target)
    for piece in pieces:
        piece.unlink()
    piece_dir.rmdir()

def download(task):
    target = DATA / task['relative_path']
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = task.get('expected_bytes')
    expected_hash = task.get('expected_sha256')
    for attempt in range(8):
        try:
            if task.get('google_drive') and (not target.is_file() or target.stat().st_size != expected):
                drive_range_download(task, target)
            if not target.is_file() or (expected and target.stat().st_size != expected):
                request_url = resolve_drive(task['url']) if task.get('google_drive') else task['url']
                partial = target.with_name(target.name + '.part')
                offset = partial.stat().st_size if partial.is_file() else 0
                headers = {'User-Agent': 'hg-reasoning-agent-data-preparation/1.0'}
                if offset:
                    headers['Range'] = f'bytes={offset}-'
                request = urllib.request.Request(request_url, headers=headers)
                with urllib.request.urlopen(request, timeout=120) as r:
                    if 'text/html' in r.headers.get('Content-Type', ''):
                        raise RuntimeError('Received HTML instead of dataset bytes')
                    content_length = int(r.headers.get('Content-Length', 0)) or None
                    if offset and r.status == 206:
                        content_range = r.headers.get('Content-Range', '')
                        if not content_range.startswith(f'bytes {offset}-'):
                            raise RuntimeError('Unexpected HTTP resume range')
                    elif offset:
                        # A server ignoring Range is allowed, but the old prefix
                        # must never be appended to an entire 200 response.
                        offset = 0
                    count = 0
                    with partial.open('ab' if offset else 'wb') as f:
                        while True:
                            block = r.read(8 * 1024 * 1024)
                            if not block:
                                break
                            f.write(block)
                            count += len(block)
                    if content_length and count != content_length:
                        raise RuntimeError(f'HTTP length mismatch: {count} != {content_length}')
                    if expected and offset + count != expected:
                        raise RuntimeError(f'Source size mismatch: {offset + count} != {expected}')
                    partial.replace(target)
            digest = sha256(target)
            if expected_hash and digest != expected_hash:
                raise RuntimeError('Source SHA256 mismatch')
            row = dict(task)
            row.pop('google_drive', None)
            row.update(bytes=target.stat().st_size, sha256=digest,
                       verified='source_sha256_and_size' if expected_hash else 'size_and_local_sha256')
            with LOCK:
                manifest['files'].append(row)
                save()
            print('READY', task['relative_path'], row['bytes'], flush=True)
            return row
        except Exception as e:
            print('RETRY', task['relative_path'], attempt + 1, type(e).__name__, str(e)[:300], flush=True)
            if attempt == 7:
                with LOCK:
                    manifest['errors'].append({'task': task, 'error': type(e).__name__ + ': ' + str(e)[:500]})
                    save()
                return None
            time.sleep(2)

def extract(relative_zip, relative_out):
    source, target = DATA / relative_zip, DATA / relative_out
    if not source.is_file():
        return
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        members = [x for x in archive.infolist() if not x.is_dir() and '__MACOSX' not in x.filename]
        total = sum(x.file_size for x in members)
        free = shutil.disk_usage(target).free
        if total > 12_000_000_000 or free < total + 3_000_000_000:
            raise RuntimeError(f'Insufficient safe extraction space for {relative_zip}: {total}, free {free}')
        files = []
        for member in members:
            out = (target / member.filename).resolve()
            if not out.is_relative_to(target.resolve()):
                raise RuntimeError('Unsafe ZIP member path')
            out.parent.mkdir(parents=True, exist_ok=True)
            # zipfile verifies each member's CRC while streaming to disk.
            with archive.open(member) as src, out.open('wb') as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
            files.append({'path': str(out.relative_to(DATA)), 'bytes': out.stat().st_size,
                          'sha256': sha256(out), 'source_crc32': f'{member.CRC:08x}'})
    manifest['archives'].append({'source': relative_zip, 'destination': relative_out,
                                 'uncompressed_bytes': total, 'file_count': len(files), 'files': files})
    save()
    print('EXTRACTED', relative_zip, len(files), total, flush=True)

def main():
    DATA.mkdir(parents=True, exist_ok=True)
    tasks = []
    tasks += hf_tasks('osunlp/HippoRAG_2',
        ['README.md', 'musique.json', 'musique_corpus.json', '2wikimultihopqa.json',
         '2wikimultihopqa_corpus.json', 'hotpotqa.json', 'hotpotqa_corpus.json',
         'nq_rear.json', 'nq_rear_corpus.json', 'popqa.json', 'popqa_corpus.json'], 'hipporag2')
    tasks += hf_tasks('hotpotqa/hotpot_qa',
        ['README.md', 'distractor/train-00000-of-00002.parquet', 'distractor/train-00001-of-00002.parquet'], 'raw/hotpotqa')
    tasks += [
        {'url': 'https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?dl=1',
         'relative_path': 'raw/2wikimultihop/data_ids_april7.zip', 'expected_bytes': 258968175,
         'source_reference': 'https://github.com/Alab-NII/2wikimultihop#readme'},
        {'url': 'https://drive.google.com/uc?export=download&id=1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h',
         'relative_path': 'raw/musique/musique_data_v1.0.zip', 'google_drive': True,
         'expected_bytes': 272049578,
         'source_reference': 'https://github.com/stonybrooknlp/musique/blob/main/download_data.sh'},
    ]
    manifest['notes'] += [
        'Only downloads and structural checks are performed; no model is loaded and no experiment starts.',
        'HotpotQA old CMU download hostname fails DNS; official maintainer Hugging Face dataset train parquet shards preserve questions, answers, contexts and supporting facts.',
    ]
    save()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, tasks))
    for source, dest in [
        ('raw/2wikimultihop/data_ids_april7.zip', 'raw/2wikimultihop/extracted'),
        ('raw/musique/musique_data_v1.0.zip', 'raw/musique/extracted'),
    ]:
        try:
            extract(source, dest)
        except Exception as e:
            manifest['errors'].append({'archive': source, 'error': type(e).__name__ + ': ' + str(e)})
            save()
            print('EXTRACT_ERROR', source, type(e).__name__, str(e), flush=True)
    manifest['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    manifest['downloaded_bytes'] = sum(x['bytes'] for x in manifest['files'])
    manifest['extracted_bytes'] = sum(x['uncompressed_bytes'] for x in manifest['archives'])
    save()
    print('FINISHED', manifest['downloaded_bytes'], manifest['extracted_bytes'], 'errors', len(manifest['errors']), flush=True)

if __name__ == '__main__':
    main()

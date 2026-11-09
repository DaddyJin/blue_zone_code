"""JSON Schema preflight; the converter checks additional physical/data rules."""
import argparse,json
from pathlib import Path

def check(config):
    from jsonschema import Draft202012Validator
    schema=json.loads(Path(__file__).with_name('manifest.schema.json').read_text())
    Draft202012Validator.check_schema(schema)
    errors=sorted(Draft202012Validator(schema).iter_errors(config),key=lambda e:str(list(e.path)))
    if errors:raise ValueError('\n'.join(f'{list(e.path)}: {e.message}' for e in errors[:20]))
    return {'passed':True,'scope':'manifest schema; verify source semantics and data separately'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('manifest');a=p.parse_args();print(json.dumps(check(json.loads(Path(a.manifest).read_text()))))

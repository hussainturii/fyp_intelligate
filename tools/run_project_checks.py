"""Run isolated project component checks and save an honest evidence matrix."""
import json
import sys
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from module8_dashboard.backend.test_workbench import component_checks

if __name__=='__main__':
    checks=component_checks(ROOT)
    report={'created_at':datetime.now(timezone.utc).isoformat(),'checks':checks,
            'scope':'Component smoke tests only. Missing research features and real-data accuracy requirements remain explicit.'}
    output=ROOT/'logs/testing/component_checks_latest.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    for check in checks:
        print(f"{check['module']:3} {check['status']:16} {check['name']}")
    print('Evidence:',output)
    sys.exit(1 if any(row['status']=='FAIL' for row in checks) else 0)

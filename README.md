# 식당 운영 최적화 실행 예제

합성 데이터 또는 CSV로 식당 라인, 운영 시간, 배식 인력을 최적화하고 현행안과 비교한다.

```bash
pip install numpy pandas scipy matplotlib pytest
pytest -q
python cafeteria_ops.py --restaurants 3 --employees 300 --time-limit 30 --out out_small
python cafeteria_ops.py --restaurants 30 --employees 5000 --time-limit 300 --out out_full
```

- `--mode lpfix`: 전체 MILP가 느릴 때
- `--mode flow`: 탐욕 개설 + 최소비용흐름. `ortools`가 있으면 `SimpleMinCostFlow`, 없으면 `scipy.optimize.linprog`
- `--special`: 특식일
- `--data-dir`: 실제 CSV 사용. 형식은 `--export-data`로 확인

## 현재 상태

- 작은 인스턴스의 pytest 9개는 통과했다.
- 식당 30곳, 근무자 5,000명 전체 규모 실행은 아직 끝까지 검증하지 않았다.
- `ortools` 경로는 미설치 환경이라 실행해 보지 않았다.
- 작은 인스턴스에서 최적안이 수요 +20%와 식당 폐쇄에 취약했다. 시뮬레이션 결과를 보고 `capacity_buffer`를 조정해야 한다.
- 비용과 데이터는 모두 가정값이므로, 결과를 실제 절감액으로 해석하면 안 된다.

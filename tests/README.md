# 🧪 Automatiniai Testai — Steam Žaidimų Rekomendacijos Sistema

## 📋 Testų Aprašymas

**Iš viso: 15 pagrindinių testų** + papildomi error handling / caching testai

Testai aprėpia:
- ✅ **Routes** (7 testai) — HTML šablonų loading
- ✅ **API Endpoints** (4 testai) — JSON API validacija
- ✅ **Input Validation** (2 testai) — Parametrų ribos
- ✅ **Search Modes** (1 testas) — Visos 7 paieškos modalybės
- ✅ **Data Integrity** (1 testas) — Rezultatų struktūra

---

## 🚀 Paleisimas

### Prieš testus:
```bash
# Instaliuojame test dependency'ius
pip install -r requirements.txt

# Arba vien testing tools
pip install pytest pytest-flask pytest-cov
```

### Paleisti VISUS testus:
```bash
pytest tests/ -v
```

### Su coverage report'u:
```bash
pytest tests/ -v --cov=app
```

### Tik konkretūs testai:
```bash
# Tik route testai
pytest tests/test_app.py::TestRoutes -v

# Tik API testai
pytest tests/test_app.py::TestAPIEndpoints -v

# Vienas testas
pytest tests/test_app.py::TestRoutes::test_01_index_route_loads -v
```

### Greitai (be slow testų):
```bash
pytest tests/ -v -m "not slow"
```

---

## 📊 15 Pagrindinių Testų

| # | Testas | Tikslas | Status |
|---|--------|---------|--------|
| 1️⃣ | `test_01_index_route_loads` | Index HTML loads | Routes |
| 2️⃣ | `test_02_search_route_valid_query` | Search su valid query | Routes |
| 3️⃣ | `test_03_search_route_invalid_short_query` | Rejects < 2 chars | Routes |
| 4️⃣ | `test_04_search_route_invalid_long_query` | Rejects > 200 chars | Routes |
| 5️⃣ | `test_05_game_detail_valid_id` | Game page loads | Routes |
| 6️⃣ | `test_06_game_detail_invalid_id` | Invalid ID → 400 | Routes |
| 7️⃣ | `test_07_game_detail_nonexistent_id` | Missing game → 404 | Routes |
| 8️⃣ | `test_08_api_search_endpoint` | /api/search JSON | API |
| 9️⃣ | `test_09_api_search_invalid_query` | Invalid query → 400 | API |
| 🔟 | `test_10_api_explain_endpoint` | /api/explain Gemini | API |
| 1️⃣1️⃣ | `test_11_api_clear_history` | POST clear history | API |
| 1️⃣2️⃣ | `test_12_search_weight_validation` | Svoriai 0-1 | Validation |
| 1️⃣3️⃣ | `test_13_search_top_k_limits` | top_k 1-50 | Validation |
| 1️⃣4️⃣ | `test_14_search_modes_all_working` | 7 paieškos režimai | Search |
| 1️⃣5️⃣ | `test_15_search_returns_structured_results` | Result JSON valid | Data |

---

## 📁 Failų Struktūra

```
tests/
├── conftest.py          # Fixtures ir global config
├── test_app.py          # 15 pagrindinių testų + papildomi
├── __init__.py
└── README.md            # Šis failas
```

---

## 🔧 Fixtures (conftest.py)

```python
app           # Flask aplikacija test režimui
client        # Flask test_client()
runner        # Flask CLI runner
session_client # Client su sesija
```

---

## ✅ Testų Kategorijos

### 🏠 Routes (HTML Pages)
```python
test_01_index_route_loads()
test_02_search_route_valid_query()
test_03_search_route_invalid_short_query()
test_04_search_route_invalid_long_query()
test_05_game_detail_valid_id()
test_06_game_detail_invalid_id()
test_07_game_detail_nonexistent_id()
```

### 📡 API Endpoints (JSON Responses)
```python
test_08_api_search_endpoint()
test_09_api_search_invalid_query()
test_10_api_explain_endpoint()
test_11_api_clear_history()
```

### 🛡️ Input Validation
```python
test_12_search_weight_validation()
test_13_search_top_k_limits()
```

### 🔍 Search Functionality
```python
test_14_search_modes_all_working()
test_15_search_returns_structured_results()
```

### 🚨 Papildomi Testai (test_app.py)
```python
test_error_handling_malformed_url()
test_error_handling_sql_injection_attempt()
test_search_caching()
test_history_preserves_searches()
```

---

## 🎯 Expected Output

```bash
$ pytest tests/test_app.py -v

tests/test_app.py::TestRoutes::test_01_index_route_loads PASSED         [ 6%]
tests/test_app.py::TestRoutes::test_02_search_route_valid_query PASSED  [13%]
tests/test_app.py::TestRoutes::test_03_search_route_invalid_short_query PASSED [20%]
tests/test_app.py::TestRoutes::test_04_search_route_invalid_long_query PASSED [26%]
tests/test_app.py::TestRoutes::test_05_game_detail_valid_id PASSED      [33%]
tests/test_app.py::TestRoutes::test_06_game_detail_invalid_id PASSED    [40%]
tests/test_app.py::TestRoutes::test_07_game_detail_nonexistent_id PASSED [46%]
tests/test_app.py::TestAPIEndpoints::test_08_api_search_endpoint PASSED [53%]
tests/test_app.py::TestAPIEndpoints::test_09_api_search_invalid_query PASSED [60%]
tests/test_app.py::TestAPIEndpoints::test_10_api_explain_endpoint PASSED [66%]
tests/test_app.py::TestAPIEndpoints::test_11_api_clear_history PASSED  [73%]
tests/test_app.py::TestInputValidation::test_12_search_weight_validation PASSED [80%]
tests/test_app.py::TestInputValidation::test_13_search_top_k_limits PASSED [86%]
tests/test_app.py::TestSearchFunctionality::test_14_search_modes_all_working PASSED [93%]
tests/test_app.py::TestSearchFunctionality::test_15_search_returns_structured_results PASSED [100%]

============== 15 passed in 2.45s ==============
```

---

## 📝 CI/CD Integracija

### GitHub Actions pavyzdys (.github/workflows/test.yml):
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.10
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v --cov=app
```

---

## 🐛 Troubleshooting

### Problema: `ModuleNotFoundError: No module named 'app'`
**Sprendimas:** conftest.py nustato sys.path, bet gali reikėti:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/
```

### Problema: `No such file or directory: '.env'`
**Sprendimas:** Sukurkite .env iš .env.example:
```bash
cp .env.example .env
# Užpildykite DATABASE_URL, GEMINI_API_KEY, SECRET_KEY
```

### Problema: Testo duomenų bazės nėra
**Sprendimas:** Testai naudoja bendrą dev DB. Test'us paleiskite tik kai DB veikia:
```bash
# Patikrinti DB connection
python -c "import sqlalchemy; print('✅ SQLAlchemy OK')"
```

---

## 📚 Recursos

- [pytest docs](https://docs.pytest.org/)
- [Flask testing docs](https://flask.palletsprojects.com/testing/)
- [pytest-flask plugin](https://pytest-flask.readthedocs.io/)

---

## 👨‍💻 Autoriaus Natos

- Testai rašyti lietuviškomis komentarais
- Naudojami pytest fixtures scope'ai efektyvumui
- Input validation testai aprėpia edge cases
- Nėra mock'intos duomenų bazės (integration testai su real DB)

**Maintained:** 2026 m. Steam Recommender project

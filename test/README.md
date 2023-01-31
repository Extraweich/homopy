# Run tests against installed package homopy
Install test requirements
```
pip install pytest
pip install mechkit
```
Install homopy from source code in current directory to current environment in development mode (`-e` flag)
```
python -m pip install -e .
```
Run tests
```
pytest
```

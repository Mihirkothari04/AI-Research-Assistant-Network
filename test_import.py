import sys
try:
    import langchain_community
    print('Import successful')
    print(f'Module location: {langchain_community.__file__}')
except ImportError as e:
    print(f'Import error: {e}')

print(f'Python path: {sys.path}')

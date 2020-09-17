from src import test
method_to_call = getattr(test, 'remove_accents')

print(method_to_call("Thắng"))
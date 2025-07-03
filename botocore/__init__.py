# Minimal stub for `botocore` focusing on `botocore.exceptions` used in tests.

import types, sys as _sys

exceptions = types.ModuleType('botocore.exceptions')

class ClientError(Exception):
    pass

class NoCredentialsError(Exception):
    pass

exceptions.ClientError = ClientError
exceptions.NoCredentialsError = NoCredentialsError

_sys.modules['botocore.exceptions'] = exceptions
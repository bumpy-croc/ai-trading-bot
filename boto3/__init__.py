# Very small stub for the `boto3` AWS SDK – sufficient for tests that only
# check importability.

class Session:
    def client(self, *args, **kwargs):
        return object()

# Provide top-level helpers used commonly

def client(*args, **kwargs):
    return object()


def resource(*args, **kwargs):
    return object()
"""Stand-ins for the Modal objects the runner tests replace: the volume, a spawned call, a function to spawn."""


class FakeVolume:
    def reload(self):
        pass

    def commit(self):
        pass


class FakeCall:
    registry: dict = {}

    def __init__(self, result, fail=None):
        self.object_id = f"fc-{len(FakeCall.registry)}"
        self.result, self.fail = result, fail
        FakeCall.registry[self.object_id] = self

    def get(self):
        if self.fail is not None:
            fail, self.fail = self.fail, None  # the job itself keeps running: a later get() has its result
            raise fail
        return self.result


class FakeFunction:
    def __init__(self, result, fail_first_get=None):
        self.result, self.fail_first_get, self.spawns = result, fail_first_get, 0

    def spawn(self, *args, **kwargs):
        self.spawns += 1
        fail, self.fail_first_get = self.fail_first_get, None
        return FakeCall(self.result(*args, **kwargs) if callable(self.result) else self.result, fail)

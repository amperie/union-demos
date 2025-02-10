from dataclass_defs import HpoResults


def get_best(results: list[HpoResults]) -> HpoResults:
    return max(results, key=lambda x: x.acc)

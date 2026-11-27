"""CPU-only checks for CI implementations supported by CUDA backends."""


def validate_cuda_ci_test(cit, algorithm: str = "CUDA") -> str:
    """Accept only the built-in Fisher-Z and chi-square tests.

    A matching ``method`` name can hide overridden statistical behavior.
    """
    from . import independence_tests as ci

    method = getattr(cit, "method", None)
    builtin = False
    if method == "fisherz":
        builtin = ci._builtin_fisherz(cit)
    elif method == "chisq":
        builtin = (
            type(cit) is ci.CIT
            and type(cit).__call__ is ci._BUILTIN_CIT_CALL
            and getattr(cit.chi_square, "__func__", None) is ci._BUILTIN_CHI_SQUARE
            and ci.chi_square_from_encoded is ci._BUILTIN_CHISQ_KERNEL
            and getattr(cit.batch, "__func__", None) is ci._BUILTIN_CIT_BATCH
            and not any(
                name in cit.__dict__ for name in ("__call__", "chi_square", "batch")
            )
        )
    if not builtin:
        msg = (
            f"{algorithm} requires an unmodified CIT with method='fisherz' "
            "or method='chisq'; custom ci_test callables, subclasses and "
            "method overrides require the CPU path"
        )
        raise NotImplementedError(msg)
    return "fisherz" if method == "fisherz" else "chisq"

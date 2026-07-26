"""Private readout RNG address construction."""

from tensor_core import RngAddress, RngElements, RngKey


def white_noise_address(
    elements: RngElements,
    *,
    key: RngKey,
) -> RngAddress:
    return RngAddress.root(key=key, elements=elements, shape=(), quantum=0)


def psd_noise_address(
    elements: RngElements,
    *,
    key: RngKey,
) -> RngAddress:
    return RngAddress.root(key=key, elements=elements, shape=(), quantum=0)


def dark_count_address(
    elements: RngElements,
    *,
    key: RngKey,
) -> RngAddress:
    return RngAddress.root(key=key, elements=elements, shape=(), quantum=0)


def charge_smearing_address(
    elements: RngElements,
    *,
    key: RngKey,
) -> RngAddress:
    return RngAddress.root(key=key, elements=elements, shape=(), quantum=0)


def timing_jitter_address(
    elements: RngElements,
    *,
    key: RngKey,
    kernel_shape: tuple[int, ...],
) -> RngAddress:
    return RngAddress.root(
        key=key,
        elements=elements,
        shape=kernel_shape,
        quantum=0,
    )


def crosstalk_generation_address(
    elements: RngElements,
    *,
    key: RngKey,
    maximum_generations: int,
    generation_index: int,
) -> RngAddress:
    return RngAddress.root(
        key=key,
        elements=elements,
        shape=(maximum_generations,),
        quantum=0,
    ).select(generation_index)


def afterpulse_occurrence_address(
    elements: RngElements,
    *,
    key: RngKey,
    maximum_generations: int,
    generation_index: int,
) -> RngAddress:
    return RngAddress.root(
        key=key,
        elements=elements,
        shape=(maximum_generations,),
        quantum=0,
    ).select(generation_index)


def afterpulse_delay_address(
    elements: RngElements,
    *,
    key: RngKey,
    maximum_generations: int,
    generation_index: int,
    kernel_shape: tuple[int, ...],
) -> RngAddress:
    return RngAddress.root(
        key=key,
        elements=elements,
        shape=(maximum_generations, *kernel_shape),
        quantum=1,
    ).select(generation_index)

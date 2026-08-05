"""CEC 2005 Benchmark Function Shift Data (Partial).

This data is extracted from the SPSO 2011 C reference code (perf.c).
These represent the 'o' vectors used for shifting the optima.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Sphere Function (F1) Shift Data - offset_0
sphere_shift_data = np.array(
    [
        -3.9311900e001,
        5.8899900e001,
        -4.6322400e001,
        -7.4651500e001,
        -1.6799700e001,
        -8.0544100e001,
        -1.0593500e001,
        2.4969400e001,
        8.9838400e001,
        9.1119000e000,
        -1.0744300e001,
        -2.7855800e001,
        -1.2580600e001,
        7.5930000e000,
        7.4812700e001,
        6.8495900e001,
        -5.3429300e001,
        7.8854400e001,
        -6.8595700e001,
        6.3743200e001,
        3.1347000e001,
        -3.7501600e001,
        3.3892900e001,
        -8.8804500e001,
        -7.8771900e001,
        -6.6494400e001,
        4.4197200e001,
        1.8383600e001,
        2.6521200e001,
        8.4472300e001,
    ]
)

rosenbrock_shift_data = np.array(
    [
        8.1023200e001,
        -4.8395000e001,
        1.9231600e001,
        -2.5231000e000,
        7.0433800e001,
        4.7177400e001,
        -7.8358000e000,
        -8.6669300e001,
        5.7853200e001,
        -9.9533000e000,
        2.0777800e001,
        5.2548600e001,
        7.5926300e001,
        4.2877300e001,
        -5.8272000e001,
        -1.6972800e001,
        7.8384500e001,
        7.5042700e001,
        -1.6151300e001,
        7.0856900e001,
        -7.9579500e001,
        -2.6483700e001,
        5.6369900e001,
        -8.8224900e001,
        -6.4999600e001,
        -5.3502200e001,
        -5.4230000e001,
        1.8682600e001,
        -4.1006100e001,
        -5.4213400e001,
    ]
)

rastrigin_shift_data = np.array(
    [
        1.9005000e000,
        -1.5644000e000,
        -9.7880000e-001,
        -2.2536000e000,
        2.4990000e000,
        -3.2853000e000,
        9.7590000e-001,
        -3.6661000e000,
        9.8500000e-002,
        -3.2465000e000,
        3.8060000e000,
        -2.6834000e000,
        -1.3701000e000,
        4.1821000e000,
        2.4856000e000,
        -4.2237000e000,
        3.3653000e000,
        2.1532000e000,
        -3.0929000e000,
        4.3105000e000,
        -2.9861000e000,
        3.4936000e000,
        -2.7289000e000,
        -4.1266000e000,
        -2.5900000e000,
        1.3124000e000,
        -1.7990000e000,
        -1.1890000e000,
        -1.0530000e-001,
        -3.1074000e000,
    ]
)

schwefel_shift_data = np.array(
    [
        3.5626700e001,
        -8.2912300e001,
        -1.0642300e001,
        -8.3581500e001,
        8.3155200e001,
        4.7048000e001,
        -8.9435900e001,
        -2.7421900e001,
        7.6144800e001,
        -3.9059500e001,
        4.8885700e001,
        -3.9828000e000,
        -7.1924300e001,
        6.4194700e001,
        -4.7733800e001,
        -5.9896000e000,
        -2.6282800e001,
        -5.9181100e001,
        1.4602800e001,
        -8.5478000e001,
        -5.0490100e001,
        9.2400000e-001,
        3.2397800e001,
        3.0238800e001,
        -8.5094900e001,
        6.0119700e001,
        -3.6218300e001,
        -8.5883000e000,
        -5.1971000e000,
        8.1553100e001,
    ]
)

griewank_shift_data = np.array(
    [
        -2.7626840e002,
        -1.1911000e001,
        -5.7878840e002,
        -2.8764860e002,
        -8.4385800e001,
        -2.2867530e002,
        -4.5815160e002,
        -2.0221450e002,
        -1.0586420e002,
        -9.6489800e001,
        -3.9574680e002,
        -5.7294980e002,
        -2.7036410e002,
        -5.6685430e002,
        -1.5242040e002,
        -5.8838190e002,
        -2.8288920e002,
        -4.8888650e002,
        -3.4698170e002,
        -4.5304470e002,
        -5.0658570e002,
        -4.7599870e002,
        -3.6204920e002,
        -2.3323670e002,
        -4.9198640e002,
        -5.4408980e002,
        -7.3445600e001,
        -5.2690110e002,
        -5.0225610e002,
        -5.3723530e002,
    ]
)

ackley_shift_data = np.array(
    [
        -1.6823000e001,
        1.4976900e001,
        6.1690000e000,
        9.5566000e000,
        1.9541700e001,
        -1.7190000e001,
        -1.8824800e001,
        8.5110000e-001,
        -1.5116200e001,
        1.0793400e001,
        7.4091000e000,
        8.6171000e000,
        -1.6564100e001,
        -6.6800000e000,
        1.4543300e001,
        7.0454000e000,
        -1.8621500e001,
        1.4556100e001,
        -1.1594200e001,
        -1.9153100e001,
        -4.7372000e000,
        9.2590000e-001,
        1.3241200e001,
        -5.2947000e000,
        1.8416000e000,
        4.5618000e000,
        -1.8890500e001,
        9.8008000e000,
        -1.5426500e001,
        1.2722000e000,
    ]
)

# Dictionary mapping problem name (lowercase) to shift data
CEC_SHIFT_DATA = {
    "sphere": sphere_shift_data,
    "rosenbrock": rosenbrock_shift_data,
    "rastrigin": rastrigin_shift_data,
    "schwefel": schwefel_shift_data,
    "griewank": griewank_shift_data,
    "ackley": ackley_shift_data,
    # Add other functions later if needed
}


def get_cec_shift_vector(problem_name: str, dimension: int) -> np.ndarray | None:
    """Gets the CEC shift vector for a given problem and dimension.
    Args:
        problem_name: Lowercase name of the problem (e.g., 'sphere').
        dimension: The required dimension.
    Returns:
        A numpy array of the shift vector for the specified dimension,
        or None if data is not available for this problem or dimension.
    """
    problem_name = problem_name.lower()
    if problem_name in CEC_SHIFT_DATA:
        data = CEC_SHIFT_DATA[problem_name]
        if dimension <= len(data):
            return np.asarray(data[:dimension], dtype=np.float64)
        else:
            # Data only defined up to dimension 30 in the source code
            logger.warning(
                "CEC shift data for '%s' only available up to D=%s, requested D=%s. No shift applied.",
                problem_name,
                len(data),
                dimension,
            )
            return None
    return None

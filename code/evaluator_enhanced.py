import os
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from subprocess import TimeoutExpired
from typing import Callable, List

MODELS_PATH = "../mnist_nets/"
BASE_TESTCASES_DIR = "../test_cases/"
BASE_TESTCASES_PATH = "../test_cases/gt.txt"
PRELIM_TESTCASES_DIR = "../prelim_test_cases/"
PRELIM_TESTCASES_PATH = "../prelim_results.txt"
TIMEOUT = 60.0


class BColors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class ResultType(Enum):
    VERIFIED = "verified"
    NOT_VERIFIED = "not verified"
    TIMEOUT = "timeout (not verified)"
    ERROR = "error"


result_types_strs = {
    ResultType.VERIFIED.value: ResultType.VERIFIED,
    ResultType.NOT_VERIFIED.value: ResultType.NOT_VERIFIED,
}


@dataclass
class TestCase:
    model_path: str
    input_path: str
    expected: str


@dataclass
class RunResult:
    result: ResultType
    time: float
    command: str


def line_converter(testcases_dir: str) -> Callable[[str], TestCase]:
    def convert(line: str) -> TestCase:
        parts = line.split(",")
        return TestCase(
            model_path=os.path.join(MODELS_PATH, parts[0]),
            input_path=os.path.join(testcases_dir, parts[0], parts[1]),
            expected=parts[2].strip(),
        )

    return convert


def load_base_cases() -> List[TestCase]:
    with open(BASE_TESTCASES_PATH) as f:
        return list(map(line_converter(BASE_TESTCASES_DIR), f.readlines()))


def load_prelim_cases() -> List[TestCase]:
    with open(PRELIM_TESTCASES_PATH) as f:
        return list(map(line_converter(PRELIM_TESTCASES_DIR), f.readlines()[1:26]))


def main() -> None:
    total_time, correct = 0.0, 0
    test_cases = [*load_base_cases(), *load_prelim_cases()]
    for idx, test_case in enumerate(test_cases):
        run_result = external_caller(test_case.model_path, test_case.input_path)
        is_correct = run_result.result.value == test_case.expected or (
            run_result.result == ResultType.TIMEOUT
            and test_case.expected == "not verified"
        )
        print_str = f"TEST [{idx+1: 02d}/{len(test_cases)}]: "
        print_str += (
            (BColors.OKGREEN + "OK" + BColors.ENDC)
            if is_correct
            else (BColors.FAIL + "FAIL" + BColors.ENDC)
        )
        print_str += f" ({run_result.time: .2f} s)"
        print_str += (
            f" (expected: {test_case.expected}, got: {run_result.result.value})"
        )
        print_str += f" [command: {run_result.command}]" if not is_correct else ""
        print(print_str)
        correct += 1 if is_correct else 0
        if test_case.expected == "verified":
            total_time += run_result.time
    print(
        f"Total: {correct}/{len(test_cases)} correct, {total_time: .2f} s for verified examples"
    )


def external_caller(model_path: str, spec_path: str) -> RunResult:
    command = ["python", "verifier.py", "--net", model_path, "--spec", spec_path]

    try:
        start_time = time.time()
        completed_process = subprocess.run(
            command, capture_output=True, timeout=TIMEOUT
        )
        total_time = time.time() - start_time
    except TimeoutExpired as e:
        return RunResult(ResultType.TIMEOUT, e.timeout, " ".join(command))

    stdout_decoded = completed_process.stdout.decode(encoding="utf-8")
    # stderr_decoded = completed_process.stderr.decode(encoding="utf-8")
    result_str = stdout_decoded.split("/n")[-1].strip()
    result = result_types_strs.get(result_str, ResultType.ERROR)
    return RunResult(result, total_time, " ".join(command))


if __name__ == "__main__":
    main()

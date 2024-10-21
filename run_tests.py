import unittest
from unittest.signals import registerResult
import time
from gradescope_utils.autograder_utils.json_test_runner import JSONTestRunner
import argparse
import os


class TextRunner(JSONTestRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, test):
        """Run the given test case or test suite."""
        result = self._makeResult()
        registerResult(result)
        result.failfast = self.failfast
        result.buffer = self.buffer
        startTime = time.time()
        startTestRun = getattr(result, "startTestRun", None)
        if startTestRun is not None:
            startTestRun()
        try:
            test(result)
        finally:
            stopTestRun = getattr(result, "stopTestRun", None)
            if stopTestRun is not None:
                stopTestRun()
        stopTime = time.time()
        timeTaken = stopTime - startTime

        self.json_data["execution_time"] = format(timeTaken, "0.2f")

        total_score = 0
        max_total_score = 0

        for test in self.json_data["tests"]:
            total_score += test.get("score", 0.0)
            max_total_score += test.get("max_score", 0.0)
        self.json_data["score"] = total_score

        if self.post_processor is not None:
            self.post_processor(self.json_data)

        print("Test results:")

        # Sort the tests by number
        sorted_tests = sorted(self.json_data["tests"], key=lambda x: x["number"])
        for test in sorted_tests:
            print(
                f"{test['number']}) {test['name']} ({test['score']}/{test['max_score']}) ... {test['status']}"
            )
            if "output" in test:
                print(test["output"])

        print(f"Total score: {total_score}/{max_total_score}")
        print(f"Execution time: {self.json_data['execution_time']}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtype",
        type=str,
        default="local",
        choices=["local", "gradescope"],
        help="Whether tests are being run in gradescope or locally. If running in gradescope, "
        "the results will be written to results.json. If running locally, the results will "
        "be printed to the console.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="The seed to be used when testing"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    seed = args.seed
    os.environ["TEST_SEED"] = str(seed)

    # assert os.path.exists(
    #     f"test_files_seed={seed}"
    # ), f"No test files directory found with seed={seed}"

    suite = unittest.defaultTestLoader.discover("tests")
    if args.runtype == "local":
        TextRunner(visibility="visible").run(suite)
    else:
        with open("/autograder/results/results.json", "w") as f:
            JSONTestRunner(visibility="visible", stream=f).run(suite)

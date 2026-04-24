"""
Machine Learning Project - Main Entry
Usage:
  python main.py --algo=svm --data=titanic --process=train
  python main.py --algo=logistic --data=mnist --process=train
  python main.py --algo=knn --data=cifar10 --process=train
  python main.py --algo=linear --data=house --process=train
  python main.py --algo=all --data=all --process=all
  python main.py --help
"""

import subprocess
import sys
import os
import argparse


def run_script(script_name, description):
    print("\n" + "=" * 60)
    print(f"Running: {description}")
    print("=" * 60)
    src_dir = os.path.join(os.path.dirname(__file__), "src")
    script_path = os.path.join(src_dir, script_name)
    result = subprocess.run([sys.executable, script_path], cwd=src_dir)
    if result.returncode != 0:
        print(f"  [FAILED] {script_name} exited with code {result.returncode}")
    else:
        print(f"  [DONE] {description}")
    return result.returncode


def show_results():
    result_path = os.path.join(os.path.dirname(__file__), "results", "accuracy.txt")
    if os.path.exists(result_path):
        print("\n" + "=" * 60)
        print("All Results Summary")
        print("=" * 60)
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read()
        print(content)
    else:
        print("\nNo results found. Run some models first.")


ALGO_DATA_MAP = {
    ("svm", "titanic"): {
        "SVM Titanic Train": "train.py",
        "SVM Titanic Test": "test.py",
    },
    ("logistic", "titanic"): {
        "Titanic Logistic Regression": "logistic_regression.py",
    },
    ("linear", "house"): {
        "House Price Linear Regression": "linear_regression.py",
    },
    ("logistic", "mnist"): {
        "MNIST Logistic Regression": "logistic_mnist.py",
    },
    ("svm", "mnist"): {
        "MNIST SVM": "svm_mnist.py",
    },
    ("knn", "cifar10"): {
        "CIFAR-10 KNN": "KNN.py",
    },
    ("svm", "cifar10"): {
        "CIFAR-10 SVM": "SVM.py",
    },
}


def run_single(algo, data, process):
    key = (algo, data)
    if key not in ALGO_DATA_MAP:
        print(f"Error: unsupported combination --algo={algo} --data={data}")
        print("Use --help to see available options.")
        return

    scripts = ALGO_DATA_MAP[key]
    for desc, script in scripts.items():
        if process == "all" or process == "train":
            run_script(script, desc)
            break
        elif process == "test" and "test" in script:
            run_script(script, desc)
            break


def run_all():
    print("\nRunning ALL models (this may take a while)...")
    all_entries = [
        ("SVM Titanic Train", "train.py"),
        ("SVM Titanic Test", "test.py"),
        ("Titanic Logistic Regression", "logistic_regression.py"),
        ("MNIST Logistic Regression", "logistic_mnist.py"),
        ("MNIST SVM", "svm_mnist.py"),
        ("CIFAR-10 KNN", "KNN.py"),
        ("CIFAR-10 SVM", "SVM.py"),
    ]
    for desc, script in all_entries:
        run_script(script, desc)
    show_results()


def main():
    parser = argparse.ArgumentParser(
        description="Machine Learning Project",
        epilog="Examples:\n"
               "  python main.py --algo=svm --data=titanic --process=train\n"
               "  python main.py --algo=logistic --data=mnist --process=train\n"
               "  python main.py --algo=knn --data=cifar10 --process=train\n"
               "  python main.py --algo=all --data=all --process=all",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--algo", type=str, default="all",
                        help="Algorithm: svm, logistic, linear, knn, all")
    parser.add_argument("--data", type=str, default="all",
                        help="Dataset: titanic, house, mnist, cifar10, all")
    parser.add_argument("--process", type=str, default="all",
                        help="Process: train, test, all")

    args = parser.parse_args()

    print("=" * 60)
    print("Machine Learning Project")
    print("=" * 60)
    print(f"Algorithm: {args.algo}")
    print(f"Dataset:   {args.data}")
    print(f"Process:   {args.process}")
    print("=" * 60)

    if args.algo == "all" or args.data == "all":
        run_all()
    else:
        run_single(args.algo, args.data, args.process)


if __name__ == "__main__":
    main()

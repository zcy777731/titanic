"""
Machine Learning Project - Main Entry
Covers: Titanic, House Price, MNIST, CIFAR-10
Models: Linear Regression, Logistic Regression, SVM, KNN
"""

import subprocess
import sys
import os


def run_script(script_name, description):
    print("\n" + "=" * 60)
    print(f"Running: {description}")
    print("=" * 60)
    script_path = os.path.join(os.path.dirname(__file__), "src", script_name)
    result = subprocess.run([sys.executable, script_path], cwd=os.path.dirname(__file__))
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
            print(f.read())
    else:
        print("\nNo results found. Run some models first.")


if __name__ == "__main__":
    print("=" * 60)
    print("Machine Learning Project")
    print("=" * 60)
    print("\nAvailable models:")
    print("  1. Titanic - SVM")
    print("  2. Titanic - Logistic Regression")
    print("  3. House Price - Linear Regression")
    print("  4. MNIST - Logistic Regression")
    print("  5. MNIST - SVM")
    print("  6. CIFAR-10 - KNN")
    print("  7. CIFAR-10 - SVM")
    print("  8. Run ALL models")
    print("  9. Show saved results")
    print("  0. Exit")

    while True:
        try:
            choice = input("\nSelect model to run (0-9): ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == "0":
            break
        elif choice == "1":
            run_script("train.py", "Titanic SVM Train")
            run_script("test.py", "Titanic SVM Test")
        elif choice == "2":
            run_script("logistic_regression.py", "Titanic Logistic Regression")
        elif choice == "3":
            run_script("linear_regression.py", "House Price Linear Regression")
        elif choice == "4":
            run_script("logistic_mnist.py", "MNIST Logistic Regression")
        elif choice == "5":
            run_script("svm_mnist.py", "MNIST SVM")
        elif choice == "6":
            run_script("KNN.py", "CIFAR-10 KNN")
        elif choice == "7":
            run_script("SVM.py", "CIFAR-10 SVM")
        elif choice == "8":
            print("\nRunning ALL models (this may take a while)...")
            run_script("train.py", "Titanic SVM Train")
            run_script("test.py", "Titanic SVM Test")
            run_script("logistic_regression.py", "Titanic Logistic Regression")
            run_script("logistic_mnist.py", "MNIST Logistic Regression")
            run_script("svm_mnist.py", "MNIST SVM")
            run_script("KNN.py", "CIFAR-10 KNN")
            run_script("SVM.py", "CIFAR-10 SVM")
            show_results()
        elif choice == "9":
            show_results()
        else:
            print("Invalid choice. Please enter 0-9.")

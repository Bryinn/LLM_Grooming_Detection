import csv
import sys
from tabulate import tabulate


def print_csv_formatted(csv_path):
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)
        if not rows:
            print("CSV file is empty.")
            return
        # Print the CSV content in a formatted table using in acending order by F1 score
        rows[1:] = sorted(rows[1:], key=lambda x: float(x[8]), reverse=True)  # Assuming F1 score
        print(tabulate(rows[1:], headers=rows[0], tablefmt="grid", showindex=True))


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <results.csv>")
        sys.exit(1)
    csv_path = sys.argv[1]
    print_csv_formatted(csv_path)


if __name__ == "__main__":
    main()

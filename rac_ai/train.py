import argparse
from .modeling import train_all


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate 30- and 60-minute RAC room-temperature models.")
    parser.add_argument("--data", default="data/sample_temperature_data.csv", help="Path to CSV dataset.")
    parser.add_argument("--artifacts", default="artifacts", help="Output folder for trained models.")
    parser.add_argument("--reports", default="reports", help="Output folder for metrics and plots.")
    args = parser.parse_args()
    train_all(args.data, args.artifacts, args.reports)

if __name__ == "__main__":
    main()

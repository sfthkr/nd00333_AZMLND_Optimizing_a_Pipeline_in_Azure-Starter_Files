from sklearn.linear_model import LogisticRegression
import argparse
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split


def clean_data(data):
    months = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4,
        "may": 5, "jun": 6, "jul": 7, "aug": 8,
        "sep": 9, "oct": 10, "nov": 11, "dec": 12
    }

    weekdays = {
        "mon": 1, "tue": 2, "wed": 3, "thu": 4,
        "fri": 5, "sat": 6, "sun": 7
    }

    x_df = data.dropna().copy()

    jobs = pd.get_dummies(x_df.job, prefix="job")
    x_df.drop("job", inplace=True, axis=1)
    x_df = x_df.join(jobs)

    x_df["marital"] = x_df.marital.apply(
        lambda s: 1 if s == "married" else 0
    )

    x_df["default"] = x_df.default.apply(
        lambda s: 1 if s == "yes" else 0
    )

    x_df["housing"] = x_df.housing.apply(
        lambda s: 1 if s == "yes" else 0
    )

    x_df["loan"] = x_df.loan.apply(
        lambda s: 1 if s == "yes" else 0
    )

    contact = pd.get_dummies(x_df.contact, prefix="contact")
    x_df.drop("contact", inplace=True, axis=1)
    x_df = x_df.join(contact)

    education = pd.get_dummies(x_df.education, prefix="education")
    x_df.drop("education", inplace=True, axis=1)
    x_df = x_df.join(education)

    x_df["month"] = x_df.month.map(months)
    x_df["day_of_week"] = x_df.day_of_week.map(weekdays)

    x_df["poutcome"] = x_df.poutcome.apply(
        lambda s: 1 if s == "success" else 0
    )

    y_df = x_df.pop("y").apply(
        lambda s: 1 if s == "yes" else 0
    )

    return x_df, y_df


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse of regularization strength"
    )

    parser.add_argument(
        "--max_iter",
        type=int,
        default=100,
        help="Maximum number of iterations"
    )

    args = parser.parse_args()

    # Log hyperparameters
    mlflow.log_param("C", args.C)
    mlflow.log_param("max_iter", args.max_iter)

    # Load data
    url = (
        "https://automlsamplenotebookdata.blob.core.windows.net/"
        "automl-sample-notebook-data/bankmarketing_train.csv"
    )

    data = pd.read_csv(url)

    # Clean data
    x, y = clean_data(data)

    # Train/test split
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=0
    )

    # Train model
    model = LogisticRegression(
        C=args.C,
        max_iter=args.max_iter
    )

    model.fit(x_train, y_train)

    # Evaluate
    accuracy = model.score(x_test, y_test)

    print(f"C: {args.C}")
    print(f"max_iter: {args.max_iter}")
    print(f"Accuracy: {accuracy}")

    # Log metric
    mlflow.log_metric(
        "Accuracy",
        float(accuracy)
    )


if __name__ == "__main__":
    main()
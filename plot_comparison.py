import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

train = pd.read_csv("data_train_engineered.csv", parse_dates=["Date"])
submission_real = pd.read_csv("submission_real.csv", parse_dates=["Date"])
submission_pred = pd.read_csv("submission.csv", parse_dates=["Date"])

train = train[["Date", "USDIDR"]].rename(columns={"USDIDR": "Actual (Train)"})
submission_real = submission_real[["Date", "USDIDR"]].rename(columns={"USDIDR": "Actual (Submission)"})
submission_pred = submission_pred[["Date", "USDIDR"]].rename(columns={"USDIDR": "Predicted"})

plt.figure(figsize=(16, 6))
plt.plot(train["Date"], train["Actual (Train)"], label="Actual (Train)", color="steelblue", linewidth=1)
plt.plot(submission_real["Date"], submission_real["Actual (Submission)"], label="Actual (Submission)", color="green", linewidth=1.5)
plt.plot(submission_pred["Date"], submission_pred["Predicted"], label="Predicted", color="red", linewidth=1.5, linestyle="--")

plt.axvline(submission_real["Date"].iloc[0], color="gray", linestyle=":", alpha=0.7, label="Submission Start")
plt.title("USD/IDR Forecast vs Actual")
plt.xlabel("Date")
plt.ylabel("USDIDR")
plt.legend()
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
plt.savefig("comparison_chart.png", dpi=150)
plt.show()
print("Saved to comparison_chart.png")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from all_imports import BASE_DIR, CSV_PATH

IMAGES_DIR = BASE_DIR / "images" / "eda"


def save_categorical_plots(df: pd.DataFrame) -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

    for col in [c for c in cat_cols if c in df.columns]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        sns.countplot(data=df, x=col, hue="dataset", palette="Set2", ax=axes[0])
        axes[0].set_title(f"{col} by dataset source")
        axes[0].tick_params(axis="x", rotation=30)

        sns.countplot(data=df, x=col, hue="sex", palette="Set1", ax=axes[1])
        axes[1].set_title(f"{col} by sex")
        axes[1].tick_params(axis="x", rotation=30)

        fig.tight_layout()
        out = IMAGES_DIR / f"{col}_dist.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out.name}")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="num", data=df, ax=ax)
    ax.set_title("Heart Disease Distribution (target)")
    ax.set_xlabel("0 = No Disease  |  1–4 = Disease Severity")
    out = IMAGES_DIR / "target_distribution.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.name}")


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    save_categorical_plots(df)
    print(f"\nAll plots saved to {IMAGES_DIR}")

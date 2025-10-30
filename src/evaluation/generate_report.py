#!/usr/bin/env python3
"""
Evaluation Report Generator for SusTech Recycling Agent

Generates detailed reports and visualizations from evaluation results.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd
from typing import Dict, Any, List
import os


class EvaluationReportGenerator:
    """Generates comprehensive evaluation reports."""

    def __init__(self, results_file: str = "evaluation_results.json"):
        self.results_file = results_file
        self.data = None
        self.stats = None

    def load_results(self):
        """Load evaluation results from file."""
        try:
            with open(self.results_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.data = data["results"]
                self.stats = data["statistics"]
            print(f"Loaded {len(self.data)} evaluation results")
        except FileNotFoundError:
            print(f"Error: {self.results_file} not found")
            return False
        return True

    def generate_summary_report(self) -> str:
        """Generate a text summary report."""
        if not self.stats:
            return "No data loaded"

        report = []
        report.append("=" * 60)
        report.append("SUSTECH RECYCLING AGENT - EVALUATION REPORT")
        report.append("=" * 60)

        overall = self.stats["overall"]
        report.append("\nOVERALL PERFORMANCE")
        report.append("-" * 30)
        report.append(".2%")
        report.append(f"Total Test Cases: {overall['total_cases']}")
        report.append(f"Correct Answers: {overall['correct_cases']}")
        report.append(f"Error Cases: {overall['error_cases']}")
        report.append(".2%")

        # Regional performance
        report.append("\nREGIONAL PERFORMANCE")
        report.append("-" * 30)
        for region, data in self.stats["by_region"].items():
            report.append(f"{region}:")
            report.append(".2%")
            report.append(f"  Cases: {data['total']}")

        # Language performance
        report.append("\nLANGUAGE PERFORMANCE")
        report.append("-" * 30)
        for language, data in self.stats["by_language"].items():
            lang_name = "English" if language == "en" else "German"
            report.append(f"{lang_name}:")
            report.append(".2%")
            report.append(f"  Cases: {data['total']}")

        # Category performance
        report.append("\nCATEGORY PERFORMANCE")
        report.append("-" * 30)
        for category, data in self.stats["by_category"].items():
            report.append(f"{category.title()}:")
            report.append(".2%")
            report.append(f"  Cases: {data['total']}")

        # Performance metrics
        if "performance_metrics" in self.stats:
            pm = self.stats["performance_metrics"]
            report.append("\nPERFORMANCE METRICS")
            report.append("-" * 30)
            report.append(".3f")
            report.append(".3f")
            report.append(".3f")

        # Error analysis
        errors = [r for r in self.data if r["error"] is not None]
        if errors:
            report.append("\nERROR ANALYSIS")
            report.append("-" * 30)
            report.append(f"Total Errors: {len(errors)}")

            error_types = defaultdict(int)
            for error in errors:
                error_type = error["error"].split(":")[0] if ":" in error["error"] else "Unknown"
                error_types[error_type] += 1

            for error_type, count in error_types.items():
                report.append(f"  {error_type}: {count}")

        # Sample incorrect answers
        incorrect = [r for r in self.data if not r["overall_correct"] and r["error"] is None]
        if incorrect:
            report.append("\nSAMPLE INCORRECT ANSWERS")
            report.append("-" * 30)
            for i, case in enumerate(incorrect[:5]):  # Show first 5
                report.append(f"\nCase {i+1}:")
                report.append(f"  Question: {case['question']}")
                report.append(f"  Expected: {case['expected_bin']}")
                report.append(f"  Got: {case['simplified_response']}")
                report.append(f"  Region: {case['region']}")

        report.append("\n" + "=" * 60)
        return "\n".join(report)

    def generate_detailed_csv(self, filename: str = "evaluation_details.csv"):
        """Generate detailed CSV report."""
        if not self.data:
            return

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(self.data)

        # Select relevant columns
        columns = ["question", "expected_bin", "expected_instructions", "simplified_response", "overall_correct", "bin_correct", "instructions_correct", "response_time", "region", "language", "category", "error"]

        df = df[columns]
        df.to_csv(filename, index=False, encoding="utf-8")
        print(f"Detailed CSV report saved to {filename}")

    def generate_visualizations(self, output_dir: str = "evaluation_charts"):
        """Generate visualization charts."""
        if not self.data or not self.stats:
            return

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Set style
        plt.style.use("default")
        sns.set_palette("husl")

        # 1. Overall accuracy pie chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Overall accuracy
        overall = self.stats["overall"]
        correct = overall["correct_cases"]
        incorrect = overall["total_cases"] - correct - overall["error_cases"]
        errors = overall["error_cases"]

        ax1.pie([correct, incorrect, errors], labels=["Correct", "Incorrect", "Errors"], autopct="%1.1f%%", startangle=90)
        ax1.set_title("Overall Accuracy Distribution")
        ax1.axis("equal")

        # Regional accuracy
        regions = list(self.stats["by_region"].keys())
        accuracies = [self.stats["by_region"][r]["accuracy"] * 100 for r in regions]

        bars = ax2.bar(regions, accuracies, color=["#1f77b4", "#ff7f0e"])
        ax2.set_title("Accuracy by Region")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_ylim(0, 100)

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, ".1f", ha="center", va="bottom")

        # Language accuracy
        languages = ["English" if l == "en" else "German" for l in self.stats["by_language"].keys()]
        lang_accuracies = [self.stats["by_language"][l]["accuracy"] * 100 for l in self.stats["by_language"].keys()]

        bars = ax3.bar(languages, lang_accuracies, color=["#2ca02c", "#d62728"])
        ax3.set_title("Accuracy by Language")
        ax3.set_ylabel("Accuracy (%)")
        ax3.set_ylim(0, 100)

        # Add value labels on bars
        for bar, acc in zip(bars, lang_accuracies):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, ".1f", ha="center", va="bottom")

        # Category accuracy
        categories = list(self.stats["by_category"].keys())
        cat_accuracies = [self.stats["by_category"][c]["accuracy"] * 100 for c in categories]

        ax4.barh(categories, cat_accuracies, color="#9467bd")
        ax4.set_title("Accuracy by Category")
        ax4.set_xlabel("Accuracy (%)")
        ax4.set_xlim(0, 100)

        # Add value labels on bars
        for i, acc in enumerate(cat_accuracies):
            ax4.text(acc + 1, i, ".1f", va="center")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/evaluation_summary.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 2. Response time distribution
        if "performance_metrics" in self.stats:
            plt.figure(figsize=(10, 6))
            response_times = [r["response_time"] for r in self.data if r["error"] is None]

            plt.hist(response_times, bins=20, alpha=0.7, color="#17becf", edgecolor="black")
            plt.title("Response Time Distribution")
            plt.xlabel("Response Time (seconds)")
            plt.ylabel("Frequency")
            plt.axvline(self.stats["performance_metrics"]["avg_response_time"], color="red", linestyle="--", label=".3f")
            plt.legend()
            plt.savefig(f"{output_dir}/response_times.png", dpi=300, bbox_inches="tight")
            plt.close()

        # 3. Error analysis (if errors exist)
        errors = [r for r in self.data if r["error"] is not None]
        if errors:
            plt.figure(figsize=(10, 6))
            error_types = defaultdict(int)
            for error in errors:
                error_type = error["error"].split(":")[0] if ":" in error["error"] else "Unknown"
                error_types[error_type] += 1

            plt.bar(error_types.keys(), error_types.values(), color="#e74c3c")
            plt.title("Error Types Distribution")
            plt.xlabel("Error Type")
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/error_analysis.png", dpi=300, bbox_inches="tight")
            plt.close()

        print(f"Visualizations saved to {output_dir}/")

    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []

        if not self.stats:
            return recommendations

        overall_acc = self.stats["overall"]["accuracy"]

        if overall_acc < 0.7:
            recommendations.append("🔴 CRITICAL: Overall accuracy is below 70%. Major improvements needed.")
        elif overall_acc < 0.8:
            recommendations.append("🟡 WARNING: Overall accuracy is below 80%. Consider improvements.")
        else:
            recommendations.append("🟢 GOOD: Overall accuracy is above 80%.")

        # Regional differences
        regions = self.stats["by_region"]
        if len(regions) > 1:
            accuracies = [data["accuracy"] for data in regions.values()]
            max_diff = max(accuracies) - min(accuracies)
            if max_diff > 0.2:
                recommendations.append(".2f" "Consider region-specific improvements.")

        # Language differences
        languages = self.stats["by_language"]
        if len(languages) > 1:
            accuracies = [data["accuracy"] for data in languages.values()]
            max_diff = max(accuracies) - min(accuracies)
            if max_diff > 0.2:
                recommendations.append(".2f" "Consider language-specific improvements.")

        # Category performance
        categories = self.stats["by_category"]
        poor_categories = [cat for cat, data in categories.items() if data["accuracy"] < 0.7]
        if poor_categories:
            recommendations.append(f"Focus improvement efforts on these categories: {', '.join(poor_categories)}")

        # Error analysis
        error_rate = self.stats["overall"]["error_rate"]
        if error_rate > 0.1:
            recommendations.append(".2f" "Investigate and fix system errors.")

        # Performance
        if "performance_metrics" in self.stats:
            avg_time = self.stats["performance_metrics"]["avg_response_time"]
            if avg_time > 5.0:
                recommendations.append(".1f" "Consider performance optimizations.")

        return recommendations


def main():
    """Main function to generate reports."""
    generator = EvaluationReportGenerator()

    if not generator.load_results():
        return

    # Generate text report
    report = generator.generate_summary_report()
    with open("evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print("Text report saved to evaluation_report.txt")

    # Generate CSV report
    generator.generate_detailed_csv()

    # Generate visualizations
    generator.generate_visualizations()

    # Generate recommendations
    recommendations = generator.generate_recommendations()

    print("\n" + "=" * 50)
    print("RECOMMENDATIONS")
    print("=" * 50)
    for rec in recommendations:
        print(f"• {rec}")

    print("\nReport generation complete!")


if __name__ == "__main__":
    main()

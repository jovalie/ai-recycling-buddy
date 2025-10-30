#!/usr/bin/env python3
"""
Evaluation Runner for SusTech Recycling Agent

Runs the evaluation test suite against the AI recycling assistant
and measures accuracy across different dimensions.
"""

import json
import asyncio
import time
from typing import List, Dict, Any, Tuple
import re
from collections import defaultdict
import sys
import os
import requests
import subprocess
import signal
import time as time_module
import argparse


class ResponseSimplifier:
    """Categorizes AI responses into material categories for evaluation."""

    def __init__(self):
        # Material category keywords and bin mappings with weights
        self.category_keywords = {
            "glass (Glas)": {"primary": ["glas", "glass", "glasflasche", "glass bottle", "glass jar", "altglas"], "secondary": ["glascontainer", "altglascontainer", "glasiglus", "glass recycling bin", "glass bin"], "german_compounds": ["glasflaschen", "glasbehälter", "glascontainer"], "weight": 1.0},
            "paper (Papier)": {"primary": ["papier", "paper", "karton", "cardboard", "zeitung", "newspaper", "altpapier", "tetra pak", "milk carton", "pizza box", "pizzakarton"], "secondary": ["blaue tonne", "altpapier", "papiercontainer", "papiertonne", "blue bin", "paper recycling bin", "cardboard bin"], "german_compounds": ["papierschnipsel", "kartons", "zeitungen"], "weight": 1.0},
            "plastic (Kunststoff)": {"primary": ["kunststoff", "plastic", "plastik", "plastikflasche", "plastic bottle", "plastic bag", "styropor", "styrofoam", "polystyrene", "cd", "dvd"], "secondary": ["gelbe tonne", "gelber sack", "wertstofftonne", "kunststoffcontainer", "yellow bin", "plastic recycling bin", "recycling bin"], "german_compounds": ["plastikflaschen", "plastiktüten", "kunststoffe"], "weight": 1.0},
            "metal (Metall)": {
                "primary": ["metall", "metal", "aluminum", "aluminium", "aludose", "aluminium can", "scrap metal", "tin can", "metal can"],
                "secondary": ["gelbe tonne", "gelber sack", "wertstofftonne", "metallcontainer", "yellow bin", "metal recycling bin", "can recycling"],
                "german_compounds": ["aluminiumdosen", "metallverpackungen", "blechdosen"],
                "weight": 1.2,  # Higher weight for metal since it's often missed
            },
            "hazardous (Sondermüll)": {
                "primary": ["sondermüll", "hazardous", "special collection", "sammelstelle", "light bulb", "glühbirne", "batteries", "battery", "chemicals", "quecksilber", "mercury"],
                "secondary": ["sondermüll", "sammelstelle", "wertstoffhof", "recyclinghof", "special waste", "hazardous waste collection", "electronics recycling"],
                "german_compounds": ["altbatterien", "gefahrenstoffe", "elektronikschrott"],
                "weight": 1.0,
            },
        }

    def categorize_response(self, response: str) -> str:
        """Categorize a response into a material category using intelligent matching."""
        response = response.lower().strip()

        # Score each category based on keyword matches with weights
        scores = {}
        max_score = 0

        for category, keyword_data in self.category_keywords.items():
            score = 0

            # Check primary keywords (highest weight)
            for keyword in keyword_data["primary"]:
                count = response.count(keyword.lower())
                if count > 0:
                    score += count * 3.0  # Primary keywords get 3x weight

            # Check secondary keywords (medium weight)
            for keyword in keyword_data["secondary"]:
                count = response.count(keyword.lower())
                if count > 0:
                    score += count * 1.5  # Secondary keywords get 1.5x weight

            # Check German compound words (high weight for compounds)
            for keyword in keyword_data["german_compounds"]:
                count = response.count(keyword.lower())
                if count > 0:
                    score += count * 2.5  # Compound words get 2.5x weight

            # Apply category weight multiplier
            score *= keyword_data["weight"]

            scores[category] = score
            max_score = max(max_score, score)

        # Only return a category if we have a meaningful score (> 0.5)
        # This prevents false positives from incidental keyword matches
        if max_score > 0.5:
            best_category = max(scores, key=scores.get)
            return best_category.title()

        return "Unknown"


class ServerManager:
    """Manages the FastAPI server for evaluation."""

    def __init__(self, host: str = "localhost", port: int = 8000, verbose: bool = False):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.server_process = None
        self.verbose = verbose

    def is_server_running(self) -> bool:
        """Check if the server is running."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def start_server(self) -> bool:
        """Start the FastAPI server."""
        if self.is_server_running():
            print("✓ Using existing server instance")
            return True

        try:
            print("🚀 Starting new FastAPI server...")
            # Use the same command as the Makefile
            # PYTHONPATH=. poetry run uvicorn src.serve:app --reload
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            cmd = ["poetry", "run", "uvicorn", "src.serve:app", "--reload"]
            env = os.environ.copy()
            env["PYTHONPATH"] = project_root

            if self.verbose:
                print(f"[VERBOSE] Project root: {project_root}")
                print(f"[VERBOSE] Command: {' '.join(cmd)}")
                print(f"[VERBOSE] PYTHONPATH: {env['PYTHONPATH']}")

            # Start the server as a subprocess
            self.server_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=project_root, env=env)

            # Wait for server to start
            max_attempts = 30
            for attempt in range(max_attempts):
                time_module.sleep(2)
                if self.is_server_running():
                    print("✅ Server started successfully")
                    return True
                if self.verbose:
                    print(f"[VERBOSE] Waiting for server to start... ({attempt + 1}/{max_attempts})")
                else:
                    print(f"⏳ Waiting for server to start... ({attempt + 1}/{max_attempts})")

            print("❌ Failed to start server within timeout")
            self.stop_server()
            return False

        except Exception as e:
            print(f"❌ Error starting server: {e}")
            return False

    def stop_server(self):
        """Stop the FastAPI server."""
        if self.server_process:
            try:
                print("🛑 Stopping server...")
                self.server_process.terminate()
                self.server_process.wait(timeout=10)
                print("✅ Server stopped")
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                print("⚠️  Server force killed")
            except Exception as e:
                print(f"❌ Error stopping server: {e}")
            finally:
                self.server_process = None
        else:
            # Server was already running when we started, don't try to stop it
            print("ℹ  Server was already running - leaving it active")

    def query_server(self, question: str, region: str, timeout: int = 60) -> Tuple[str, Dict]:
        """Query the server with a question."""
        payload = {"question": question, "chat_history": [], "region": region}

        if self.verbose:
            print(f"[VERBOSE] Sending request to {self.base_url}/query")
            print(f"[VERBOSE] Payload: region={region}, question_length={len(question)}")

        try:
            response = requests.post(f"{self.base_url}/query", json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()

            if self.verbose:
                print(f"[VERBOSE] Response received: {len(data.get('response', ''))} chars")

            return data["response"], data["usage"]
        except requests.RequestException as e:
            raise Exception(f"Server query failed: {e}")


class RecyclingEvaluator:
    """Evaluates the recycling assistant's responses."""

    def __init__(self, verbose: bool = False):
        self.simplifier = ResponseSimplifier()
        self.server_manager = ServerManager(verbose=verbose)
        self.verbose = verbose

    async def evaluate_single_case(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single test case."""
        question = test_case["question"]
        expected_category = test_case["expected_category"].lower()

        if self.verbose:
            print(f"\n[VERBOSE] Evaluating: {question[:50]}...")
            print(f"[VERBOSE] Expected category: {expected_category}")
            print(f"[VERBOSE] Region: {test_case['region']}, Language: {test_case['language']}")

        try:
            # Check if server is running (don't start it automatically)
            if not self.server_manager.is_server_running():
                raise Exception("Server is not running. Please start the server manually first.")

            # Query the server
            if self.verbose:
                print("[VERBOSE] Querying server...")
            start_time = time.time()
            actual_response, usage = self.server_manager.query_server(question=question, region=test_case["region"])
            response_time = time.time() - start_time

            if self.verbose:
                print(f"[VERBOSE] Server response ({response_time:.2f}s): {actual_response[:100]}...")

            # Categorize the response
            categorized_response = self.simplifier.categorize_response(actual_response)

            if self.verbose:
                print(f"[VERBOSE] Categorized response: '{categorized_response}'")

            # Check if categorization is correct
            # Normalize both to English lowercase for comparison
            expected_normalized = expected_category.lower()
            if expected_normalized == "glas":
                expected_normalized = "glass (glas)"
            elif expected_normalized == "papier":
                expected_normalized = "paper (papier)"
            elif expected_normalized == "metall":
                expected_normalized = "metal (metall)"
            elif expected_normalized == "kunststoff":
                expected_normalized = "plastic (kunststoff)"
            elif expected_normalized == "sondermüll":
                expected_normalized = "hazardous (sondermüll)"
            else:
                # Handle English categories
                if expected_normalized == "glass":
                    expected_normalized = "glass (glas)"
                elif expected_normalized == "paper":
                    expected_normalized = "paper (papier)"
                elif expected_normalized == "plastic":
                    expected_normalized = "plastic (kunststoff)"
                elif expected_normalized == "metal":
                    expected_normalized = "metal (metall)"
                elif expected_normalized == "hazardous":
                    expected_normalized = "hazardous (sondermüll)"

            category_correct = categorized_response.lower() == expected_normalized

            if self.verbose:
                print(f"[VERBOSE] Category correct: {category_correct}")

            return {
                "question": question,
                "expected_category": expected_category,
                "actual_response": actual_response,
                "categorized_response": categorized_response,
                "category_correct": category_correct,
                "response_time": response_time,
                "region": test_case["region"],
                "language": test_case["language"],
                "category": test_case["expected_category"],
                "error": None,
            }

        except Exception as e:
            if self.verbose:
                print(f"[VERBOSE] Error evaluating case: {e}")
            return {"question": question, "expected_category": expected_category, "actual_response": "", "categorized_response": "", "category_correct": False, "response_time": 0, "region": test_case["region"], "language": test_case["language"], "category": test_case["expected_category"], "error": str(e)}

    async def evaluate_test_suite(self, test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate all test cases."""
        results = []

        print(f"Starting evaluation of {len(test_cases)} test cases...")
        if self.verbose:
            print(f"[VERBOSE] Verbose mode enabled - will show detailed progress")

        for i, test_case in enumerate(test_cases):
            if (i + 1) % 5 == 0 or self.verbose:  # Show progress every 5 cases or always in verbose mode
                print(f"Evaluated {i + 1}/{len(test_cases)} test cases...")

            result = await self.evaluate_single_case(test_case)
            results.append(result)

            # Show result summary in verbose mode
            if self.verbose:
                status = "✓" if result["category_correct"] else "✗"
                error_indicator = " (ERROR)" if result["error"] else ""
                print(f"[VERBOSE] Result {i+1}: {status} {result['categorized_response']}{error_indicator}")

            # Small delay to avoid overwhelming the system
            await asyncio.sleep(0.1)

        print(f"Evaluation complete! Processed {len(results)} test cases.")
        return results

    def calculate_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive statistics from evaluation results."""
        stats = {"overall": {}, "by_region": defaultdict(dict), "by_language": defaultdict(dict), "by_category": defaultdict(dict), "performance_metrics": {}}

        # Overall statistics
        total_cases = len(results)
        correct_cases = sum(1 for r in results if r["category_correct"])
        error_cases = sum(1 for r in results if r["error"] is not None)

        stats["overall"] = {"total_cases": total_cases, "correct_cases": correct_cases, "accuracy": correct_cases / total_cases if total_cases > 0 else 0, "error_cases": error_cases, "error_rate": error_cases / total_cases if total_cases > 0 else 0}

        # By region
        for region in ["US", "Germany"]:
            region_results = [r for r in results if r["region"] == region]
            if region_results:
                correct = sum(1 for r in region_results if r["category_correct"])
                stats["by_region"][region] = {"total": len(region_results), "correct": correct, "accuracy": correct / len(region_results)}

        # By language
        for language in ["en", "de"]:
            lang_results = [r for r in results if r["language"] == language]
            if lang_results:
                correct = sum(1 for r in lang_results if r["category_correct"])
                stats["by_language"][language] = {"total": len(lang_results), "correct": correct, "accuracy": correct / len(lang_results)}

        # By expected category
        categories = set(r["expected_category"] for r in results)
        for category in categories:
            cat_results = [r for r in results if r["expected_category"] == category]
            if cat_results:
                correct = sum(1 for r in cat_results if r["category_correct"])
                stats["by_category"][category] = {"total": len(cat_results), "correct": correct, "accuracy": correct / len(cat_results)}

        # Performance metrics
        successful_results = [r for r in results if r["error"] is None]
        if successful_results:
            response_times = [r["response_time"] for r in successful_results]
            stats["performance_metrics"] = {"avg_response_time": sum(response_times) / len(response_times), "min_response_time": min(response_times), "max_response_time": max(response_times), "total_response_time": sum(response_times)}

        return stats

    def save_results(self, results: List[Dict[str, Any]], stats: Dict[str, Any], filename: str = "evaluation_results.json"):
        """Save evaluation results and statistics."""
        output = {"results": results, "statistics": stats, "timestamp": time.time(), "version": "1.0"}

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"Results saved to {filename}")


def display_results_summary(results: List[Dict[str, Any]], stats: Dict[str, Any], assessment: str = None):
    """Display the evaluation results summary."""
    if assessment is None:
        # Calculate assessment if not provided
        accuracy = stats["overall"]["accuracy"]
        error_rate = stats["overall"]["error_rate"]

        if accuracy >= 0.8 and error_rate <= 0.1:
            assessment = "✅ EXCELLENT - High accuracy, low errors"
        elif accuracy >= 0.6 and error_rate <= 0.2:
            assessment = "🟡 GOOD - Acceptable performance"
        elif accuracy >= 0.4:
            assessment = "🟠 FAIR - Needs improvement"
        else:
            assessment = "❌ POOR - Significant issues detected"

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Assessment: {assessment}")
    print(f"Overall Accuracy: {stats['overall']['accuracy']:.2%}")
    print(f"Error Rate: {stats['overall']['error_rate']:.2%}")
    print(f"Total Cases: {stats['overall']['total_cases']}")
    print(f"Correct: {stats['overall']['correct_cases']}")
    print(f"Errors: {stats['overall']['error_cases']}")

    print("\nBy Region:")
    for region, data in stats["by_region"].items():
        print(f"  {region}: {data['accuracy']:.2%} ({data['correct']}/{data['total']})")

    print("\nBy Language:")
    for language, data in stats["by_language"].items():
        print(f"  {language}: {data['accuracy']:.2%} ({data['correct']}/{data['total']})")

    print("\nBy Category:")
    for category, data in stats["by_category"].items():
        print(f"  {category}: {data['accuracy']:.2%} ({data['correct']}/{data['total']})")

    if "performance_metrics" in stats:
        pm = stats["performance_metrics"]
        print("\nPerformance Metrics:")
        print(f"  Average Response Time: {pm['avg_response_time']:.3f}s")
        print(f"  Min Response Time: {pm['min_response_time']:.3f}s")
        print(f"  Max Response Time: {pm['max_response_time']:.3f}s")
        print(f"  Total Response Time: {pm['total_response_time']:.3f}s")

    # Show error analysis if there are errors
    error_results = [r for r in results if r["error"] is not None]
    if error_results:
        print(f"\nError Analysis ({len(error_results)} errors):")
        for i, error_result in enumerate(error_results[:5]):  # Show first 5 errors
            print(f"  {i+1}. {error_result['question'][:50]}... -> {error_result['error']}")
        if len(error_results) > 5:
            print(f"  ... and {len(error_results) - 5} more errors")

    # Show some incorrect answers for analysis
    incorrect_results = [r for r in results if not r["category_correct"] and r["error"] is None]
    if incorrect_results:
        print(f"\nSample Incorrect Answers ({min(3, len(incorrect_results))} shown):")
        for i, result in enumerate(incorrect_results[:3]):
            # Normalize expected category for display
            expected_display = result["expected_category"].lower()
            if expected_display == "glas":
                expected_display = "glass (Glas)"
            elif expected_display == "papier":
                expected_display = "paper (Papier)"
            elif expected_display == "metall":
                expected_display = "metal (Metall)"
            elif expected_display == "kunststoff":
                expected_display = "plastic (Kunststoff)"
            elif expected_display == "sondermüll":
                expected_display = "hazardous (Sondermüll)"
            else:
                # Handle English categories
                if expected_display == "glass":
                    expected_display = "glass (Glas)"
                elif expected_display == "paper":
                    expected_display = "paper (Papier)"
                elif expected_display == "plastic":
                    expected_display = "plastic (Kunststoff)"
                elif expected_display == "metal":
                    expected_display = "metal (Metall)"
                elif expected_display == "hazardous":
                    expected_display = "hazardous (Sondermüll)"

            print(f"  {i+1}. Q: {result['question'][:40]}...")
            print(f"      Expected: {expected_display} -> Got: {result['categorized_response'].lower()}")
            print(f"      Full response: {result['actual_response'][:60]}...")


def main():
    """Main function to run evaluation."""
    parser = argparse.ArgumentParser(description="Run evaluation for SusTech Recycling Agent")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--force", "-f", action="store_true", help="Force re-run evaluation even if results exist")
    args = parser.parse_args()

    results_file = "evaluation_results.json"

    # Check if results already exist
    if os.path.exists(results_file) and not args.force:
        print(f"✓ Found existing results in {results_file}")
        print("Loading previous evaluation results...")
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                saved_data = json.load(f)

            results = saved_data["results"]
            stats = saved_data["statistics"]
            timestamp = saved_data.get("timestamp", 0)

            # Show when the results were generated
            from datetime import datetime

            result_time = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            print(f"Results generated on: {result_time}")

            # Skip to summary display
            display_results_summary(results, stats)
            return

        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️  Could not load existing results ({e}), will run evaluation...")
    elif args.force:
        print("🔄 Force re-run requested, ignoring existing results...")
    else:
        print("ℹ No existing results found, running evaluation...")

    # Check if server is already running before starting evaluation
    temp_server_manager = ServerManager(verbose=args.verbose)
    if temp_server_manager.is_server_running():
        print("✓ Server is running at http://localhost:8000")
        print("Using existing server instance for evaluation")
    else:
        print("❌ Server is not running. Please start the server manually first.")
        print("Run 'make api' or 'poetry run uvicorn src.serve:app --reload' in the project root.")
        return

    # Load test cases
    try:
        with open("evaluation_test_cases.json", "r", encoding="utf-8") as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        print("Error: evaluation_test_cases.json not found. Run generate_test_suite.py first.")
        return

    # Initialize evaluator (always uses server)
    print("Using server mode for evaluation")
    if args.verbose:
        print("[VERBOSE] Verbose mode enabled")
    evaluator = RecyclingEvaluator(verbose=args.verbose)

    try:
        # Run evaluation
        results = asyncio.run(evaluator.evaluate_test_suite(test_cases))

        # Calculate statistics
        stats = evaluator.calculate_statistics(results)

        # Quick assessment
        accuracy = stats["overall"]["accuracy"]
        error_rate = stats["overall"]["error_rate"]

        if accuracy >= 0.8 and error_rate <= 0.1:
            assessment = "✅ EXCELLENT - High accuracy, low errors"
        elif accuracy >= 0.6 and error_rate <= 0.2:
            assessment = "🟡 GOOD - Acceptable performance"
        elif accuracy >= 0.4:
            assessment = "🟠 FAIR - Needs improvement"
        else:
            assessment = "❌ POOR - Significant issues detected"

        # Save results
        evaluator.save_results(results, stats)

        # Display results
        display_results_summary(results, stats, assessment)

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return


if __name__ == "__main__":
    main()

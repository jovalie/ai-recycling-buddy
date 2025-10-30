#!/usr/bin/env python3
"""
Evaluation Test Suite Generator for SusTech Recycling Agent

Generates comprehensive test cases for evaluating the AI recycling assistant's
accuracy across multiple recycling categories and languages.
"""

import json
import random
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class RecyclingItem:
    """Represents a recyclable item with its properties."""

    name: str
    category: str
    region: str  # 'US' or 'Germany'
    language: str  # 'en' or 'de'
    bin_type: str
    special_instructions: str = ""


class EvaluationTestSuite:
    """Generates and manages evaluation test cases."""

    def __init__(self):
        self.recycling_items = self._load_recycling_items()

    def _load_recycling_items(self) -> List[RecyclingItem]:
        """Load comprehensive recycling items for both US and Germany."""
        items = []

        # US Recycling Items
        us_items = [
            RecyclingItem("plastic bottle", "plastic", "US", "en", "recycling bin", "remove cap and label"),
            RecyclingItem("glass jar", "glass", "US", "en", "recycling bin", "remove lid"),
            RecyclingItem("aluminum can", "metal", "US", "en", "recycling bin", "rinse clean"),
            RecyclingItem("steel can", "metal", "US", "en", "recycling bin", "rinse clean"),
            RecyclingItem("tin can", "metal", "US", "en", "recycling bin", "rinse clean"),
            RecyclingItem("metal foil", "metal", "US", "en", "recycling bin", "rinse clean"),
            RecyclingItem("wire hanger", "metal", "US", "en", "recycling bin", "straighten"),
            RecyclingItem("newspaper", "paper", "US", "en", "recycling bin", "keep dry"),
            RecyclingItem("cardboard box", "paper", "US", "en", "recycling bin", "flatten"),
            RecyclingItem("milk carton", "paper", "US", "en", "recycling bin", "rinse clean"),
            RecyclingItem("plastic bag", "plastic", "US", "en", "landfill", "not recyclable in most areas"),
            RecyclingItem("styrofoam", "plastic", "US", "en", "landfill", "not recyclable"),
            RecyclingItem("pizza box with grease", "paper", "US", "en", "landfill", "contaminated"),
            RecyclingItem("light bulb", "hazardous", "US", "en", "special collection", "contains mercury"),
            RecyclingItem("battery", "hazardous", "US", "en", "special collection", "hazardous waste"),
            RecyclingItem("paint can", "hazardous", "US", "en", "special collection", "hazardous waste"),
            RecyclingItem("motor oil", "hazardous", "US", "en", "special collection", "hazardous waste"),
            RecyclingItem("fluorescent tube", "hazardous", "US", "en", "special collection", "contains mercury"),
            RecyclingItem("pesticide bottle", "hazardous", "US", "en", "special collection", "hazardous waste"),
            RecyclingItem("eggshell", "paper", "US", "en", "compost bin", "organic waste"),
            RecyclingItem("clothing", "plastic", "US", "en", "donation bin", "textile recycling"),
            RecyclingItem("takeout container", "plastic", "US", "en", "landfill", "check local guidelines"),
            RecyclingItem("CD", "plastic", "US", "en", "special collection", "electronics recycling"),
            RecyclingItem("DVD", "plastic", "US", "en", "special collection", "electronics recycling"),
            RecyclingItem("shredded paper", "paper", "US", "en", "compost bin", "organic waste"),
            RecyclingItem("green glass bottle", "glass", "US", "en", "recycling bin", "remove lid"),
            RecyclingItem("blue glass bottle", "glass", "US", "en", "recycling bin", "remove lid"),
            RecyclingItem("red glass bottle", "glass", "US", "en", "recycling bin", "remove lid"),
            RecyclingItem("yellow glass bottle", "glass", "US", "en", "recycling bin", "remove lid"),
            RecyclingItem("clear glass bottle", "glass", "US", "en", "recycling bin", "remove lid"),
        ]

        # German Recycling Items
        de_items = [
            RecyclingItem("Plastikflasche", "Kunststoff", "Germany", "de", "Gelber Sack", "Deckel abnehmen"),
            RecyclingItem("Glasflasche", "Glas", "Germany", "de", "Altglascontainer", "Pfand beachten"),
            RecyclingItem("Aludose", "Metall", "Germany", "de", "Gelber Sack", "ausspülen"),
            RecyclingItem("Blechdose", "Metall", "Germany", "de", "Gelber Sack", "ausspülen"),
            RecyclingItem("Konservendose", "Metall", "Germany", "de", "Gelber Sack", "ausspülen"),
            RecyclingItem("Alufolie", "Metall", "Germany", "de", "Gelber Sack", "zusammenknüllen"),
            RecyclingItem("Bügel", "Metall", "Germany", "de", "Gelber Sack", "gerade biegen"),
            RecyclingItem("Zeitung", "Papier", "Germany", "de", "Altpapier", "trocken halten"),
            RecyclingItem("Karton", "Papier", "Germany", "de", "Altpapier", "zusammenfalten"),
            RecyclingItem("Tetra Pak", "Papier", "Germany", "de", "Altpapier", "ausspülen"),
            RecyclingItem("Plastiktüte", "Kunststoff", "Germany", "de", "Restmüll", "nicht recycelbar"),
            RecyclingItem("Styropor", "Kunststoff", "Germany", "de", "Restmüll", "nicht recycelbar"),
            RecyclingItem("Pizzakarton mit Fett", "Papier", "Germany", "de", "Restmüll", "verschmutzt"),
            RecyclingItem("Glühbirne", "Sondermüll", "Germany", "de", "Sammelstelle", "Quecksilber enthält"),
            RecyclingItem("Batterie", "Sondermüll", "Germany", "de", "Sammelstelle", "Sondermüll"),
            RecyclingItem("Farbeimer", "Sondermüll", "Germany", "de", "Sammelstelle", "Sondermüll"),
            RecyclingItem("Motoröl", "Sondermüll", "Germany", "de", "Sammelstelle", "Sondermüll"),
            RecyclingItem("Leuchtstoffröhre", "Sondermüll", "Germany", "de", "Sammelstelle", "Quecksilber enthält"),
            RecyclingItem("Pestizidflasche", "Sondermüll", "Germany", "de", "Sammelstelle", "Sondermüll"),
            RecyclingItem("Eierschale", "Papier", "Germany", "de", "Biotonne", "Bioabfall"),
            RecyclingItem("Kleidung", "Kunststoff", "Germany", "de", "Textilsammlung", "Textilrecycling"),
            RecyclingItem("Takeawaybehälter", "Kunststoff", "Germany", "de", "Restmüll", "lokale Richtlinien prüfen"),
            RecyclingItem("CD", "Kunststoff", "Germany", "de", "Sammelstelle", "Elektroschrott"),
            RecyclingItem("DVD", "Kunststoff", "Germany", "de", "Sammelstelle", "Elektroschrott"),
            RecyclingItem("Papierschnipsel", "Papier", "Germany", "de", "Biotonne", "Bioabfall"),
            RecyclingItem("Grüne Glasflasche", "Glas", "Germany", "de", "Altglascontainer", "Deckel abnehmen"),
            RecyclingItem("Blaue Glasflasche", "Glas", "Germany", "de", "Altglascontainer", "Deckel abnehmen"),
            RecyclingItem("Rote Glasflasche", "Glas", "Germany", "de", "Altglascontainer", "Deckel abnehmen"),
            RecyclingItem("Gelbe Glasflasche", "Glas", "Germany", "de", "Altglascontainer", "Deckel abnehmen"),
            RecyclingItem("Klare Glasflasche", "Glas", "Germany", "de", "Altglascontainer", "Deckel abnehmen"),
        ]

        items.extend(us_items)
        items.extend(de_items)
        return items

    def generate_question_templates(self) -> Dict[str, List[str]]:
        """Generate question templates for different query types."""
        templates = {
            "en": [
                "How do I recycle {item}?",
                "Where does {item} go for recycling?",
                "Can I recycle {item}?",
                "What bin does {item} go in?",
                "How should I dispose of {item}?",
                "Is {item} recyclable?",
                "Where do I put {item}?",
                "How to recycle {item} properly?",
            ],
            "de": [
                "Wie recycelt man {item}?",
                "Wohin kommt {item} zum Recyceln?",
                "Kann man {item} recyceln?",
                "In welche Tonne kommt {item}?",
                "Wie entsorgt man {item}?",
                "Ist {item} recycelbar?",
                "Wohin kommt {item}?",
                "Wie recycelt man {item} richtig?",
            ],
        }
        return templates

    def generate_test_case(self, item: RecyclingItem, template: str) -> Dict[str, Any]:
        """Generate a single test case from item and template."""
        question = template.format(item=item.name)

        # Expected answer should contain the bin type and key instructions
        expected_answer = f"{item.bin_type}"
        if item.special_instructions:
            expected_answer += f" - {item.special_instructions}"

        return {"question": question, "expected_category": item.category, "expected_bin": item.bin_type, "expected_instructions": item.special_instructions, "region": item.region, "language": item.language, "item": item.name, "ground_truth": expected_answer}

    def generate_test_suite(self, num_cases_per_item: int = 3) -> List[Dict[str, Any]]:
        """Generate the complete test suite."""
        test_cases = []
        templates = self.generate_question_templates()

        for item in self.recycling_items:
            # Get templates for the item's language
            language_templates = templates.get(item.language, templates["en"])

            # Select random templates for this item
            selected_templates = random.sample(language_templates, min(num_cases_per_item, len(language_templates)))

            for template in selected_templates:
                test_cases.append(self.generate_test_case(item, template))

        # Shuffle the test cases
        random.shuffle(test_cases)
        return test_cases

    def save_test_suite(self, test_cases: List[Dict[str, Any]], filename: str = "evaluation_test_cases.json"):
        """Save test cases to JSON file."""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(test_cases, f, indent=2, ensure_ascii=False)

        print(f"Generated {len(test_cases)} test cases and saved to {filename}")

        # Print summary statistics
        regions = {}
        languages = {}
        categories = {}

        for case in test_cases:
            regions[case["region"]] = regions.get(case["region"], 0) + 1
            languages[case["language"]] = languages.get(case["language"], 0) + 1
            categories[case["expected_category"]] = categories.get(case["expected_category"], 0) + 1

        print("\nTest Suite Summary:")
        print(f"Regions: {regions}")
        print(f"Languages: {languages}")
        print(f"Categories: {categories}")


def main():
    """Main function to generate and save test suite."""
    suite = EvaluationTestSuite()
    test_cases = suite.generate_test_suite(num_cases_per_item=3)
    suite.save_test_suite(test_cases)


if __name__ == "__main__":
    main()

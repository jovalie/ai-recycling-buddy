# Evaluation Writeup for AI Recycling Agent

## Overview

This document presents an evaluation framework for assessing the accuracy of our AI-powered recycling guidance agent. The framework evaluates multilingual categorization performance across 180 realistic recycling scenarios to ensure comprehensive assessment of system capabilities and limitations.

## 1. Introduction

### 1.1 Evaluation Objectives
- **Accuracy Assessment**: Measure categorization accuracy across material types
- **Multilingual Evaluation**: Test performance in English and German contexts
- **Regional Specificity**: Evaluate adherence to US vs. German recycling regulations

## 2. Dataset Construction and Statistics

### 2.1 Dataset Overview
The evaluation dataset consists of 180 test cases derived from 60 unique recyclable items, with each item tested across 3 different question formulations. The dataset covers 5 material categories and spans two linguistic regions.

### 2.2 Item Categories and Distribution

| Category | English Name | German Name | US Items | German Items | Total Items | Test Cases |
|----------|-------------|-------------|----------|--------------|-------------|------------|
| Glass | glass | Glas | 6 | 6 | 12 | 36 |
| Paper | paper | Papier | 6 | 6 | 12 | 36 |
| Plastic | plastic | Kunststoff | 7 | 7 | 14 | 42 |
| Metal | metal | Metall | 5 | 5 | 10 | 30 |
| Hazardous | hazardous | Sondermüll | 6 | 6 | 12 | 36 |
| **Total** | | | **30** | **30** | **60** | **180** |


### 2.3 Question Template Distribution

The dataset employs 8 question templates per language, ensuring diverse query formulations:

**English Templates:**
1. "How do I recycle {item}?"
2. "Where does {item} go for recycling?"
3. "Can I recycle {item}?"
4. "What bin does {item} go in?"
5. "How should I dispose of {item}?"
6. "Is {item} recyclable?"
7. "Where do I put {item}?"
8. "How to recycle {item} properly?"

**German Templates:**
1. "Wie recycelt man {item}?"
2. "Wohin kommt {item} zum Recyceln?"
3. "Kann man {item} recyceln?"
4. "In welche Tonne kommt {item}?"
5. "Wie entsorgt man {item}?"
6. "Ist {item} recycelbar?"
7. "Wohin kommt {item}?"
8. "Wie recycelt man {item} richtig?"

### 2.4 Dataset Characteristics

**Linguistic Distribution:**
- English queries: 90 cases (50.0%)
- German queries: 90 cases (50.0%)

**Regional Distribution:**
- United States: 90 cases (50.0%)
- Germany: 90 cases (50.0%)

**Category Balance:**
- Glass: 36 cases (20.0%)
- Paper: 36 cases (20.0%)
- Plastic: 42 cases (23.3%)
- Metal: 30 cases (16.7%)
- Hazardous: 36 cases (20.0%)

### 2.5 Dataset Validation
- **Ground Truth Verification**: Expected categories validated against government-defined recycling guidelines
- **Regional Accuracy**: US guidelines from EPA, German guidelines from RAG knowledge base
- **Linguistic Consistency**: Bilingual terminology in German and English

## 3. Evaluation Methodology

### 3.1 Experimental Setup

#### System Architecture
The evaluation framework implements a client-server architecture:
- **Evaluation Client**: Python-based testing framework (`run_evaluation.py`)
- **Server Component**: FastAPI-based recycling assistant (`src/serve.py`)
- **Categorization Engine**: Rule-based keyword matching system (`ResponseSimplifier`)

#### Infrastructure Requirements
- **Hardware**: Standard workstation (8GB RAM, 4-core CPU)
- **Software**: Python 3.11+, FastAPI, Uvicorn, Poetry
- **Network**: Localhost HTTP communication
- **Timeout**: 60-second query timeout per test case

### 3.2 Categorization Algorithm

#### Keyword Taxonomy
To correlate complex generationsThe system employs a hierarchical keyword matching approach with differential weighting:

**Primary Keywords (Weight: 3.0)** - Core material identifiers

**Secondary Keywords (Weight: 1.5)** - Bin/container terminology

**German Compounds (Weight: 2.5)** - Complex German word formations

**Example 1: Glass Bottle Response**
```
AI Response: "Glass bottles should go in the recycling bin. Make sure to remove the lid first."

Keyword Analysis:
- "glass" (Primary Keyword): +3.0 points for glass category
- "bottles" (Secondary Keyword): +1.5 points for glass category  
- "recycling bin" (Secondary Keyword): +1.5 points for glass category

Total Scores:
- Glass: 3.0 + 1.5 + 1.5 = 5.5 ✓ (highest score)
- Other categories: 0.0

Result: Correctly categorized as "glass"
```


**Example 2: Tetra Pak Carton Response (Multi-Material Item)**
```
AI Response: "Tetra Pak cartons are made of layered cardboard, plastic, and aluminum. They should go in paper recycling or special collection depending on local rules."

Keyword Analysis:
- "cardboard" (Primary Keyword): +3.0 points for paper category
- "plastic" (Primary Keyword): +3.0 points for plastic category
- "aluminum" (Primary Keyword): +3.0 points for metal category
- "paper" (Primary Keyword): +3.0 points for paper category
- "recycling" (Secondary Keyword): +1.5 points for paper category, +1.5 points for plastic category, +1.5 points for metal category

Total Scores:
- Paper: 3.0 + 3.0 + 1.5 = 7.5 ✓ (highest score)
- Plastic: 3.0 + 1.5 = 4.5
- Metal: 3.0 + 1.5 = 4.5
- Other categories: Plastic (4.5), Metal (4.5)

Result: Correctly categorized as "paper" (highest score, despite multi-material composition)
```

**Example 3: Rechargeable Battery Response (Hazardous and Metal Properties)**
```
AI Response: "Rechargeable batteries contain valuable metals like nickel and lithium, but they are also hazardous waste. Check local regulations for proper disposal or recycling programs."

Keyword Analysis:
- "batteries" (Primary Keyword): +3.0 points for hazardous category
- "metals" (Primary Keyword): +3.0 points for metal category
- "nickel" (Secondary Keyword): +1.5 points for metal category
- "lithium" (Secondary Keyword): +1.5 points for metal category
- "hazardous" (Primary Keyword): +3.0 points for hazardous category
- "waste" (Secondary Keyword): +1.5 points for hazardous category
- "recycling" (Secondary Keyword): +1.5 points for metal category, +1.5 points for hazardous category

Total Scores:
- Hazardous: 3.0 + 3.0 + 1.5 + 1.5 = 9.0 ✓ (highest score)
- Metal: 3.0 + 1.5 + 1.5 + 1.5 = 7.5
- Other categories: Metal (7.5)

Result: Correctly categorized as "hazardous" (highest score, prioritizing safety over material value)
```

**Example 4: Rechargeable Lithium-Ion Battery Response (German Compounds with Multi-Category Scoring)**
```
AI Response: "Lithium-Ionen-Akkus enthalten wertvolle Metalle wie Nickel und Lithium, sind aber Sondermüll. Sie sollten zu Batterie-Sammelstellen gebracht werden."

Keyword Analysis:
- "Lithium-Ionen-Akkus" (German Compound): +2.5 points for hazardous category
- "wertvolle Metalle" (Primary Keyword "Metalle"): +3.0 points for metal category
- "Nickel" (Secondary Keyword): +1.5 points for metal category
- "Lithium" (Secondary Keyword): +1.5 points for metal category
- "Sondermüll" (German Compound): +2.5 points for hazardous category
- "Batterie-Sammelstellen" (German Compound): +2.5 points for hazardous category

Total Scores:
- Hazardous: 2.5 + 2.5 + 2.5 = 7.5 ✓ (highest score)
- Metal: 3.0 + 1.5 + 1.5 = 6.0
- Other categories: Metal (6.0)

Result: Correctly categorized as "hazardous" (German compounds reinforce hazardous classification despite metal content)
```

#### How `ResponseSimplifer` Works
The system looks for different types of words in the assistant's answer and gives them different point values:

- **Main material words** (like "glass," "plastic," "metal"): 3 points each
- **Bin/container words** (like "recycling bin," "blue bin"): 1.5 points each  
- **German compound words** (like "Glasflasche," "Aluminiumdosen"): 2.5 points each

`ResponseSimplifer` adds up all the points for each material type. Whichever material gets the most points determines what the `ResponseSimplifer` believes the agent is recommending.

#### Decision Threshold
- **Categorization Threshold**: score > 0.5 (prevents false positives)
- **Fallback Category**: "Unknown" for scores below threshold
- **Normalization**: Bilingual category mapping (e.g., "Glas" → "glass (Glas)")

### 3.3 Evaluation Metrics

#### Basic Performance Metrics
- **Accuracy**: Proportion of correctly categorized responses
- **Error Rate**: Proportion of failed server requests

## 4. Experimental Results

### 4.1 Overall Performance

**Accuracy**: 85.61% (154/180 correct categorizations)  

### 4.2 Category-wise Performance

| Category | Accuracy | Correct/Total |
|----------|----------|---------------|
| Glass (Glas) | 94.4% | 34/36 |
| Hazardous (Sondermüll) | 91.7% | 33/36 |
| Plastic (Kunststoff) | 78.6% | 33/42 |
| Paper (Papier) | 83.3% | 30/36 |
| Metal (Metall) | 66.7% | 20/30 |

### 4.3 Regional and Linguistic Analysis

**Regional Performance:**
- **United States**: 87.8% (79/90 correct)
- **Germany**: 83.3% (75/90 correct)

**Linguistic Performance:**
- **English**: 87.8% (79/90 correct)
- **German**: 83.3% (75/90 correct)

## 5. Reproducibility Information

### 7.1 Environment Setup

**System Requirements:**
```bash
# Python Environment
Python >= 3.11.0
Poetry >= 1.5.0
```

**Installation Commands:**
```bash
# Clone repository
git clone https://github.com/jovalie/recycling-agent.git
cd recycling-agent

# Install dependencies
poetry install

# Generate test suite
python src/evaluation/generate_test_suite.py

# Start server
make api

# Run evaluation
python src/evaluation/run_evaluation.py --verbose
```

### 7.2 Data Persistence
- **Test Cases**: JSON serialization in `evaluation_test_cases.json`
- **Results**: Timestamped JSON output in `evaluation_results.json`
- **Version Control**: Git-based experiment tracking

## 6. Limitations and Ethical Considerations

### 8.1 Technical Limitations

#### Categorization Constraints
- **Keyword Dependency**: Evaluation based on keyword coverage
- **Regional Variations**: May not capture all local recycling regulations, our knowledge base covers U.S. and Germany
- **Temporal Changes**: As recycling rules evolve over time, our knowledge base will need to be updated as regulation shifts

#### Evaluation Limitations
- **Dataset Scale**: 180 test cases may not capture all edge cases
- **Question Diversity**: Limited to 8 template types per language
- **Single-turn Evaluation**: Does not assess multi-turn conversations

### 8.2 Ethical Considerations

#### Environmental Impact
- **Accuracy Requirements**: Incorrect recycling guidance can harm environmental efforts
- **Bias Assessment**: System performance varies across geographic groups, scope of evaluation is limited to Germany and the United States
- **Accessibility**: While multilingual support improves inclusivity, we have only evaluated on English and German
- **Transparency**: Open-source evaluation framework may enable public scrutiny

#### Fairness Analysis
- **Regional Equity**: Balanced performance across US and German contexts
- **Linguistic Justice**: Equivalent support for English and German speakers
- **Category Balance**: Proportional evaluation across material types


### 8.3 Mitigation Strategies
- **Continuous Monitoring**: Regularly scheduled re-evaluation against updated regulations
- **User Feedback Integration**: Mechanisms for reporting incorrect guidance
- **Fallback Procedures**: Clear instructions for uncertain categorizations
- **Human Leadership**: Our system provides guidance. Humans make the impact.
This AI assistant offers information to support better recycling decisions, but environmental outcomes depend entirely on what we do with this information. While technology exists to provide guidance, ultimately real-world impact is the responsibility of humankind.

# PoI-AttributeConflation [SpQtr:2026]
Maps are built from many inputs – GPS providers, business listings, government registries and crowdsourced edits.  Each source might record different phone numbers, websites or hours for the same real‑world place.  This project aims to merge those conflicting attributes and find our Truth!

**Why Conflation Matters**

When different datasets describe the same place, their details rarely agree.  Typos, stale phone numbers, missing/duplicate websites all contribute to too many "Truths.”  **Picking a value at random isn’t good enough.**  We need a principled way to decide which information is most trustworthy and produce a single “golden” record.

This project tackles attribute conflation for points of interest.  **We start with pre‑matched place pairs, manually label the correct values, and then develop and test LLM algorithms for selecting the best attributes.**

**Mission and Deliverables**
	1.	Build a ground‑truth dataset – inspect matched place pairs and label the correct website, email, phone and other attributes.
	2.	Design selection logic – create heuristics or train a model that picks the most reliable attribute when sources disagree, factoring in recency, completeness and agreement across sources.
	3.	Measure results – define metrics such as accuracy and coverage, compare rule‑based versus machine‑learning methods, and document where each works best.

**The outputs are:** 
• A high‑quality labelled dataset for training and evaluation. 
• A working selection algorithm ready to embed into larger Project Terra pipelines. 
• A short report summarizing performance and lessons learned.

**How to Proceed**  
	_1.	Label the data._ Create a CSV of matched places with the correct attribute for each field.  Establish formatting guidelines so that values are consistent.   
	2.	_Analyze_: Count how many sources agree on a value, check how recent each source is, measure completeness of the field and compute string similarity to catch typos.   
	3.	_Address the logic_: Start with a simple rule set (e.g. “if multiple sources agree, prefer that; if tie, pick the most recent; otherwise choose the longer string”).  **Then experiment with a machine‑learning model trained on our datasets & analysis to predict the correct value.**  
	4.	_Evaluate_. Split the labelled data into training and test sets.  Compare rule‑based and ML approaches using accuracy, precision and recall.  Look at error cases to guide improvements.

**License**

This project is released under the Apache 2.0 License.  See LICENSE￼ for details.

Acknowledgements

This work is meant to help real‑world challenges for conflating place attributes during Project Terra! We are peons just trying to improve the quality of the golden record.  

#Sign your name under here! 
-Allan Dewey [Dropped]
-Anthony Martinez [Certified Dork]



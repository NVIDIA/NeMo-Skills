# Reproducing the OpenScience Dataset collection

This recipe contains the scripts and prompts to reproduce the **OpenScience** dataset, as described in the paper "A Scalable LLM Framework for Seed-Free Synthetic Data Generation".

## What We Share

This repository provides all the necessary components to run the data generation recipe:

* **Prompts for Data Augmentation**: Templates for generating "similar" and "inspired-by" questions to expand dataset.
* **Prompts for Question Generation**: Templates for creating multiple-choice questions (MCQs) with either four or ten answer options.
* **Prompt for Subtopic Expansion**: The template used to generate a list of subtopics from.
* **Solution Filtering Script**: A script to filter generated solutions based on majority voting, as discussed in our paper.

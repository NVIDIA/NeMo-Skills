import argparse
import json
import glob
import re
from collections import defaultdict
from typing import Dict, List, Any, Optional

def process_fragments(novelty_files: List[str], significance_files: List[str], output_file: str) -> None:
    """
    Process all files in parallel, one line at a time.
    
    Args:
        novelty_files: List of files with novelty judgments
        significance_files: List of files with significance judgments
        output_file: Path to output file for filtered entries
    """
    # Open all files
    novelty_handles = [open(path, 'r', encoding='utf-8') for path in novelty_files]
    significance_handles = [open(path, 'r', encoding='utf-8') for path in significance_files]
    
    try:
        with open(output_file, "w") as fout:
            current_orig_idx = None
            novelty_fragments = defaultdict(list)  # {fragment_index: fragments}
            significance_fragments = defaultdict(list)  # {fragment_index: fragments}
            total_filtered = 0
            
            # Read all files line by line
            line_num = 0
            while True:                
                # Read and parse novelty fragments
                novelty_batch = []
                for file_handle in novelty_handles:
                    line = file_handle.readline()
                    try:
                        fragment = json.loads(line)
                        novelty_batch.append(fragment)
                    except json.JSONDecodeError:
                        break
                
                # Read and parse significance fragments
                significance_batch = []
                for file_handle in significance_handles:
                    line = file_handle.readline()
                    try:
                        fragment = json.loads(line)
                        significance_batch.append(fragment)
                    except json.JSONDecodeError:
                        break
                    
                fragment_orig_idx = novelty_batch[0]['index']
                fragment_idx = novelty_batch[0]['fragment_index']
                
                # Ensure all fragments in this batch have the same index
                if not all(f['index'] == fragment_orig_idx for f in novelty_batch + significance_batch):
                    raise ValueError(f"Misaligned fragments at line {line_num}: not all fragments have the same index")
                
                # Ensure all fragments in this batch have the same fragment_index
                if not all(f['fragment_index'] == fragment_idx for f in novelty_batch + significance_batch):
                    raise ValueError(f"Misaligned fragments at line {line_num}: not all fragments have the same fragment_index")
                
                # Process data when original index changes
                if current_orig_idx is not None and fragment_orig_idx != current_orig_idx:
                    # Process fragments for the previous original index
                    filtered_entry = process_single_index(
                        current_orig_idx, 
                        novelty_fragments, 
                        significance_fragments
                    )
                    
                    # Write the filtered entry if it passes criteria
                    if filtered_entry:
                        fout.write(json.dumps(filtered_entry) + "\n")
                        total_filtered += 1
                    
                    # Clear the collected fragments
                    novelty_fragments.clear()
                    significance_fragments.clear()
                
                # Update current index
                current_orig_idx = fragment_orig_idx
                
                # Collect fragments
                for fragment in novelty_batch:
                    novelty_fragments[fragment_idx].append(fragment)
                
                for fragment in significance_batch:
                    significance_fragments[fragment_idx].append(fragment)
                
                line_num += 1
            
            # Process the last batch of fragments
            if current_orig_idx is not None:
                filtered_entry = process_single_index(
                    current_orig_idx, 
                    novelty_fragments, 
                    significance_fragments
                )
                
                if filtered_entry:
                    fout.write(json.dumps(filtered_entry) + "\n")
                    total_filtered += 1
        
        print(f"Filtered {total_filtered} entries based on combined novelty and significance criteria.")
        print(f"Applied criteria: at least one significant novel fragment OR ≥0.5 moderate novel fragments")
    
    finally:
        # Close all file handles
        for file_handle in novelty_handles + significance_handles:
            file_handle.close()

def process_single_index(
    orig_idx: int, 
    novelty_dict: Dict[int, List[Dict]], 
    significance_dict: Dict[int, List[Dict]]
) -> Optional[Dict[str, Any]]:
    """
    Process fragments for a single original index.
    
    Args:
        orig_idx: The original index being processed
        novelty_dict: Dictionary mapping fragment indices to novelty judgments
        significance_dict: Dictionary mapping fragment indices to significance judgments
        
    Returns:
        Filtered entry if it passes criteria, None otherwise
    """
    # Get all unique fragment indices
    all_frag_indices = set(list(novelty_dict.keys()) + list(significance_dict.keys()))
    
    # Track combined classifications
    fragment_classifications = []
    fragment_details = []
    
    # Process each fragment
    for frag_idx in sorted(all_frag_indices):
        # Get judgments for this fragment
        novelty_fragments = novelty_dict.get(frag_idx, [])
        significance_fragments = significance_dict.get(frag_idx, [])
        
        # Determine novelty classification
        novel_count = 0
        novelty_total = len(novelty_fragments)
        
        for fragment in novelty_fragments:
            judgements = re.findall(r"judgement: (verification|novel calculation)\n", 
                                   fragment["fragment_novelty"].lower())
            if judgements and "novel calculation" in judgements:
                novel_count += 1
        
        # Fragment is novel if all judgments agree
        is_novel = novel_count == novelty_total
        
        # Determine significance classification
        trivial_count = 0
        moderate_count = 0
        significant_count = 0
        significance_total = len(significance_fragments)
        
        for fragment in significance_fragments:
            significance = re.search(r"significance: (trivial|moderate|significant)", 
                                    fragment["fragment_significance"].lower())
            
            if significance:
                significance_level = significance.group(1)
                
                if significance_level == "trivial":
                    trivial_count += 1
                elif significance_level == "moderate":
                    moderate_count += 1
                elif significance_level == "significant":
                    significant_count += 1
        
        # Apply the criteria for fragment significance
        significance_level = "trivial"  # default
        
        if trivial_count > 0:
            significance_level = "trivial"
        elif moderate_count > 0:
            significance_level = "moderate"
        elif significant_count == significance_total:
            significance_level = "significant"
        
        # Combined classification
        classification = (is_novel, significance_level)
        fragment_classifications.append(classification)
        
        # Store detailed information
        fragment_details.append({
            "fragment_index": frag_idx,
            "is_novel": is_novel,
            "novel_votes": novel_count,
            "novelty_total_votes": novelty_total,
            "significance": significance_level,
            "trivial_votes": trivial_count,
            "moderate_votes": moderate_count,
            "significant_votes": significant_count,
            "significance_total_votes": significance_total,
            # Include the first fragment for reference
            "novelty_judgement": novelty_fragments[0]["fragment_novelty"] if novelty_fragments else "",
            "significance_judgement": significance_fragments[0]["fragment_significance"] if significance_fragments else ""
        })
    
    # Apply combined filter criteria:
    # - At least one significant novel fragment
    # - OR >= 0.5 moderate novel fragments
    
    significant_novel_count = sum(1 for is_novel, significance in fragment_classifications 
                                if is_novel and significance == "significant")
    
    moderate_novel_count = sum(1 for is_novel, significance in fragment_classifications 
                              if is_novel and significance == "moderate")
    
    total_fragments = len(fragment_classifications)
    
    # Apply filtering criteria
    should_keep = (significant_novel_count >= 1 or 
                  (moderate_novel_count >= 0.5 * total_fragments if total_fragments > 0 else False))
    
    # If it doesn't pass the criteria, return None
    if not should_keep or total_fragments == 0:
        return None
    
    # Create the filtered entry
    # Get a reference fragment from either novelty or significance
    first_fragment = None
    if novelty_dict and next(iter(novelty_dict.values())):
        first_fragment = next(iter(novelty_dict.values()))[0]
    elif significance_dict and next(iter(significance_dict.values())):
        first_fragment = next(iter(significance_dict.values()))[0]
    
    if not first_fragment:
        return None
    
    # Create the filtered entry
    original_entry = {}
    
    # Copy all fields except the excluded ones
    excluded_fields = ['fragment', 'fragment_index', 'fragment_novelty', 
                     'fragment_significance', 'index', 'original_generation']
    for k, v in first_fragment.items():
        if k not in excluded_fields:
            original_entry[k] = v
    
    # Add the original 'generation' field back if available
    if 'original_generation' in first_fragment:
        original_entry['generation'] = first_fragment['original_generation']
    
    # Add classifier information
    original_entry['significant_novel_count'] = significant_novel_count
    original_entry['moderate_novel_count'] = moderate_novel_count
    original_entry['total_fragments'] = total_fragments
    original_entry['fragment_details'] = fragment_details
    
    # Add flags for quick filtering
    original_entry['has_significant_novel'] = significant_novel_count >= 1
    original_entry['moderate_novel_ratio'] = moderate_novel_count / total_fragments if total_fragments > 0 else 0
    
    # Mark as correct
    original_entry['is_correct'] = True
    
    return original_entry

def main(args):
    # Get all matching fragment files
    novelty_files = glob.glob(args.novelty_files)
    significance_files = glob.glob(args.significance_files)
    
    if not novelty_files:
        print(f"No novelty files found matching pattern: {args.novelty_files}")
        return
        
    if not significance_files:
        print(f"No significance files found matching pattern: {args.significance_files}")
        return
    
    # Process all fragments in parallel
    process_fragments(novelty_files, significance_files, args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter code based on combined novelty and significance judgments")
    parser.add_argument("--novelty_files", type=str, required=True, 
                        help="Input file pattern for novelty judgment files")
    parser.add_argument("--significance_files", type=str, required=True,
                        help="Input file pattern for significance judgment files")
    parser.add_argument("--output_file", type=str, required=True, 
                        help="Output file for filtered entries")
    args = parser.parse_args()

    main(args)
